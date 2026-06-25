'''
Vanilla (fully-connected) GAN for 1-D spectroscopic profiling data.

PyTorch implementation (the previous version was TensorFlow/Keras). Standard
non-saturating BCE adversarial training, device selected with the
CUDA -> MPS -> CPU priority. Inputs are min-max scaled to [0, 1] (the generator
ends with a Sigmoid) and mapped back / clipped to non-negative on output.
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from ...device import resolve_device


class Generator(nn.Module):
    '''noise -> fully-connected stack -> feature_dim (Sigmoid, data in [0,1]).'''

    def __init__(self, noise_dim, feature_dim, h1=128, h2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(h2, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    '''feature_dim -> fully-connected stack -> real/fake logit (BCEWithLogits).'''

    def __init__(self, feature_dim, h1=256, h2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_gan(X, noise_dim=100, epochs=500, batch_size=16, lr=2e-4,
              device=None, verbose=True):
    '''Train a vanilla GAN on a single (already scaled) class matrix X.'''
    device = resolve_device(device)
    feature_dim = X.shape[1]

    data = torch.tensor(np.asarray(X), dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True)

    G = Generator(noise_dim, feature_dim).to(device)
    D = Discriminator(feature_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        g_loss = d_loss = 0.0
        for (real,) in loader:
            b = real.size(0)
            if b < 2:                      # BatchNorm in G needs >1 sample
                continue
            real = real.to(device)
            ones = torch.ones(b, 1, device=device)
            zeros = torch.zeros(b, 1, device=device)

            # --- train discriminator ---
            z = torch.randn(b, noise_dim, device=device)
            fake = G(z)
            lossD = criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
            optD.zero_grad()
            lossD.backward()
            optD.step()

            # --- train generator (non-saturating) ---
            lossG = criterion(D(fake), ones)
            optG.zero_grad()
            lossG.backward()
            optG.step()

            g_loss, d_loss = float(lossG), float(lossD)

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f'Epoch {epoch + 1}/{epochs}  G {g_loss:.4f}  D {d_loss:.4f}')

    return G, D, device


def expand_dataset(X, y, nobs, X_names=None, epochs=500, batch_size=16,
                   noise_dim=100, cuda=None, verbose=True):
    '''
    Generate ``nobs`` synthetic samples per class with a vanilla GAN.
    Returns the SYNTHETIC samples only (X_new, y_new).
    '''
    if not X_names:
        X_names = list(range(X.shape[1]))

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    device = resolve_device(cuda)

    synth_data = pd.DataFrame()
    for label in np.unique(y):
        X0 = X[y == label]

        # scale to [0, 1] for stable adversarial training
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(X0)

        G, _, device = train_gan(train_data, noise_dim=noise_dim, epochs=epochs,
                                 batch_size=batch_size, device=device, verbose=verbose)

        G.eval()
        with torch.no_grad():
            z = torch.randn(nobs, noise_dim, device=device)
            generated = G(z).cpu().numpy()
        generated = np.clip(scaler.inverse_transform(generated), 0, None)  # original scale, non-negative

        df = pd.DataFrame(generated, columns=X_names)
        df['label'] = [label] * len(df)
        synth_data = pd.concat([synth_data, df], axis=0)

        if verbose:
            try:
                from torchviz import make_dot
                import IPython.display
                dummy = torch.zeros(2, noise_dim, device=device)
                IPython.display.display(IPython.display.HTML('<b>GAN generator</b>'))
                IPython.display.display(make_dot(G(dummy)))
            except Exception:
                pass

    synth_data.reset_index(drop=True, inplace=True)
    return synth_data.iloc[:, :-1], synth_data.iloc[:, -1]
