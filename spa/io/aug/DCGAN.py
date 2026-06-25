'''
DCGAN (Deep Convolutional GAN) for 1-D spectroscopic profiling data.

PyTorch implementation (the previous version was TensorFlow/Keras). It uses 1-D
(transpose-)convolutions, BatchNorm and LeakyReLU following the DCGAN recipe,
trains with the standard (non-saturating) BCE adversarial loss, and selects the
device with the CUDA -> MPS -> CPU priority.

Inputs are min-max scaled to [0, 1] before training (the generator ends with a
Sigmoid) and mapped back / clipped to be non-negative on output, which keeps the
adversarial training stable for raw spectra whose magnitudes span many orders.
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from ...device import resolve_device


def _conv_out(length, kernel=4, stride=2, padding=1):
    return (length + 2 * padding - kernel) // stride + 1


class Generator(nn.Module):
    '''noise -> (transpose-)conv stack -> linear projection to feature_dim.'''

    def __init__(self, noise_dim, feature_dim, base_len=16, base_ch=128):
        super().__init__()
        self.base_len = base_len
        self.base_ch = base_ch
        self.fc = nn.Linear(noise_dim, base_ch * base_len)
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(base_ch, base_ch // 2, 4, 2, 1, bias=False),   # x2
            nn.BatchNorm1d(base_ch // 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(base_ch // 2, base_ch // 4, 4, 2, 1, bias=False),  # x2
            nn.BatchNorm1d(base_ch // 4),
            nn.ReLU(True),
            nn.Conv1d(base_ch // 4, 1, 3, 1, 1),
        )
        self.out = nn.Linear(base_len * 4, feature_dim)   # exact mapping to feature_dim
        self.act = nn.Sigmoid()                            # data scaled to [0, 1]

    def forward(self, z):
        x = self.fc(z).view(z.size(0), self.base_ch, self.base_len)
        x = self.conv(x).flatten(1)
        return self.act(self.out(x))


class Discriminator(nn.Module):
    '''1-D conv classifier; outputs a real/fake logit (use BCEWithLogitsLoss).'''

    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        l2 = _conv_out(_conv_out(feature_dim))
        assert l2 >= 1, 'feature_dim too small for the 1-D DCGAN discriminator'
        self.fc = nn.Linear(64 * l2, 1)

    def forward(self, x):
        x = x.unsqueeze(1)                 # (B, 1, feature_dim)
        x = self.net(x).flatten(1)
        return self.fc(x)                  # logits


def train_dcgan(X, noise_dim=100, epochs=500, batch_size=16, lr=2e-4,
                device=None, verbose=True):
    '''Train a DCGAN on a single (already scaled) class matrix X.'''
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
            if b < 2:                      # BatchNorm needs >1 sample
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
    Generate ``nobs`` synthetic samples per class with a 1-D DCGAN.

    Returns the SYNTHETIC samples only (X_new, y_new) as DataFrames/Series,
    consistent with the other generators in this package.
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

        G, _, device = train_dcgan(train_data, noise_dim=noise_dim, epochs=epochs,
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
                IPython.display.display(IPython.display.HTML('<b>DCGAN generator</b>'))
                IPython.display.display(make_dot(G(dummy)))
            except Exception:
                pass

    synth_data.reset_index(drop=True, inplace=True)
    return synth_data.iloc[:, :-1], synth_data.iloc[:, -1]
