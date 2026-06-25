import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def expand_dataset(X, y, nobs, X_names=None, cuda=None, verbose=True):
    '''
    Generate ``nobs`` synthetic samples per class with a VAE.

    Returns the SYNTHETIC samples only (consistent with the other generators in
    this package). The previous version returned "original + synthetic", which
    duplicated the real data once it was re-added downstream.
    '''
    if not X_names:
        X_names = list(range(X.shape[1]))

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    X_new_all = []
    y_new_all = []
    for label in np.unique(y):
        X_grp = X[y == label]
        new_data = create_random_samples(X_grp, sample_size=nobs, cuda=cuda, verbose=verbose)
        X_new_all.append(new_data)
        y_new_all.extend([label] * nobs)

    Xn = pd.DataFrame(np.vstack(X_new_all), columns=X_names)
    yn = pd.Series(y_new_all, name='label')
    return Xn, yn


def create_random_samples(X, sample_size=1, batch_size=32, h_dim=200, z_dim=10,
                          save_path=None, cuda=None, verbose=True):
    '''
    Train a VAE model with one hidden layer and generate random samples from its decoder.

    Parameters
    ----------
    X : dataset.
    sample_size : how many samples to generate.
    batch_size : batch size for training.
    h_dim : hidden layer dimension.
    z_dim : latent dimension.
    save_path : path to save the trained model.
    cuda : None -> auto-detect GPU; True/False -> force. Keeps the code runnable
        on CPU / Apple-Silicon machines (the old code hard-coded ``.cuda()``).
    '''

    from cs1.basis.adaptive.vae import train_vae  # cs1 version should >= 0.2.2
    import torch
    from ...device import resolve_device

    device = resolve_device(cuda)   # CUDA -> MPS -> CPU

    scaler = StandardScaler()  # MinMaxScaler() # StandardScaler()
    X = scaler.fit_transform(X)

    n = X.shape[1]
    h_dim1 = h_dim
    h_dim2 = 0

    model = train_vae(X, batch_size=batch_size,
                      h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim, 
                      verbose=verbose)
    model = model.to(device)

    if save_path is not None and save_path != '':
        if not save_path.endswith('.pth'):
            save_path = save_path + '.pth'
        torch.save(model.state_dict(), save_path)

    if verbose: # show model structure

        from torchviz import make_dot
        import IPython.display

        input_vec = torch.zeros(1, n, dtype=torch.float, requires_grad=False).to(device)
        IPython.display.display(make_dot(model(input_vec)))

    with torch.no_grad():
        # Generating random z in the representation space
        z = torch.randn(sample_size, z_dim).to(device)
        # Evaluating the decoder on each of them
        sample = model.decoder(z).cpu().numpy()

    return scaler.inverse_transform(sample)