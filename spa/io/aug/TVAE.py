'''
Tabular VAE
'''

import pandas as pd
from ctgan import TVAE

def expand_dataset(X,y,nobs,
                   embedding_dim=128,compress_dims=(128, 128),decompress_dims=(128, 128),
                   epochs = 20, cuda=None, verbose=True):
    import torch
    from ...device import resolve_device

    device = resolve_device(cuda)            # CUDA -> MPS -> CPU
    # ctgan's TVAE reliably supports only CUDA/CPU, so pass a plain bool
    # (True only when a CUDA device is selected); otherwise it runs on CPU.
    tvae_cuda = (device.type == 'cuda')

    train_data=pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=[-1])], axis=1)
    train_data.columns = train_data.columns.astype(str)
    label_col = train_data.columns[-1]            # the y column
    tvae = TVAE(epochs=epochs, 
                embedding_dim=embedding_dim,compress_dims=compress_dims,decompress_dims=decompress_dims,
                cuda=tvae_cuda, verbose = verbose)

    # The label MUST be declared discrete, otherwise TVAE treats it as a
    # continuous column and ``sample`` returns float "labels" (e.g. 0.37).
    tvae.fit(train_data, discrete_columns=[label_col])

    if verbose: # show model structure

        from torchviz import make_dot
        import IPython.display

        dev = getattr(tvae, '_device', torch.device('cpu'))
        input_vec = torch.zeros(1, tvae.embedding_dim, dtype=torch.float, requires_grad=False).to(dev)
        IPython.display.display('<b>TVAE decoder</b>')
        IPython.display.display(make_dot(tvae.decoder(input_vec)))

    synthetic_data = tvae.sample(nobs)
    X_new = synthetic_data.iloc[:, :-1]
    y_new = synthetic_data.iloc[:, -1]

    return X_new, y_new