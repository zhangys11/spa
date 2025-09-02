'''
Tabular VAE
'''

import pandas as pd
from ctgan import TVAE

def expand_dataset(X,y,nobs,
                   embedding_dim=128,compress_dims=(128, 128),decompress_dims=(128, 128),
                   epochs = 20, cuda=True, verbose=True):
    train_data=pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=[-1])], axis=1)
    train_data.columns = train_data.columns.astype(str)
    tvae = TVAE(epochs=epochs, 
                embedding_dim=embedding_dim,compress_dims=compress_dims,decompress_dims=decompress_dims,
                cuda=cuda, verbose = verbose)
    
    tvae.fit(train_data)

    if verbose: # show model structure

        import torch
        from torchviz import make_dot
        import IPython.display

        input_vec = torch.zeros(1, tvae.embedding_dim, dtype=torch.float, requires_grad=False).to('cuda')
        IPython.display.display('<b>TVAE decoder</b>')
        IPython.display.display(make_dot(tvae.decoder(input_vec)))

    synthetic_data = tvae.sample(nobs)
    X_new = synthetic_data.iloc[:, :-1]
    y_new = synthetic_data.iloc[:, -1]

    return X_new, y_new