'''
Dataset augmentation methods
'''
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from ...vis import plot_components_2d
from . import SMOTE, Gaussian, GMM, KDE, metrics

def upsample(target_path, X, y, X_names, method = 'SMOTE', folds = 3, d = 0.5, 
embedding_dim = 128, generator_dim = (256, 256), discriminator_dim = (256, 256),
epochs = 10, batch_size = 100, cuda = True, display = False, verbose = True):
    '''
    Upsample a dataset by SMOTE, KDE, GMM, VAE, TVAE, GAN, DCGAN, ctGAN, etc.

    Parameters
    ----------
    target_path : where to save the generated dataset. 
        If None or empty string or False, skip saving.
    X_names : the labels for each X feature
    folds : expand to N folds
    d : sampling distance in SMOTE
    embedding_dim, generator_dim, discriminator_dim, epochs, batch_size, cuda : gan-family-specific params
    '''

    if folds == 0:
        return X, y
    
    nobs = folds*len(X)

    if method == 'SMOTE':
        Xn, yn = SMOTE.expand_dataset(X, y, d, folds)
    elif method == 'Gaussian' or method=='gaussian':
        Xn, yn = Gaussian.expand_dataset(X, y, nobs)
    elif method == 'GMM' or method=='gmm':
        Xn, yn = GMM.expand_dataset(X, y, nobs,verbose=verbose)
    elif method == 'KDE':
        Xn, yn = KDE.expand_dataset(X, y, nobs)
    elif method == 'MDN':
        from . import MDN
        Xn, yn = MDN.expand_dataset(X, y, nobs, epochs=epochs, verbose=verbose)
    elif method == 'VAE':
        from . import VAE
        Xn, yn = VAE.expand_dataset(X, y, nobs, X_names, verbose=verbose)
    elif method == 'TVAE':
        from . import TVAE
        Xn, yn = TVAE.expand_dataset(X, y, nobs, epochs=epochs, verbose=verbose)
    elif method == 'GAN':
        from . import GAN
        Xn, yn = GAN.expand_dataset(X, y, nobs, epochs=epochs, batch_size=batch_size, noise_dim=100, X_names=X_names, verbose=verbose)
    elif method == 'DCGAN':
        from . import DCGAN
        Xn, yn = DCGAN.expand_dataset(X, y, nobs, epochs=epochs, batch_size=batch_size, noise_dim=100, X_names=X_names, verbose=verbose)
    elif method == 'WGAN':
        from . import WGAN
        Xn, yn = WGAN.expand_dataset(X, y, nobs, epochs=epochs, batch_size=batch_size, noise_dim=100, X_names=X_names, verbose=verbose)
    elif method == 'ctGAN' or method == 'CTGAN':
        from . import ctGAN
        Xn, yn = ctGAN.expand_dataset(X, y, nobs,
                                      embedding_dim = embedding_dim, generator_dim = generator_dim, discriminator_dim = discriminator_dim,
                                      epochs=epochs, batch_size=batch_size, cuda = cuda, verbose = verbose)
    else:
        print('Unsupported method: ' + method )
        return X, y

    if target_path:
        
        dfX = pd.DataFrame(Xn)
        dfX.columns = X_names

        dfY = pd.DataFrame(yn)
        dfY.columns = ['label']

        df = pd.concat([dfY, dfX], axis=1)
        df = df.sort_values(by=['label'], ascending=True)
        df.to_csv(target_path, index=False)  # don't create the index column

    if display:

        pca = PCA(n_components=2) # keep the first 2 components
        X_pca = pca.fit_transform(Xn)
        plot_components_2d(X_pca, yn)
        plt.title('PCA of the augmented dataset')
        plt.show()

    # convert pandas dataframe to numpy array
    if isinstance(Xn, pd.DataFrame):
        Xn = Xn.values
    if isinstance(yn, pd.DataFrame):
        yn = yn.values

    return Xn, yn

def upsample_tryall(X, y, X_names,
                    methods = ['Gaussian', 'GMM', 'KDE', 'MDN', 'VAE', 'TVAE', 'GAN', 'DCGAN', 'WGAN', 'CTGAN'], 
                    target_label=None, folds = 3, d = 0.5, output_dir = './output',
                    embedding_dim = 128, generator_dim = (256, 256), discriminator_dim = (256, 256),
                    epochs = 10, batch_size = 100, cuda = True, display = False, verbose = True):

    if isinstance(target_label, int):
        Xs = X[np.where(y == target_label)]
        ys = y[np.where(y == target_label)]
    else:
        Xs = X
        ys = y

    from datetime import datetime
    ts = datetime.today().strftime('%Y%m%d%H')
    
    X_synth = []
    y_synth = []

    X_aug = [X]
    y_aug = [y]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model in methods:
        
        if verbose:
            print('--- Data Augmentation using', model, '---')

        X_new, y_new = upsample(output_dir + '/' + model + '_' + ts + '.csv', Xs, ys, X_names, method=model, folds=folds, d=d,
                                embedding_dim=embedding_dim, generator_dim=generator_dim, discriminator_dim=discriminator_dim,
                                epochs=epochs, batch_size=batch_size, cuda=cuda, display=display, verbose=verbose)
        X_synth.append(X_new)
        y_synth.append(y_new)
        X_aug.append(np.vstack((X_new, X))) # synth data + original data
        y_aug.append(np.hstack((y_new, y))) # synth data + original data


    baseline = pd.concat([pd.DataFrame(X, columns=X_names),pd.DataFrame(y,columns=['label'])],axis=1)
    
    if verbose:
        print('--- Calculating Similarity Metrics ---')
    result_s = metrics.evaluate_fidelity(X_synth, y_synth, methods, baseline, verbose=verbose)
    
    if verbose:
        print('--- Calculating Classification Metrics ---')
    result_c = metrics.evaluate_classification(X_aug, y_aug, ['Original'] + methods, verbose=verbose)

    return result_s, result_c


def visualize_similarity_result(result_s):
    '''
    Visualize the similarity metrics

    Example
    -------
    from spa.io import aug
    result_s, result_c = aug.upsample_tryall(X, y, ...)
    visualize_similarity_result(result_s)
    '''
    
    df=pd.DataFrame(result_s)
    df = df.apply(lambda x: round(x,3))
    display(HTML(df.to_html()))

def visualize_cls_result(result_c):
    '''
    Visualize the classification metrics

    Example
    -------
    from spa.io import aug
    result_s, result_c = aug.upsample_tryall(X, y, ...)
    visualize_cls_result(result_c)
    '''

    def _round(x): # a column-wise round function
        if isinstance(x,list):
            return list(np.round(x,3))
        elif isinstance(x,float):
            return round(x,3)
    
    df=pd.DataFrame()
    for k,v in result_c.items():
        v['model']=len(v)*[k]
        df=pd.concat([df,v],axis=0)
    df=df.reset_index(names='metrics')
    b_col = df.pop('model')
    df.insert(1, 'model', b_col)
    df=df.sort_values('metrics')
    df = df.reset_index(drop=True)

    for index in range(2, df.shape[1]):
        columnSeriesObj = df.iloc[:, index].apply(_round)
        df.iloc[:, index] = columnSeriesObj.values
    
    display(HTML(df.to_html()))