from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

def expand_dataset(X, y, nobs, n_components=10, verbose = True):
    '''
    Generate equal number of samples for each class.

    nobs : samples to generate per class.
    n_components : number of Gaussian components (paper Table 2: k=10).

    Notes
    -----
    * The GMM is now fitted on the samples of EACH class separately (the previous
      version fitted on the whole ``X`` every iteration, ignoring the label).
    * ``covariance_type='diag'`` matches the paper's naive/diagonal-covariance
      assumption and stays numerically stable for high-dimensional spectra,
      where a full covariance (n_features >> n_samples) would be singular.
    '''

    X = np.asarray(X, dtype=float)
    yv = np.asarray(y)

    final_data = pd.DataFrame()
    for label in np.unique(yv):
        X_label = X[yv == label]
        k = max(1, min(n_components, len(X_label)))     # cannot exceed #samples
        gmm = GaussianMixture(n_components=k, covariance_type='diag')
        gmm.fit(X_label)
        if verbose:
            print('GMM weights for label', label, ":",
                  str(np.round(gmm.weights_, 2)).replace('\r', '').replace('\n', ''))
        X_new, _ = gmm.sample(nobs)
        X_new = np.clip(X_new, 0, None)                 # spectra are non-negative
        batch = pd.DataFrame(X_new)
        batch['label'] = [label] * len(batch)
        final_data = pd.concat([final_data, batch], axis=0)

    final_data.reset_index(drop=True, inplace=True)
    return final_data.iloc[:, :-1], final_data.iloc[:, -1]