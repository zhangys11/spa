from pyNNRW.nnrw import homo_stacking as _homo_stacking, hetero_stacking, FSSE, fsse_homo_stacking
from pyNNRW.mlp import create_mlp_instance
from pyNNRW.dtc import create_dtc_instance, create_stump_instance
from pyNNRW.elm import create_elm_instance, create_elmcv_instance, ELMClassifier
from pyNNRW.knn import create_knn_instance
from pyNNRW.lr import create_lr_instance
from pyNNRW.rvfl import create_rvfl_instance, create_rvflcv_instance, RVFLClassifier
import time
import numpy as np
from sklearn.impute import SimpleImputer


def homo_stacking(X, y, *args, **kwargs):
    """Wrapper around pyNNRW homo_stacking that handles NaN values."""
    if np.any(np.isnan(X)):
        print(f"Warning: X contains NaN. Imputing with median. "
              f"Consider preprocessing first (e.g., baseline removal, thresholding).")
        X = SimpleImputer(strategy='median').fit_transform(X)
    kwargs.pop('repeat', None)  # legacy notebook parameter, no longer used
    return _homo_stacking(X, y, *args, **kwargs)


def ELMClf(X, y, L=20, test_size=0.2, verbose=True):
    """Train/test an ELM classifier and return accuracies."""
    if y is None:
        raise ValueError("y is None. Please use a dataset with labeled classes (e.g., 'vintage_526').")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
    t0 = time.time()
    clf = ELMClassifier(n_hidden_nodes=L)
    clf.fit(X_train, y_train)
    t = time.time() - t0
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    if verbose:
        print(f'ELM (L={L}): train acc={train_acc:.3f}, val acc={val_acc:.3f}, time={t:.2f}s')
    return train_acc, val_acc, t


def RVFLClf(X, y, L=20, test_size=0.2, verbose=True):
    """Train/test an RVFL classifier and return accuracies."""
    if y is None:
        raise ValueError("y is None. Please use a dataset with labeled classes (e.g., 'vintage_526').")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
    t0 = time.time()
    clf = RVFLClassifier(n_hidden_nodes=L)
    clf.fit(X_train, y_train)
    t = time.time() - t0
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    if verbose:
        print(f'RVFL (L={L}): train acc={train_acc:.3f}, val acc={val_acc:.3f}, time={t:.2f}s')
    return train_acc, val_acc, t