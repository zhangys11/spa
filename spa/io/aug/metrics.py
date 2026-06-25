"""
Fidelity / quality metrics for evaluating data-augmentation (generative) models
on 1-D spectroscopic profiling data.

Two complementary families of metrics are provided:

1. Pairwise sample-level similarity metrics (kept for backward compatibility).
   They average a pairwise similarity over every (real, synthetic) sample pair.
   NOTE: such pairwise averages mix within-set and between-set variation and
   must NOT be read as a true distributional distance. They are kept only as a
   coarse, easy-to-interpret fidelity proxy.

2. Distribution-level metrics that compare the two *sets* directly and are the
   recommended way to judge a generative model:
       - MMD (RBF kernel, unbiased, median-heuristic bandwidth)  -> 0 is best
       - Sliced-Wasserstein distance                             -> 0 is best
       - Energy distance                                         -> 0 is best
       - C2ST (classifier two-sample test accuracy)              -> 0.5 is best
       - Improved precision / recall (fidelity / diversity)      -> 1 is best
   plus spectroscopy-specific diagnostics:
       - NN memorization ratio (are samples just copies?)        -> ~1 healthy
       - Peak-position F1 (do generated peaks land on real ones?) -> 1 is best
       - Sparsity error (is the zero/baseline structure kept?)   -> 0 is best
"""

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from skimage.metrics import structural_similarity
from ...cla import run_multiclass_clfs


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_2d(X):
    '''Return a contiguous 2-D float numpy array.'''
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def _to_prob(v, eps=1e-12):
    '''
    Turn a 1-D intensity vector into a valid probability distribution.

    Spectral intensities are *not* probabilities. KL / JS divergence are only
    meaningful on normalized, strictly-positive distributions, so we take the
    magnitude, add a small floor (to avoid log(0) / division-by-0) and
    renormalize to sum to 1. This replaces the previous, data-destroying hack
    that overwrote every value < 1e-4 with the constant 1.
    '''
    v = np.abs(np.asarray(v, dtype=float))
    v = v + eps
    s = v.sum()
    return v / s if s > 0 else np.full_like(v, 1.0 / len(v))


def _kl(p, q):
    return float(np.sum(p * np.log(p / q)))


# --------------------------------------------------------------------------- #
# 1. pairwise sample-level similarity (backward compatible)
# --------------------------------------------------------------------------- #
def calculate_pairwise_metrics(px, py):
    '''Compute the 9 pairwise similarity metrics between two sample vectors.'''
    arr_x = np.asarray(px, dtype=float)
    arr_y = np.asarray(py, dtype=float)

    # KL / JS are computed on properly normalized probability vectors.
    p = _to_prob(arr_x)
    q = _to_prob(arr_y)
    KL = _kl(p, q)
    JS = float(scipy.spatial.distance.jensenshannon(p, q))

    px = pd.Series(arr_x)
    py = pd.Series(arr_y)
    pearson = px.corr(py, method="pearson")
    spearman = px.corr(py, method="spearman")
    kendall = px.corr(py, method="kendall")
    similarity = 1 - scipy.spatial.distance.cosine(arr_x, arr_y)
    r2 = r2_score(arr_x, arr_y)
    mse = mean_squared_error(arr_x, arr_y)

    # SSIM was designed for images. We tile the 1-D signal into a small 2-D
    # patch. Unlike the old code we keep float precision (no .astype(int),
    # which silently zeroed sub-unit baselines) and use a correct, symmetric
    # data_range together with a valid odd window size.
    SSIM = _ssim_1d(arr_x, arr_y)

    return KL, JS, pearson, spearman, kendall, similarity, r2, mse, SSIM


def _ssim_1d(x, y):
    length = x.shape[0]
    win = min(7, length if length % 2 == 1 else length - 1)
    if win < 3:
        return np.nan
    data_range = float(max(x.max(), y.max()) - min(x.min(), y.min()))
    if data_range <= 0:
        data_range = 1.0
    x2 = np.tile(x, (win, 1))
    y2 = np.tile(y, (win, 1))
    return float(structural_similarity(x2, y2, data_range=data_range, win_size=win))


def calculate_similarity_metrics(dataset_raw, dataset_syn):
    '''Average the 9 pairwise metrics over all (real, synth) sample pairs.'''
    KL, JS, pearson, spearman = [], [], [], []
    kendall, similarity, r2, mse, SSIM = [], [], [], [], []

    for i in range(dataset_raw.shape[0]):
        for j in range(dataset_syn.shape[0]):
            px = dataset_raw.iloc[i, :]
            py = dataset_syn.iloc[j, :]
            result = calculate_pairwise_metrics(px, py)
            KL.append(result[0])
            JS.append(result[1])
            pearson.append(result[2])
            spearman.append(result[3])
            kendall.append(result[4])
            similarity.append(result[5])
            r2.append(result[6])
            mse.append(result[7])
            SSIM.append(result[8])

    # nanmean: a single degenerate (constant) pair no longer poisons the mean.
    return [np.nanmean(KL), np.nanmean(JS), np.nanmean(pearson),
            np.nanmean(spearman), np.nanmean(kendall), np.nanmean(similarity),
            np.nanmean(r2), np.nanmean(mse), np.nanmean(SSIM)]


# --------------------------------------------------------------------------- #
# 2. distribution-level metrics (recommended)
# --------------------------------------------------------------------------- #
DISTRIBUTION_METRIC_NAMES = [
    'MMD', 'SlicedW', 'EnergyDist', 'C2ST_acc',
    'precision', 'recall', 'NN_mem_ratio', 'PeakPos_F1', 'SparsityErr',
]


def _median_gamma(Z):
    '''Median-heuristic bandwidth for the RBF kernel.'''
    if len(Z) > 800:  # subsample to bound the O(n^2) pdist cost
        idx = np.random.default_rng(0).choice(len(Z), 800, replace=False)
        Z = Z[idx]
    d = pdist(Z, 'sqeuclidean')
    med = np.median(d)
    return 1.0 / med if med > 0 else 1.0


def mmd_rbf(X, Y, gamma=None):
    '''Unbiased squared MMD with an RBF kernel (0 = identical distributions).'''
    if gamma is None:
        gamma = _median_gamma(np.vstack([X, Y]))
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m, n = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = (Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1))
            - 2.0 * Kxy.mean())
    return float(max(mmd2, 0.0))


def sliced_wasserstein(X, Y, n_proj=64, seed=0):
    '''Average 1-D Wasserstein distance over random linear projections.'''
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    total = 0.0
    for _ in range(n_proj):
        v = rng.normal(size=d)
        v /= (np.linalg.norm(v) + 1e-12)
        total += scipy.stats.wasserstein_distance(X @ v, Y @ v)
    return float(total / n_proj)


def energy_distance(X, Y):
    '''Multivariate energy distance (0 = identical distributions).'''
    a = cdist(X, Y).mean()
    b = cdist(X, X).mean()
    c = cdist(Y, Y).mean()
    return float(max(2.0 * a - b - c, 0.0))


def c2st(X, Y, seed=0):
    '''
    Classifier two-sample test: cross-validated accuracy of telling real from
    synthetic apart. 0.5 means the two sets are indistinguishable (ideal).
    '''
    cv = min(5, len(X), len(Y))
    if cv < 2:
        return np.nan
    Z = np.vstack([X, Y])
    lab = np.r_[np.zeros(len(X)), np.ones(len(Y))]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    scores = cross_val_score(clf, Z, lab, cv=cv, scoring='accuracy')
    return float(scores.mean())


def precision_recall(real, fake, k=3):
    '''
    Improved precision / recall (Kynkaanniemi et al., 2019).
    precision = fraction of synthetic samples inside the real manifold (fidelity)
    recall    = fraction of real samples inside the synthetic manifold (diversity,
                a low value flags mode collapse).
    '''
    if len(real) <= k or len(fake) <= k:
        return np.nan, np.nan

    def _radii(A):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(A)
        dist, _ = nn.kneighbors(A)
        return dist[:, -1]

    r_real = _radii(real)
    r_fake = _radii(fake)
    precision = np.mean((cdist(fake, real) <= r_real[None, :]).any(axis=1))
    recall = np.mean((cdist(real, fake) <= r_fake[None, :]).any(axis=1))
    return float(precision), float(recall)


def nn_memorization_ratio(real, fake):
    '''
    Mean nearest-neighbour distance (synth -> real) divided by the mean
    nearest-neighbour distance within the real set. A value well below 1 means
    the generator copies / collapses onto training samples; ~1 is healthy.
    '''
    if len(real) < 2 or len(fake) < 1:
        return np.nan
    nn = NearestNeighbors(n_neighbors=1).fit(real)
    d_fr, _ = nn.kneighbors(fake)
    nn2 = NearestNeighbors(n_neighbors=2).fit(real)
    d_rr, _ = nn2.kneighbors(real)
    denom = d_rr[:, 1].mean()
    return float(d_fr[:, 0].mean() / denom) if denom > 0 else np.nan


def peak_position_f1(real, fake, tol=2):
    '''
    F1 between the peak positions of the mean real and mean synthetic spectrum.
    Captures whether a generator reproduces the chemically meaningful peaks
    rather than only the global shape.
    '''
    mr = np.asarray(real, dtype=float).mean(axis=0)
    mf = np.asarray(fake, dtype=float).mean(axis=0)
    pr = set(signal.find_peaks(mr, prominence=0.1 * (mr.max() - mr.min() + 1e-12))[0])
    pf = set(signal.find_peaks(mf, prominence=0.1 * (mf.max() - mf.min() + 1e-12))[0])
    if not pr or not pf:
        return np.nan
    tp = sum(any(abs(p - q) <= tol for q in pf) for p in pr)
    prec = tp / len(pf)
    rec = tp / len(pr)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def sparsity_error(real, fake):
    '''
    Absolute difference in the fraction of near-baseline (near-zero) bins.
    Sparse modalities such as TOF-MS are easily violated by smooth generators
    (e.g. KDE / DCGAN) that fill the baseline with noise.
    '''
    r = np.asarray(real, dtype=float)
    f = np.asarray(fake, dtype=float)
    thr = 0.01 * (r.max() - r.min() + 1e-12)
    return float(abs(np.mean(np.abs(r) < thr) - np.mean(np.abs(f) < thr)))


def calculate_distribution_metrics(real, synth):
    '''
    Compute all distribution-level metrics between a real and a synthetic set.

    Parameters
    ----------
    real, synth : array-like of shape (n_samples, n_features)

    Returns
    -------
    list of floats in the order of ``DISTRIBUTION_METRIC_NAMES``.
    Any metric that fails (e.g. too few samples) is returned as ``np.nan``
    instead of aborting the whole evaluation.
    '''
    real = _as_2d(real)
    synth = _as_2d(synth)

    def _safe(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001 - one bad metric must not kill all
            print(f'[metrics] {fn.__name__} failed: {exc}')
            return np.nan

    precision, recall = _safe(precision_recall, real, synth)
    if isinstance(precision, float) and np.isnan(precision):
        precision, recall = np.nan, np.nan

    return [
        _safe(mmd_rbf, real, synth),
        _safe(sliced_wasserstein, real, synth),
        _safe(energy_distance, real, synth),
        _safe(c2st, real, synth),
        precision,
        recall,
        _safe(nn_memorization_ratio, real, synth),
        _safe(peak_position_f1, real, synth),
        _safe(sparsity_error, real, synth),
    ]


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def evaluate_fidelity(X_synth, y_synth, model_names, baseline,
                      distribution_metrics=True, verbose=True):
    '''
    Evaluate how each synthesised dataset resembles the original data.

    Parameters
    ----------
    X_synth : list of synth feature sets
    y_synth : list of corresponding y values
    model_names : synth / data-augmentation method names
    baseline : a pandas DataFrame of the original dataset (a 'label' column,
        if present, is ignored). The caller's DataFrame is NOT modified.
    distribution_metrics : also report the recommended distribution-level
        metrics (MMD, sliced-Wasserstein, C2ST, precision/recall, ...).
    '''
    data_org = baseline.copy()  # do not mutate the caller's DataFrame
    if 'label' in data_org.columns:
        data_org = data_org.drop('label', axis=1)
    data_org = data_org.reset_index(drop=True)

    if verbose:
        plt.figure(figsize=(12, 4))
        plt.plot(data_org.columns, data_org.mean(), c='r', label='original data')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(prop={'size': 14})
        plt.show()

    result = pd.DataFrame()
    for (i, j, k) in zip(X_synth, y_synth, model_names):
        x_sam = pd.DataFrame(i)
        y_sam = pd.DataFrame(j)
        y_sam.columns = ['label']
        data_synth = pd.concat([x_sam, y_sam], axis=1)
        data_synth = data_synth.iloc[:, :-1]
        data_synth.columns = data_org.columns
        data_synth = data_synth.reset_index(drop=True)

        if verbose:
            plt.figure(figsize=(12, 4))
            plt.plot(data_synth.columns, data_synth.mean(), c='r', label=k)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(prop={'size': 14})
            plt.show()

        # 1. pairwise similarity (computed on the raw, uncorrupted data)
        scores = calculate_similarity_metrics(data_org, data_synth)

        # 2. distribution-level metrics (recommended)
        if distribution_metrics:
            scores = scores + calculate_distribution_metrics(
                data_org.values, data_synth.values)

        result[k] = scores

    index = ['KL', 'JS', 'pearson', 'spearman', 'kendall',
             'similarity', 'r2', 'mse', 'SSIM']
    if distribution_metrics:
        index = index + DISTRIBUTION_METRIC_NAMES
    result.index = index
    return result


def evaluate_classification(X_synth, y_synth, model_names, verbose=False):

    dic = {}
    for (i, j, k) in zip(X_synth, y_synth, model_names):
        result, _ = run_multiclass_clfs(i, j, show=verbose)
        df = pd.DataFrame()
        for w in range(len(result)):
            df_tem = pd.DataFrame([result[w]])
            df_tem.columns = [col.split('(')[0] for col in df_tem.columns]
            df = pd.concat([df, df_tem], axis=0)
        df.index = ['train_accs', 'test_accs', 'train_precisions', 'test_precisions',
                    'train_recalls', 'test_recalls', 'train_f1s', 'test_f1s']
        dic[k] = df
    return dic
