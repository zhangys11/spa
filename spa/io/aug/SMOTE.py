'''
SMOTE family of over-sampling methods for imbalanced data.

All methods share the same idea: instead of duplicating minority samples
(which causes over-fitting), they *synthesise* new minority samples by
interpolating between real ones. They differ in (a) WHICH minority samples are
used as the seed for interpolation, (b) HOW the interpolation/neighbourhood is
defined, and (c) WHETHER a cleaning step is applied afterwards.

Public over-samplers (``fit_resample`` style, return the FULL resampled set):
    smote            - canonical k-NN SMOTE (single lambda, on the segment)
    smote_original   - the 2002 paper version (per-feature gap -> hyper-rectangle)
    borderline_smote - only oversample borderline (danger) minority samples
    adasyn           - generate MORE samples for harder-to-learn minority samples
    svm_smote        - use SVM support vectors to locate the border, then synthesise
    kmeans_smote     - cluster first, synthesise inside minority-safe clusters
    smote_nc         - mixed continuous + categorical features
    smote_n          - all-categorical features (VDM distance + majority vote)
    smote_enn        - SMOTE + Edited-Nearest-Neighbour cleaning
    smote_tomek      - SMOTE + Tomek-link cleaning

Legacy entry point (kept for backward compatibility, returns NEW data only):
    expand_dataset, create_one_random_sample

Notes
-----
* k-NN uses Euclidean distance, so continuous features should be scaled first.
* Vanilla over-samplers only interpolate (lambda in [0, 1]) -> they fill in the
  minority manifold but never extend beyond its convex hull.
* Because every synthetic sample is a convex blend of two *whole* real vectors,
  the SMOTE family preserves inter-feature correlation (e.g. co-occurring peaks
  in a spectrum) far better than per-dimension density models.
'''

import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist


# =========================================================================== #
# low-level helpers
# =========================================================================== #
def _mode(values):
    '''Most frequent value (used for categorical synthesis / k-NN voting).'''
    vals, counts = np.unique(values, return_counts=True)
    return vals[np.argmax(counts)]


def _sampling_targets(y, sampling_strategy='auto'):
    '''
    How many NEW samples each class needs.

    sampling_strategy :
        'auto'  -> bring every non-majority class up to the majority count.
        dict    -> {label: desired_total_count}.
    '''
    classes, counts = np.unique(y, return_counts=True)
    count_of = dict(zip(classes, counts))
    if isinstance(sampling_strategy, dict):
        return {c: max(int(sampling_strategy.get(c, count_of[c]) - count_of[c]), 0)
                for c in classes}
    n_max = counts.max()
    return {c: int(n_max - count_of[c]) for c in classes}


def _knn_within(X_grp, k):
    '''k nearest neighbours of every row within the same group (self excluded).'''
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_grp)
    return nn.kneighbors(X_grp, return_distance=False)[:, 1:]


def _jitter(X_grp, n_new, rng, scale=1e-6):
    '''Fallback when a class has a single sample: tiny-noise replication.'''
    pts = X_grp[rng.integers(0, len(X_grp), size=n_new)]
    return pts + rng.normal(0.0, scale, size=pts.shape)


def _populate(X_grp, neigh, seed_pool, n_new, rng,
              per_feature=False, weights=None, extrapolate=None, max_lambda=1.0):
    '''
    Core synthesiser shared by every SMOTE variant.

    x_new = x_i + lambda * (x_j - x_i)            (interpolation, on the segment)
    x_new = x_i + lambda * (x_i - x_j)            (extrapolation, pushed outward)

    Parameters
    ----------
    X_grp       : the minority samples to interpolate among.
    neigh       : (m, k) array of within-group neighbour indices.
    seed_pool   : indices (into X_grp) allowed to be used as the base point x_i.
    weights     : optional probability over ALL m points (ADASYN); overrides
                  seed_pool, sampling x_i proportionally to "learning difficulty".
    per_feature : draw an independent lambda per feature (original 2002 SMOTE).
    extrapolate : optional {seed_index: bool}; if True push the point outward.
    max_lambda  : upper bound of lambda (>1 allows extrapolation).
    '''
    d = X_grp.shape[1]
    k = neigh.shape[1]
    out = np.empty((n_new, d))
    seed_pool = np.asarray(seed_pool)
    for t in range(n_new):
        if weights is not None:
            i = int(rng.choice(len(weights), p=weights))
        else:
            i = int(seed_pool[rng.integers(len(seed_pool))])
        j = int(neigh[i][rng.integers(k)]) if k > 1 else int(neigh[i][0])
        lam = (rng.random(d) if per_feature else rng.random()) * max_lambda
        if extrapolate is not None and extrapolate.get(i, False):
            out[t] = X_grp[i] + lam * (X_grp[i] - X_grp[j])
        else:
            out[t] = X_grp[i] + lam * (X_grp[j] - X_grp[i])
    return out


def _oversample(X, y, gen_fn, sampling_strategy, random_state, **kw):
    '''Loop over classes, call a per-class generator, return the full set.'''
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    targets = _sampling_targets(y, sampling_strategy)
    X_parts, y_parts = [X], [y]
    for cls, n_new in targets.items():
        if n_new <= 0:
            continue
        new = gen_fn(X, y, cls, n_new, rng, **kw)
        if len(new):
            X_parts.append(new)
            y_parts.append(np.full(len(new), cls))
    return np.vstack(X_parts), np.concatenate(y_parts)


# =========================================================================== #
# per-class generators (one for each variant)
# =========================================================================== #
def _gen_smote(X, y, cls, n_new, rng, k_neighbors=5, per_feature=False):
    X_min = X[y == cls]
    m = len(X_min)
    if m == 0:
        return np.empty((0, X.shape[1]))
    if m == 1:
        return _jitter(X_min, n_new, rng)
    k = min(k_neighbors, m - 1)
    neigh = _knn_within(X_min, k)
    return _populate(X_min, neigh, np.arange(m), n_new, rng, per_feature=per_feature)


def _gen_borderline(X, y, cls, n_new, rng, k_neighbors=5, m_neighbors=10):
    '''Borderline-SMOTE: only the "danger" minority points are used as seeds.'''
    X_min = X[y == cls]
    m = len(X_min)
    if m <= 1:
        return _jitter(X_min, n_new, rng) if m == 1 else np.empty((0, X.shape[1]))
    # 用全数据集的 m 近邻判断每个少数类点的处境：
    #   多数类邻居 == m   -> noise（噪声，跳过）
    #   m/2 <= 多数类邻居 < m -> danger（位于决策边界，重点过采样）
    #   否则                 -> safe（内部，跳过）
    mm = min(m_neighbors, len(X) - 1)
    idx = NearestNeighbors(n_neighbors=mm + 1).fit(X).kneighbors(
        X_min, return_distance=False)[:, 1:]
    maj = (y[idx] != cls).sum(axis=1)
    danger = np.where((maj >= mm / 2.0) & (maj < mm))[0]
    if len(danger) == 0:
        danger = np.arange(m)                      # 没有边界点则退化为普通 SMOTE
    k = min(k_neighbors, m - 1)
    neigh = _knn_within(X_min, k)                  # 插值仍只在少数类内部找近邻
    return _populate(X_min, neigh, danger, n_new, rng)


def _gen_adasyn(X, y, cls, n_new, rng, k_neighbors=5):
    '''ADASYN: allocate more synthetic samples to harder minority points.'''
    X_min = X[y == cls]
    m = len(X_min)
    if m <= 1:
        return _jitter(X_min, n_new, rng) if m == 1 else np.empty((0, X.shape[1]))
    # r_i = (该少数类点周围多数类邻居数) / k，越大表示越"难学"（越靠近边界/被包围）
    mm = min(k_neighbors, len(X) - 1)
    idx = NearestNeighbors(n_neighbors=mm + 1).fit(X).kneighbors(
        X_min, return_distance=False)[:, 1:]
    r = (y[idx] != cls).sum(axis=1) / float(mm)
    if r.sum() == 0:
        r = np.ones(m)                             # 全部很"安全"时退化为均匀
    weights = r / r.sum()                          # 按难度归一化为采样概率
    k = min(k_neighbors, m - 1)
    neigh = _knn_within(X_min, k)
    return _populate(X_min, neigh, np.arange(m), n_new, rng, weights=weights)


def _gen_svm(X, y, cls, n_new, rng, k_neighbors=5, m_neighbors=10, svm_C=1.0):
    '''SVM-SMOTE: synthesise around the minority *support vectors* (the border).'''
    X_min = X[y == cls]
    m = len(X_min)
    if m <= 1:
        return _jitter(X_min, n_new, rng) if m == 1 else np.empty((0, X.shape[1]))
    try:
        svc = SVC(C=svm_C, gamma='scale').fit(X, y)
    except Exception:
        return _gen_smote(X, y, cls, n_new, rng, k_neighbors)
    # 取属于该少数类的支持向量（它们刻画了决策边界）
    orig_min = np.where(y == cls)[0]
    pos = {o: p for p, o in enumerate(orig_min)}
    sv_min = np.array([pos[s] for s in svc.support_ if y[s] == cls], dtype=int)
    if len(sv_min) == 0:
        return _gen_smote(X, y, cls, n_new, rng, k_neighbors)
    # 对每个少数类支持向量看其 m 邻域里多数类占比，决定插值还是外推
    mm = min(m_neighbors, len(X) - 1)
    idx = NearestNeighbors(n_neighbors=mm + 1).fit(X).kneighbors(
        X_min[sv_min], return_distance=False)[:, 1:]
    maj_frac = (y[idx] != cls).mean(axis=1)
    keep = sv_min[maj_frac < 1.0]                  # 全是多数类邻居的视为噪声丢弃
    if len(keep) == 0:
        keep = sv_min
    # 边界附近(多数类占比高) -> 外推以向边界扩张；内部 -> 普通插值
    extrapolate = {int(sv_min[i]): bool(maj_frac[i] > 0.5) for i in range(len(sv_min))}
    k = min(k_neighbors, m - 1)
    neigh = _knn_within(X_min, k)
    return _populate(X_min, neigh, keep, n_new, rng, extrapolate=extrapolate)


def _allocate(n_new, weights, rng):
    '''Split n_new into integer counts proportional to weights (sum preserved).'''
    raw = np.asarray(weights, dtype=float) * n_new
    counts = np.floor(raw).astype(int)
    rem = n_new - counts.sum()
    if rem > 0:                                    # 把取整剩余按小数部分大的优先分配
        order = np.argsort(-(raw - counts))
        for t in range(rem):
            counts[order[t % len(order)]] += 1
    return counts


def _gen_kmeans(X, y, cls, n_new, rng, k_neighbors=5,
                n_clusters=10, balance_threshold=0.5):
    '''KMeans-SMOTE: only synthesise inside clusters that are minority-safe.'''
    nc = min(n_clusters, len(X))
    try:
        labels = KMeans(n_clusters=nc, n_init=10, random_state=0).fit_predict(X)
    except Exception:
        return _gen_smote(X, y, cls, n_new, rng, k_neighbors)
    # 1) 聚类整个特征空间；2) 只保留"少数类占比 >= 阈值"的簇（安全区，避免在
    #    多数类/重叠区造噪声）；3) 越稀疏的安全簇分配越多合成样本（去填补空洞）。
    clusters = []
    for c in range(nc):
        mask = labels == c
        if mask.sum() == 0:
            continue
        ratio = np.mean(y[mask] == cls)
        X_min_c = X[mask & (y == cls)]
        if ratio >= balance_threshold and len(X_min_c) >= 2:
            density = np.mean(pdist(X_min_c)) if len(X_min_c) > 1 else 1.0
            sparsity = density / len(X_min_c)      # 稀疏度：越分散、越少 -> 越大
            clusters.append((X_min_c, sparsity))
    if not clusters:
        return _gen_smote(X, y, cls, n_new, rng, k_neighbors)
    w = np.array([s for _, s in clusters], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(clusters)) / len(clusters)
    counts = _allocate(n_new, w, rng)
    parts = []
    for (X_min_c, _), cnt in zip(clusters, counts):
        if cnt <= 0:
            continue
        k = min(k_neighbors, len(X_min_c) - 1)
        neigh = _knn_within(X_min_c, k)
        parts.append(_populate(X_min_c, neigh, np.arange(len(X_min_c)), cnt, rng))
    return np.vstack(parts) if parts else np.empty((0, X.shape[1]))


# =========================================================================== #
# public over-samplers
# =========================================================================== #
def smote(X, y, sampling_strategy='auto', k_neighbors=5, random_state=None):
    '''
    Canonical k-NN SMOTE (Chawla et al., 2002 / imbalanced-learn style).

    原理：对每个待过采样的类，在该类内部用 k 近邻找邻居；随机选一个样本 x_i 与它
    的某个近邻 x_j，沿连线插值 x_new = x_i + lambda*(x_j - x_i)，lambda~U(0,1)。
    单一 lambda 使新点严格落在两真实样本的连线段上，贴着少数类局部流形。
    '''
    return _oversample(X, y, _gen_smote, sampling_strategy, random_state,
                       k_neighbors=k_neighbors)


def smote_original(X, y, sampling_strategy='auto', k_neighbors=5, random_state=None):
    '''
    Original 2002-paper SMOTE (per-feature gap).

    原理：与 k-NN SMOTE 相同，但对**每个特征单独**抽一个 gap~U(0,1)，于是新点落在
    x_i 与 x_j 张成的"轴对齐超矩形"内，而非连线段上。是论文原始伪代码的写法。
    '''
    return _oversample(X, y, _gen_smote, sampling_strategy, random_state,
                       k_neighbors=k_neighbors, per_feature=True)


def borderline_smote(X, y, sampling_strategy='auto', k_neighbors=5,
                     m_neighbors=10, random_state=None):
    '''
    Borderline-SMOTE (Han et al., 2005).

    原理：只对位于决策边界的"danger"少数类样本过采样。判定方式是看每个少数类点在
    **全数据集**中的 m 个近邻里多数类占多少：一半到全部之间者为边界点（最易被误分，
    最值得增强）；全是多数类者视为噪声丢弃；几乎没有多数类者为安全内部点，跳过。
    '''
    return _oversample(X, y, _gen_borderline, sampling_strategy, random_state,
                       k_neighbors=k_neighbors, m_neighbors=m_neighbors)


def adasyn(X, y, sampling_strategy='auto', k_neighbors=5, random_state=None):
    '''
    ADASYN - Adaptive Synthetic Sampling (He et al., 2008).

    原理：自适应地"按难度分配"。对每个少数类点计算其 k 邻域中多数类的比例 r_i，r_i
    越大说明该点越被多数类包围、越难学；据此让难学的点生成更多合成样本，从而把
    生成密度自动倾斜到决策边界附近。（注：ADASYN 不保证完全平衡。）
    '''
    return _oversample(X, y, _gen_adasyn, sampling_strategy, random_state,
                       k_neighbors=k_neighbors)


def svm_smote(X, y, sampling_strategy='auto', k_neighbors=5, m_neighbors=10,
              svm_C=1.0, random_state=None):
    '''
    SVM-SMOTE (Nguyen et al., 2009).

    原理：先训练 SVM，用**少数类支持向量**来定位决策边界（支持向量恰好落在边界上）。
    以这些支持向量为种子合成：若其邻域多数类较少(安全)则在少数类内部插值；若邻域被
    多数类主导(边界)则向外**外推**，把少数类区域朝边界方向扩张。
    '''
    return _oversample(X, y, _gen_svm, sampling_strategy, random_state,
                       k_neighbors=k_neighbors, m_neighbors=m_neighbors, svm_C=svm_C)


def kmeans_smote(X, y, sampling_strategy='auto', k_neighbors=5, n_clusters=10,
                 balance_threshold=0.5, random_state=None):
    '''
    KMeans-SMOTE (Last et al., 2017).

    原理：先用 k-means 对整个特征空间聚类；只在"少数类占比高"的簇内做 SMOTE（避开
    多数类区/重叠区，避免造噪声）；并按簇的稀疏程度分配生成量——越稀疏的安全簇分到
    越多合成样本，用来填补少数类内部的空洞、平衡簇间密度。
    '''
    return _oversample(X, y, _gen_kmeans, sampling_strategy, random_state,
                       k_neighbors=k_neighbors, n_clusters=n_clusters,
                       balance_threshold=balance_threshold)


def smote_nc(X, y, categorical_features, sampling_strategy='auto',
             k_neighbors=5, random_state=None):
    '''
    SMOTE-NC - Nominal & Continuous (Chawla et al., 2002).

    原理：处理"连续+类别"混合特征。
    1) 距离：连续特征用欧氏距离；任意类别特征不同则给距离加一个惩罚项，大小为少数类
       连续特征标准差的中位数 med（用 one-hot 缩放 med/sqrt(2) 实现，使一次类别不同
       恰好贡献 med 的距离），让类别差异也影响近邻判定。
    2) 生成：连续特征照常插值；类别特征取"种子+其 k 近邻"在该列上的**众数**。
    categorical_features : 类别特征的列索引列表（这里假设已用整数编码）。
    '''
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    cat = list(categorical_features)
    cont = [c for c in range(X.shape[1]) if c not in set(cat)]
    targets = _sampling_targets(y, sampling_strategy)

    X_parts, y_parts = [X], [y]
    for cls, n_new in targets.items():
        if n_new <= 0:
            continue
        X_min = X[y == cls]
        m = len(X_min)
        if m == 0:
            continue
        if m == 1:
            X_parts.append(_jitter(X_min, n_new, rng))
            y_parts.append(np.full(n_new, cls))
            continue
        cont_vals = X_min[:, cont] if cont else np.empty((m, 0))
        med = np.median(cont_vals.std(axis=0)) if cont else 1.0
        if not np.isfinite(med) or med == 0:
            med = 1.0
        # 构造用于近邻搜索的编码矩阵
        enc = [cont_vals]
        for col in cat:
            cats = np.unique(X_min[:, col])
            onehot = (X_min[:, col][:, None] == cats[None, :]).astype(float)
            enc.append(onehot * (med / np.sqrt(2)))
        enc = np.hstack(enc) if enc else cont_vals
        k = min(k_neighbors, m - 1)
        neigh = _knn_within(enc, k)

        new = np.empty((n_new, X.shape[1]))
        for t in range(n_new):
            i = int(rng.integers(m))
            j = int(neigh[i][rng.integers(k)])
            lam = rng.random()
            row = np.empty(X.shape[1])
            for c in cont:
                row[c] = X_min[i, c] + lam * (X_min[j, c] - X_min[i, c])
            members = np.concatenate([[i], neigh[i]])
            for col in cat:
                row[col] = _mode(X_min[members, col])
            new[t] = row
        X_parts.append(new)
        y_parts.append(np.full(n_new, cls))
    return np.vstack(X_parts), np.concatenate(y_parts)


def smote_n(X, y, sampling_strategy='auto', k_neighbors=5, random_state=None):
    '''
    SMOTE-N - all Nominal features (Chawla et al., 2002).

    原理：特征全为类别时欧氏距离无意义，改用 VDM（Value Difference Metric）：两个取值
    的距离 = 各类别下条件概率 P(class|value) 之差的绝对值之和；样本间距离对各特征求和。
    用 VDM 找 k 近邻，合成样本的每个特征取"种子+近邻"在该列上的**众数**。
    '''
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    nf = X.shape[1]

    # 预计算每个特征每个取值的类别条件概率表 P(class | feature=value)
    prob = []
    for f in range(nf):
        col = X[:, f]
        table = {}
        for v in np.unique(col):
            mask = col == v
            table[v] = np.array([np.mean(y[mask] == c) for c in classes])
        prob.append(table)

    def _vdm(a, b):
        s = 0.0
        for f in range(nf):
            s += np.abs(prob[f][a[f]] - prob[f][b[f]]).sum()
        return s

    targets = _sampling_targets(y, sampling_strategy)
    X_parts, y_parts = [X], [y]
    for cls, n_new in targets.items():
        if n_new <= 0:
            continue
        X_min = X[y == cls]
        m = len(X_min)
        if m == 0:
            continue
        if m == 1:
            X_parts.append(np.repeat(X_min, n_new, axis=0))
            y_parts.append(np.full(n_new, cls))
            continue
        k = min(k_neighbors, m - 1)
        D = np.zeros((m, m))
        for a in range(m):
            for b in range(a + 1, m):
                D[a, b] = D[b, a] = _vdm(X_min[a], X_min[b])
        neigh = np.argsort(D, axis=1)[:, 1:k + 1]
        new = np.empty((n_new, nf), dtype=X.dtype)
        for t in range(n_new):
            i = int(rng.integers(m))
            members = np.concatenate([[i], neigh[i]])
            for f in range(nf):
                new[t, f] = _mode(X_min[members, f])
        X_parts.append(new)
        y_parts.append(np.full(n_new, cls))
    return np.vstack(X_parts), np.concatenate(y_parts)


# --------------------------------------------------------------------------- #
# combined over-sampling + cleaning
# --------------------------------------------------------------------------- #
def _enn_clean(X, y, n_neighbors=3):
    '''Edited Nearest Neighbours: drop samples that disagree with their k-NN vote.'''
    k = min(n_neighbors, len(X) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
        X, return_distance=False)[:, 1:]
    keep = [i for i in range(len(X)) if _mode(y[idx[i]]) == y[i]]
    return X[keep], y[keep]


def _tomek_clean(X, y):
    '''Remove Tomek links: mutual nearest neighbours of opposite classes.'''
    nn1 = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(
        X, return_distance=False)[:, 1]
    drop = set()
    for i in range(len(X)):
        j = nn1[i]
        if nn1[j] == i and y[i] != y[j]:           # 互为最近邻且类别相反 -> 边界重叠
            drop.add(i)
            drop.add(j)
    keep = [i for i in range(len(X)) if i not in drop]
    return X[keep], y[keep]


def smote_enn(X, y, sampling_strategy='auto', k_neighbors=5,
              enn_neighbors=3, random_state=None):
    '''
    SMOTE-ENN (Batista et al., 2004).

    原理：先用 SMOTE 过采样，再用 ENN（编辑最近邻）清洗——删除"类别与其多数近邻不一致"
    的样本（两类都删）。SMOTE 负责扩充少数类，ENN 负责清掉边界/重叠的噪声点，使两类
    的决策边界更干净。
    '''
    X_res, y_res = smote(X, y, sampling_strategy=sampling_strategy,
                         k_neighbors=k_neighbors, random_state=random_state)
    return _enn_clean(X_res, y_res, enn_neighbors)


def smote_tomek(X, y, sampling_strategy='auto', k_neighbors=5, random_state=None):
    '''
    SMOTE-Tomek (Batista et al., 2004).

    原理：先用 SMOTE 过采样，再删除 Tomek link——即互为最近邻却分属不同类的样本对
    （它们处在两类重叠/边界上）。相比 ENN，Tomek 清洗更轻，只移除最贴近边界的重叠对，
    使类间间隔更清晰。
    '''
    X_res, y_res = smote(X, y, sampling_strategy=sampling_strategy,
                         k_neighbors=k_neighbors, random_state=random_state)
    return _tomek_clean(X_res, y_res)


# =========================================================================== #
# legacy API (kept for backward compatibility with spa.io.aug.upsample)
# =========================================================================== #
def create_one_random_sample(X, l=[], d=0.5):
    '''
    [Deprecated] Create one sample by interpolating two random reference points.

    Kept only for backward compatibility. Prefer ``smote`` (true k-NN SMOTE).
    '''
    m = len(X)
    if m < 2:
        return None
    if len(l) == 0:
        l = np.random.choice(m, 2, replace=False)
    x1 = X[l[0], :]
    x2 = X[l[1], :]
    if np.allclose(x1, x2):
        return None
    k = random.random() * 2 * d - d
    return (x1 + x2) / 2 + np.multiply(k, (x2 - x1))


def expand_dataset(X, y, d=0.5, NX=3, append=False, k_neighbors=5, random_state=None):
    '''
    Legacy entry point used by ``spa.io.aug.upsample``.

    Now backed by true k-NN SMOTE (per class, k近邻插值). Returns NEW samples
    only by default, ``append=True`` prepends the original data.

    Parameters
    ----------
    NX : how many times to expand each class (n_new = NX * class_size).
    d  : kept for backward compatibility; the interpolation fraction is scaled
         to [0, 2*d] (d=0.5 -> standard lambda in [0, 1]).

    Bug fix: the previous version seeded the output with ``np.empty(X.shape)``,
    which leaked m rows of uninitialised memory into the synthetic set. Output
    is now assembled from scratch.
    '''
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    X_parts = [X] if append else []
    y_parts = [y] if append else []

    for cls in np.unique(y):
        X_min = X[y == cls]
        m = len(X_min)
        n_new = m * NX
        if n_new == 0:
            continue
        if m == 1:
            new = _jitter(X_min, n_new, rng)
        else:
            k = min(k_neighbors, m - 1)
            neigh = _knn_within(X_min, k)
            new = _populate(X_min, neigh, np.arange(m), n_new, rng,
                            max_lambda=2.0 * d)
        X_parts.append(new)
        y_parts.append(np.full(n_new, cls))

    if not X_parts:
        return np.empty((0, X.shape[1])), np.empty((0,))
    return np.vstack(X_parts), np.concatenate(y_parts)
