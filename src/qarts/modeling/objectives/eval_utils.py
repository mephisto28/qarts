import math
import typing as T
import numpy as np
import pandas as pd
from numba import njit, prange

try:
    from scipy.stats import rankdata
except Exception:
    rankdata = None


def nan_corr_xs(x: np.ndarray, y: np.ndarray, axis: int = 0, min_count: int = 20) -> np.ndarray:
    """
    NaN-safe Pearson correlation along `axis`, computed vectorized.

    x, y: same shape
    returns: correlation with x/y reduced along axis

    If valid count < min_count or zero variance -> NaN.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    cnt = mask.sum(axis=axis)

    # avoid divide-by-zero by temporarily setting cnt=1 (will be masked later)
    cnt_safe = np.where(cnt > 0, cnt, 1)

    x0 = np.where(mask, x, 0.0)
    y0 = np.where(mask, y, 0.0)

    mx = x0.sum(axis=axis) / cnt_safe
    my = y0.sum(axis=axis) / cnt_safe

    # broadcast means back
    mx_b = np.expand_dims(mx, axis=axis)
    my_b = np.expand_dims(my, axis=axis)

    xc = np.where(mask, x - mx_b, 0.0)
    yc = np.where(mask, y - my_b, 0.0)

    # sample cov/var (ddof=1)
    denom = np.where(cnt_safe > 1, (cnt_safe - 1), 1)
    cov = (xc * yc).sum(axis=axis) / denom
    vx = (xc * xc).sum(axis=axis) / denom
    vy = (yc * yc).sum(axis=axis) / denom

    corr = cov / np.sqrt(vx * vy)
    corr = np.where((cnt >= min_count) & (vx > 0) & (vy > 0), corr, np.nan)
    return corr


def _nan_quantile_along_axis(a: np.ndarray, q: float, axis: int = 0) -> np.ndarray:
    """
    Compute nan-quantile along axis for each cross-section, vectorized.
    Uses np.nanquantile. Non-finite values are treated as NaN.
    """
    a2 = np.where(np.isfinite(a), a, np.nan)
    return np.nanquantile(a2, q, axis=axis)


def nan_corr_xs_long(
    pred: np.ndarray,
    y: np.ndarray,
    quantiles=(0.1, 0.2, 0.5),
    axis: int = 0,
    min_count: int = 20,
) -> np.ndarray:
    """
    Long-only cross-sectional IC.

    For each cross-section:
      1) compute quantile threshold(s) on pred (ignoring non-finite)
      2) select stocks with pred >= threshold
      3) compute Pearson corr(pred, y) on the selected subset (nan-safe)

    Returns:
      ic: array with an extra leading dimension for quantiles if len(quantiles)>1.
          - if quantiles is scalar-like (float), returns same shape as nan_corr_xs output.
          - if quantiles is iterable, returns shape (len(quantiles), ...) where ... is the reduced shape.
    """
    qs = np.atleast_1d(np.array(quantiles, dtype=float))
    thr = np.stack([_nan_quantile_along_axis(pred, q, axis=axis) for q in qs], axis=0)  # (Q, ...)

    # broadcast thr back to pred shape: insert axis dimension at the same place as reduction axis
    # pred shape: (..., N, ...) ; thr shape: (Q, ..., ...) without N
    # we need thr_b shape: (Q, ..., 1, ...) to compare with pred[None,...]
    thr_b = np.expand_dims(thr, axis=axis + 1)

    pred_b = pred[None, ...]
    y_b = y[None, ...]

    finite = np.isfinite(pred_b) & np.isfinite(y_b)
    long_mask = pred_b >= thr_b
    mask = finite & long_mask

    cnt = mask.sum(axis=axis + 1)
    cnt_safe = np.where(cnt > 0, cnt, 1)

    px0 = np.where(mask, pred_b, 0.0)
    py0 = np.where(mask, y_b, 0.0)

    mx = px0.sum(axis=axis + 1) / cnt_safe
    my = py0.sum(axis=axis + 1) / cnt_safe

    mx_b = np.expand_dims(mx, axis=axis + 1)
    my_b = np.expand_dims(my, axis=axis + 1)

    xc = np.where(mask, pred_b - mx_b, 0.0)
    yc = np.where(mask, y_b - my_b, 0.0)

    denom = np.where(cnt_safe > 1, (cnt_safe - 1), 1)
    cov = (xc * yc).sum(axis=axis + 1) / denom
    vx = (xc * xc).sum(axis=axis + 1) / denom
    vy = (yc * yc).sum(axis=axis + 1) / denom

    ic = cov / np.sqrt(vx * vy)
    ic = np.where((cnt >= min_count) & (vx > 0) & (vy > 0), ic, np.nan)

    # if user passed a scalar quantile, return squeezed
    if np.ndim(quantiles) == 0 or (isinstance(quantiles, (float, int))):
        return ic[0]
    return ic


def nan_hit_rate_xs(pred: np.ndarray, gt: np.ndarray, axis: int = 0, frac: float = 0.5, min_count: int = 20) -> np.ndarray:
    """Fraction of sign matches along axis, NaN-safe."""
    threshold = np.quantile(pred, 1 - frac)
    mask = np.isfinite(pred) & np.isfinite(gt)
    cnt = mask.sum(axis=axis)
    ok = ((pred > threshold) == (gt > 0)) & mask
    hr = ok.sum(axis=axis) / np.where(cnt > 0, cnt, 1)
    hr = np.where(cnt >= min_count, hr, np.nan)
    return hr


def auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Fast AUC for binary labels using rank statistic.
    Returns NaN if not enough positives/negatives or invalid data.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size < 20:
        return np.nan
    y = (y_true > 0).astype(np.int8)  # positive = gt>0
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan

    # ranks of scores; handle ties by average rank
    if rankdata is None:
        # fallback: argsort-based ranks (ties not perfectly handled)
        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, y_score.size + 1, dtype=float)
    else:
        ranks = rankdata(y_score, method="average")

    sum_ranks_pos = float(ranks[y == 1].sum())
    # Mann-Whitney U for positives
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auc = u / (n_pos * n_neg)
    return float(auc)


def newey_west_tstat(x: np.ndarray, lag: int = 0) -> float:
    """
    Newey-West t-stat for the mean of series x.
    x: 1D array with NaNs allowed.
    lag: HAC lag (>=0). If 0 -> reduces to classic i.i.d. standard error.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return np.nan
    mu = x.mean()
    xc = x - mu

    # autocovariances
    gamma0 = float(np.dot(xc, xc) / n)
    var = gamma0
    L = int(max(lag, 0))
    for l in range(1, min(L, n - 1) + 1):
        w = 1.0 - l / (L + 1.0)
        gamma_l = float(np.dot(xc[l:], xc[:-l]) / n)
        var += 2.0 * w * gamma_l

    se_mean = math.sqrt(max(var, 0.0) / n)
    if se_mean == 0:
        return np.nan
    return float(mu / se_mean)


def quantile_stats_one_column(sig: np.ndarray, ret: np.ndarray, q: int = 10, top_counts: T.Tuple[int] = (20, 50, 100)) -> T.Tuple[np.ndarray, float, float, float]:
    """
    For a single cross-section column:
      - quantile mean returns (length q)
      - top-bottom spread
      - monotonicity score: Spearman corr between quantile index and quantile returns
      - extreme_tail: (top 1% mean - bottom 1% mean)
    """
    mask = np.isfinite(sig) & np.isfinite(ret)
    sig = sig[mask]
    ret = ret[mask]
    n = sig.size
    if n < max(50, q * 5):
        return (np.full(q, np.nan), np.nan, np.nan, np.nan)

    order = np.argsort(sig)
    sig_s = sig[order]
    ret_s = ret[order]

    # quantile edges by equal counts
    idx = np.linspace(0, n, q + 1).astype(int)
    q_means = np.empty(q + len(top_counts), dtype=float)
    for i in range(q):
        sl = slice(idx[i], idx[i + 1])
        r = ret_s[sl]
        q_means[i] = np.nanmean(r) if r.size else np.nan
    for i, count in enumerate(top_counts):
        r = ret_s[-count:]
        q_means[q + i] = np.nanmean(r) if r.size else np.nan

    top = q_means[q-1]
    bot = q_means[0]
    spread = top - bot

    # monotonicity: corr(quantile_index, q_means)
    qi = np.arange(1, q + 1, dtype=float)
    if np.isfinite(q_means).sum() >= max(3, q // 2):
        # Spearman via ranks of q_means
        if rankdata is not None:
            r1 = rankdata(qi, method="average")
            r2 = rankdata(q_means[:q], method="average")
            mono = np.corrcoef(r1, r2)[0, 1]
        else:
            mono = np.corrcoef(qi, q_means[:q])[0, 1]
    else:
        mono = np.nan

    # extreme tails: top 1% - bottom 1%
    k = max(1, int(round(0.01 * n)))
    tail_top = float(np.nanmean(ret_s[-k:]))
    tail_bot = float(np.nanmean(ret_s[:k]))
    tail_spread = tail_top - tail_bot

    return q_means, float(spread), float(mono), float(tail_spread)


def value_bucket_stats_one_column(sig: np.ndarray, ret: np.ndarray, bins: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Bucket returns by numeric bins on *standardized* signal (z-score).
    bins: array of bin edges, e.g. [-inf,-2,-1,0,1,2,inf]
    returns:
      - bucket centers (len K)
      - mean return per bucket (len K)
    """
    mask = np.isfinite(sig) & np.isfinite(ret)
    sig = sig[mask]
    ret = ret[mask]
    if sig.size < 50:
        k = len(bins) - 1
        return (np.full(k, np.nan), np.full(k, np.nan))

    # z-score in this cross-section
    m = float(sig.mean())
    s = float(sig.std(ddof=0))
    if s == 0:
        k = len(bins) - 1
        return (np.full(k, np.nan), np.full(k, np.nan))
    z = (sig - m) / s

    k = len(bins) - 1
    bucket_means = np.full(k, np.nan)
    bucket_centers = np.full(k, np.nan)

    inds = np.digitize(z, bins) - 1  # [0..k-1]
    for i in range(k):
        r = ret[inds == i]
        if r.size:
            bucket_means[i] = float(np.nanmean(r))
            lo, hi = bins[i], bins[i + 1]
            # center for plotting
            if np.isfinite(lo) and np.isfinite(hi):
                bucket_centers[i] = (lo + hi) / 2.0
            elif np.isfinite(lo) and not np.isfinite(hi):
                bucket_centers[i] = lo + 0.5
            elif not np.isfinite(lo) and np.isfinite(hi):
                bucket_centers[i] = hi - 0.5
            else:
                bucket_centers[i] = 0.0
    return bucket_centers, bucket_means


# 定义分桶的配置 (常量)
# 0-98: 每 1% 一个桶 (共99个)
# 99: 99% - 99.5%
# 100: 99.5% - 99.9%
# 101: 99.9% - 99.95%
# 102: 99.95% - 99.99%
# 103: > 99.99%
NUM_BINS = 104

@njit(fastmath=True, cache=True)
def _get_bin_index(q: float) -> int:
    """
    将分位数映射到 0-103 的桶索引。
    硬编码判断逻辑以获得最高执行速度。
    """
    if q < 0.99:
        # 0.00 ~ 0.9899... -> 0 ~ 98
        return int(q * 100)
    
    # 处理尾部高精度分位
    if q < 0.995:
        return 99
    elif q < 0.999:
        return 100
    elif q < 0.9995:
        return 101
    elif q < 0.9999:
        return 102
    else:
        return 103

@njit(parallel=True)
def _accumulate_daily_stats(
    quantiles: np.ndarray,      # (N_samples, n_scales)
    gts: np.ndarray,            # (N_samples, n_scales)
    out_sums: np.ndarray,       # (n_scales, NUM_BINS) 累加器引用
    out_counts: np.ndarray      # (n_scales, NUM_BINS) 计数器引用
):
    """
    核心累加逻辑。
    单线程执行即可，因为直接操作内存比并行加锁更快。
    处理 120万行 * 10列 仅需几十毫秒。
    """
    n_samples, n_scales = quantiles.shape
    
    for i in prange(n_samples):
        for scale_idx in range(n_scales):
            q_val = quantiles[i, scale_idx]
            gt_val = gts[i, scale_idx]
            
            # 过滤无效数据 (NaN in Score or NaN in GT)
            if np.isnan(q_val) or np.isnan(gt_val):
                continue
            
            # 1. 计算桶索引
            bin_idx = _get_bin_index(q_val)
            
            # 2. 原地累加 (In-place update)
            out_sums[scale_idx, bin_idx] += gt_val
            out_counts[scale_idx, bin_idx] += 1.0


class GlobalQuantileAnalyzer:
    def __init__(self, num_variables: int = 10):
        self.num_variables = num_variables
        self.num_bins = 104
        
        # 状态矩阵：(Scale, Bin)
        # 使用 float64 防止溢出
        self.total_return_sums = np.zeros((num_variables, self.num_bins), dtype=np.float64)
        self.total_counts = np.zeros((num_variables, self.num_bins), dtype=np.float64)
        
        # 定义桶的标签，方便后续展示
        self.bin_labels = [f"{i}%-{i+1}%" for i in range(99)] + [
            "99%-99.5%", "99.5%-99.9%", "99.9%-99.95%", "99.95%-99.99%", ">99.99%"
        ]

    def update(self, daily_score_quantiles: np.ndarray, daily_gts: np.ndarray):
        if daily_score_quantiles.ndim == 3:
            N, T, D = daily_score_quantiles.shape
            # Flatten to (N*T, D)
            flat_quantiles = daily_score_quantiles.reshape(-1, D)
            flat_gts = daily_gts.reshape(-1, D)
        else:
            flat_quantiles = daily_score_quantiles
            flat_gts = daily_gts
            
        assert flat_quantiles.shape == flat_gts.shape, "Score and GT shapes must match"
        assert flat_quantiles.shape[1] == self.num_variables, "Scale dimension mismatch"
        
        # 3. 累加统计 (Numba 加速)
        _accumulate_daily_stats(
            flat_quantiles,
            flat_gts,
            self.total_return_sums,
            self.total_counts
        )

    def get_report(self, scale_idx: int = None) -> pd.DataFrame:
        """
        生成统计报告。
        :param scale_idx: 如果指定，只返回该尺度的 DataFrame；否则返回所有尺度的 Dict。
        """
        # 防止除以 0
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_returns = self.total_return_sums / self.total_counts
            
        # 封装结果
        results = {}
        for s in range(self.num_variables):
            df = pd.DataFrame({
                'Bin': self.bin_labels,
                'Avg_Return': mean_returns[s],
                'Count': self.total_counts[s],
                'Sum_Return': self.total_return_sums[s]
            })
            results[s] = df
            
        if scale_idx is not None:
            return results[scale_idx]
        return results

    def reset(self):
        self.total_return_sums.fill(0)
        self.total_counts.fill(0)