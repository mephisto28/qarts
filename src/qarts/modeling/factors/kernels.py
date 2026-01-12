import numba as nb
import numpy as np


@nb.njit(parallel=True)
def ffill2d(a, out, reverse: bool = False):
    """
    Out-of-place forward fill along axis=1 for 2D float array.
    out must be preallocated with same shape as a.
    """
    n0, n1 = a.shape
    for i in nb.prange(n0):
        last = np.nan
        if reverse:
            j_range = range(n1 - 1, -1, -1)
        else:
            j_range = range(n1)
        for j in j_range:
            x = a[i, j]
            if x == x:
                last = x
                out[i, j] = x
            else:
                if last == last:
                    out[i, j] = last
                else:
                    out[i, j] = x   # still NaN
    return out
    

@nb.njit(parallel=True)
def reverse_cumsum_2d(a: np.ndarray, out: np.ndarray):
    n0, n1 = a.shape
    for i in nb.prange(n0):
        acc = 0
        for j in range(n1 - 1, -1, -1):
            v = a[i, j]
            if v == v:
                acc += v
            out[i, j] = acc
    return out


@nb.njit(parallel=True)
def fast_binned_percentile_2d(arr, n_bins=1000, sigma_clip=3.5, out: np.ndarray = None):
    """
    将 2D 数组（沿最后一维）的值转换为近似的 Percentile (0~1)。
    使用分桶 + 线性插值的方法，比 argsort 快很多。
    
    Parameters:
    -----------
    arr : 2d array
        输入数组 (n_samples, n_features) 或者 (n_time, n_stocks). 
        计算是沿着 axis=1 进行的。
    n_bins : int
        分桶的数量。数量越高精度越高，但速度微降。1000通常足够。
    sigma_clip : float
        用于 Winsorize 的标准差倍数。
        有效分桶范围将被设定为 [mean - sigma*std, mean + sigma*std]。
        这能防止极端离群值拉伸分桶范围，导致中间密集数据分辨率不足。
        
    Returns:
    --------
    out : 2d array
        范围在 [0, 1] 之间的 percentile rank。
    """
    rows, cols = arr.shape
    if out is None:
        out = np.empty_like(arr)
    
    # 并行处理每一行
    for i in nb.prange(rows):
        row_data = arr[i, :]

        mean_val = 0.0
        _n = 0
        for j in range(cols):
            val = row_data[j]
            if val == val:
                _n += 1
                mean_val += val
        mean_val /= _n
        
        var_val = 0.0
        _n = 0
        for j in range(cols):
            val = row_data[j]
            if val == val:
                _n += 1
                d = val - mean_val
                var_val += d * d
        std_val = np.sqrt(var_val / (_n - 1))
        
        # 确定 Winsorize 的边界, 只有在这个范围内的数据会被精细分桶, 极值将被压缩到 0 或 1
        low_limit = mean_val - sigma_clip * std_val
        high_limit = mean_val + sigma_clip * std_val
        
        # 防止标准差为0的情况
        if high_limit == low_limit:
            # 数据全一样
            for j in range(cols):
                out[i, j] = 0.5
            continue

        bin_width = (high_limit - low_limit) / n_bins
        inv_bin_width = 1.0 / bin_width
        
        # 2. 构建直方图 (Histogram)
        # 注意：Numba 中在循环内分配小数组通常很快，或者可以预分配
        hist = np.zeros(n_bins, dtype=np.int32)
        
        # 第一次遍历：填充直方图
        for j in range(cols):
            val = row_data[j]
            if val <= low_limit:
                # 小于下界的暂时不算入桶，或者算入第0桶，这里我们为了逻辑清晰
                # 放在最后算 rank 时处理
                pass
            elif val >= high_limit:
                pass
            else:
                bin_idx = int((val - low_limit) * inv_bin_width)
                if bin_idx >= n_bins: 
                    bin_idx = n_bins - 1
                hist[bin_idx] += 1
        
        # 3. 计算累计分布 (CDF) - 为了插值
        # cum_hist[k] 表示在第 k 个桶 *之前* 有多少个元素
        cum_hist = np.zeros(n_bins, dtype=np.int32)
        current_sum = 0
        
        # 统计小于 low_limit 的数量
        count_below_range = 0
        for j in range(cols):
            if row_data[j] < low_limit:
                count_below_range += 1
        
        current_sum = count_below_range
        for k in range(n_bins):
            cum_hist[k] = current_sum
            current_sum += hist[k]
        
        total_count = cols
        
        # 4. 第二次遍历：计算 Percentile 并插值
        for j in range(cols):
            val = row_data[j]
            
            if val <= low_limit:
                # 极小值直接给 0 或极小的 rank
                # 为了保持连续性，可以设为 count_below_range / total / 2 或者 0
                out[i, j] = 0.0 
            elif val >= high_limit:
                out[i, j] = 1.0
            else:
                # 找到桶
                bin_idx = int((val - low_limit) * inv_bin_width)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                
                # 核心插值逻辑：
                # Rank = (在该桶之前的数量 + 该桶内的相对位置 * 该桶内的数量) / 总数
                
                count_prev = cum_hist[bin_idx]
                count_in_bin = hist[bin_idx]
                
                # 桶内相对位置 (0.0 ~ 1.0)
                bin_start_val = low_limit + bin_idx * bin_width
                fraction = (val - bin_start_val) * inv_bin_width
                
                # 简单的线性插值
                rank_score = count_prev + count_in_bin * fraction
                out[i, j] = rank_score / total_count
                
    return out


@nb.njit(parallel=True)
def binned_percentile_axis1_strided(x, B=1024, k_sigma=3.0, eps=1e-12, out: np.ndarray = None):
    """
    x: (A, M, T) C-order
    返回: (A, M, T) float32, 每个 (a,t) 对 m 维做近似 percentile
    """
    A, M, T = x.shape
    if out is None:
        out = np.empty((A, M, T), dtype=np.float32)

    for a in nb.prange(A):
        # 这两个数组在每个 a 内复用，避免反复分配
        hist = np.empty(B, dtype=np.int32)
        cdf  = np.empty(B, dtype=np.float32)

        for t in range(T):
            # 1) mean/std over m
            s = 0.0
            ss = 0.0
            count = 0
            for m in range(M):
                v = x[a, m, t]
                if v == v:
                    s += v
                    count += 1
                    ss += v * v
            mu = s / count
            var = ss / count - mu * mu
            if var < 0.0:
                var = 0.0
            sig = np.sqrt(var)

            if sig < eps:
                for m in range(M):
                    out[a, m, t] = 0.5
                continue

            L = mu - k_sigma * sig
            U = mu + k_sigma * sig
            rng = U - L
            if rng < eps:
                for m in range(M):
                    out[a, m, t] = 0.5
                continue

            # 2) reset hist
            for b in range(B):
                hist[b] = 0

            # 3) fill hist (winsorize + bin)
            inv_rng = 1.0 / rng
            for m in range(M):
                v = x[a, m, t]
                if v < L:
                    v = L
                elif v > U:
                    v = U
                tt = (v - L) * inv_rng
                b = int(tt * B)
                if b >= B:
                    b = B - 1
                hist[b] += 1

            # 4) prefix sum -> cdf
            running = 0
            inv_M = 1.0 / M
            for b in range(B):
                running += hist[b]
                cdf[b] = running * inv_M

            # 5) lookup (含 bin 内插值，减小台阶感)
            for m in range(M):
                v = x[a, m, t]
                if v <= L:
                    out[a, m, t] = 0.0
                    continue
                if v >= U:
                    out[a, m, t] = 1.0
                    continue

                tt = (v - L) * inv_rng
                b = int(tt * B)
                if b >= B:
                    b = B - 1

                p1 = cdf[b]
                p0 = 0.0 if b == 0 else cdf[b - 1]

                left = b / B
                frac = (tt - left) * B  # [0,1]
                out[a, m, t] = p0 + (p1 - p0) * frac

    return out