import numpy as np
from numba import prange, njit
from typing import List, Dict, Tuple

# ==========================================
# Part 1: Numba Accelerated Kernels (Core)
# ==========================================

@njit(fastmath=True, cache=True)
def _scale_fn(q: float, delta: float) -> float:
    """T-Digest k-size function mapping quantile to scale index."""
    # 使用 arcsin 保证尾部的高精度 (Standard T-Digest metric)
    return delta / (2 * np.pi) * np.arcsin(2 * q - 1)

@njit(fastmath=True, cache=True)
def _inv_scale_fn(k: float, delta: float) -> float:
    """Inverse mapping from scale index to quantile."""
    return 0.5 * (np.sin(2 * np.pi * k / delta) + 1)

@njit(fastmath=True, cache=True)
def _merge_centroids(
    means: np.ndarray,
    weights: np.ndarray,
    count: int,
    delta: float,
    max_capacity: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    核心合并逻辑：将排序后的所有点/质心压缩合并。
    Input: 已排序的 means 和对应的 weights
    """
    total_weight = np.sum(weights[:count])
    if total_weight == 0:
        return means, weights, 0
    
    # 输出容器
    out_means = np.zeros(max_capacity, dtype=np.float64)
    out_weights = np.zeros(max_capacity, dtype=np.float64)
    out_idx = 0
    
    # 初始化第一个质心
    w_so_far = 0.0
    k_lower = _scale_fn(0.0, delta) # always -delta/4
    
    current_mean = means[0]
    current_weight = weights[0]
    w_so_far = 0.0
    
    # 遍历合并
    for i in range(1, count):
        next_mean = means[i]
        next_weight = weights[i]
        
        # 计算当前合并后的潜在权重带来的 q limit
        proposed_weight = current_weight + next_weight
        q_next = (w_so_far + proposed_weight) / total_weight
        k_next = _scale_fn(q_next, delta)
        
        # 判定是否可以合并：如果增加后的 k 值变化小于 1，则属于同一质心
        if (k_next - k_lower) <= 1.0:
            # 加权平均合并
            current_mean = (current_mean * current_weight + next_mean * next_weight) / proposed_weight
            current_weight = proposed_weight
        else:
            # 结束当前质心，写入
            if out_idx >= max_capacity:
                # 极端防御，理论不应触发，除非 delta 设置过小
                break 
                
            out_means[out_idx] = current_mean
            out_weights[out_idx] = current_weight
            out_idx += 1
            
            # 累加权重，开启新质心
            w_so_far += current_weight
            k_lower = _scale_fn(w_so_far / total_weight, delta)
            
            current_mean = next_mean
            current_weight = next_weight

    # 写入最后一个质心
    if out_idx < max_capacity:
        out_means[out_idx] = current_mean
        out_weights[out_idx] = current_weight
        out_idx += 1
        
    return out_means, out_weights, out_idx

@njit(fastmath=True, cache=True)
def _process_buffer(
    c_means: np.ndarray, 
    c_weights: np.ndarray, 
    c_count: int,
    buffer: np.ndarray, 
    b_count: int,
    delta: float,
    max_capacity: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    将新数据缓冲区与现有质心合并。
    1. 拼接 (Existing Centroids + New Data)
    2. 排序
    3. 压缩 (Merge)
    """
    total_points = c_count + b_count
    
    # 临时数组用于排序
    temp_means = np.empty(total_points, dtype=np.float64)
    temp_weights = np.empty(total_points, dtype=np.float64)
    
    # 填充现有质心
    temp_means[:c_count] = c_means[:c_count]
    temp_weights[:c_count] = c_weights[:c_count]
    
    # 填充新数据 (weight=1.0)
    temp_means[c_count:] = buffer[:b_count]
    temp_weights[c_count:] = 1.0
    
    # 排序 (Based on means)
    sort_idx = np.argsort(temp_means)
    sorted_means = temp_means[sort_idx]
    sorted_weights = temp_weights[sort_idx]
    
    # 执行合并
    return _merge_centroids(sorted_means, sorted_weights, total_points, delta, max_capacity)

@njit(fastmath=True, cache=True)
def _get_quantile(
    c_means: np.ndarray, 
    c_weights: np.ndarray, 
    c_count: int, 
    q: float
) -> float:
    """
    从质心计算分位数 (Interpolation)
    """
    if c_count == 0:
        return np.nan
    if c_count == 1:
        return c_means[0]
        
    total_weight = np.sum(c_weights[:c_count])
    target_idx = q * total_weight
    
    cur_weight_accum = 0.0
    
    # 边界处理：min
    if target_idx <= c_weights[0] * 0.5:
        return c_means[0]
    
    # 边界处理：max
    if target_idx >= total_weight - c_weights[c_count-1] * 0.5:
        return c_means[c_count-1]

    # 遍历寻找区间
    for i in range(c_count - 1):
        w_i = c_weights[i]
        w_next = c_weights[i+1]
        
        # 质心一般被视为覆盖范围 [cum - w/2, cum + w/2]
        # 这里简化为两个质心中点之间的线性插值
        
        mid_cum_w_i = cur_weight_accum + w_i * 0.5
        mid_cum_w_next = cur_weight_accum + w_i + w_next * 0.5
        
        if mid_cum_w_i <= target_idx <= mid_cum_w_next:
            # 线性插值
            fraction = (target_idx - mid_cum_w_i) / (mid_cum_w_next - mid_cum_w_i)
            return c_means[i] + fraction * (c_means[i+1] - c_means[i])
            
        cur_weight_accum += w_i
        
    return c_means[c_count-1]



@njit(fastmath=True, cache=True, parallel=True)
def _batch_cdf_kernel(
    values: np.ndarray,
    c_means: np.ndarray,
    c_weights: np.ndarray,
    c_count: int
) -> np.ndarray:
    """
    批量计算数值对应的分位数 (CDF)。
    Input:
        values: 待查询的数值数组 (N,)
        c_means: 质心均值 (已排序)
        c_weights: 质心权重
        c_count: 有效质心数量
    Output:
        quantiles: 对应的分位数 (0.0 ~ 1.0)
    """
    n = len(values)
    results = np.empty(n, dtype=np.float64)
    
    # 0. 边界与空状态处理
    if c_count == 0:
        results[:] = np.nan
        return results
    
    total_weight = np.sum(c_weights[:c_count])
    if total_weight == 0:
        results[:] = 0.0
        return results

    # 1. 预计算每个质心中心的累积权重位置 (Midpoint Cumulative Weight)
    # 这对应于 quantile 函数中的插值节点
    mid_cum_weights = np.empty(c_count, dtype=np.float64)
    current_cum = 0.0
    for i in range(c_count):
        w = c_weights[i]
        mid_cum_weights[i] = current_cum + w * 0.5
        current_cum += w
        
    # 2. 并行批量处理查询
    # T-Digest 的 means 是严格有序的，利用这一特性进行二分查找
    valid_means = c_means[:c_count]
    
    for i in prange(n):
        val = values[i]
        
        # 处理 NaN
        if np.isnan(val):
            results[i] = np.nan
            continue
            
        # 边界情况：小于最小值
        if val <= valid_means[0]:
            results[i] = 0.0 # 或者 1.0 / total_weight (取决于定义)
            continue
            
        # 边界情况：大于最大值
        if val >= valid_means[c_count - 1]:
            results[i] = 1.0
            continue
            
        # 二分查找插入位置
        # side='right' -> valid_means[idx-1] <= val < valid_means[idx]
        idx = np.searchsorted(valid_means, val, side='right')
        
        # 既然处理了边界，idx 必然在 [1, c_count-1] 之间
        # 在 idx-1 和 idx 两个质心之间做线性插值
        
        # 获取左右质心的均值和累计权重位置
        mean_left = valid_means[idx - 1]
        mean_right = valid_means[idx]
        
        cw_left = mid_cum_weights[idx - 1]
        cw_right = mid_cum_weights[idx]
        
        # 插值计算
        # Fraction: val 在两个 mean 之间的位置 (0~1)
        fraction = (val - mean_left) / (mean_right - mean_left)
        
        # 映射到 weight 空间
        interpolated_weight = cw_left + fraction * (cw_right - cw_left)
        
        results[i] = interpolated_weight / total_weight
        
    return results

# ==========================================
# Part 2: Python Class Wrapper (State Mgmt)
# ==========================================

class TDigestNumba:
    """
    T-Digest 的高性能 Python 包装器。
    非线程安全（通常一个 scale 一个实例，串行写入）。
    """
    def __init__(self, delta: float = 300.0, buffer_size: int = 5000):
        """
        :param delta: 压缩因子。越大精度越高，内存占用越大。
                      Delta=300 此时质心数约为 ~150-200，足以应对 10^7 数据量的 0.01% 尾部精度。
        :param buffer_size: 缓冲区大小，满后触发合并。
        """
        self.delta = float(delta)
        # T-Digest 理论最大质心数约为 2 * delta 或更小，预留稍微多一点空间防止溢出
        self.max_capacity = int(delta * 2 + 50) 
        self.buffer_limit = buffer_size
        
        # 状态数组 (Pre-allocated)
        self.means = np.zeros(self.max_capacity, dtype=np.float64)
        self.weights = np.zeros(self.max_capacity, dtype=np.float64)
        self.n_centroids = 0
        
        # 写入缓冲区
        self.buffer = np.zeros(self.buffer_limit * 2, dtype=np.float64) # 预留空间
        self.buffer_count = 0
        
    def update(self, values: np.ndarray):
        """
        批量更新数据。
        """
        # 如果输入太大，分块处理以避免内存爆炸
        n = len(values)
        cursor = 0
        while cursor < n:
            remaining = n - cursor
            space = self.buffer_limit - self.buffer_count
            
            chunk_size = min(remaining, space)
            
            # Copy data to buffer
            self.buffer[self.buffer_count : self.buffer_count + chunk_size] = values[cursor : cursor + chunk_size]
            self.buffer_count += chunk_size
            cursor += chunk_size
            
            # Buffer full? Merge.
            if self.buffer_count >= self.buffer_limit:
                self._flush()
                
    def _flush(self):
        if self.buffer_count == 0:
            return
            
        self.means, self.weights, self.n_centroids = _process_buffer(
            self.means, self.weights, self.n_centroids,
            self.buffer, self.buffer_count,
            self.delta, self.max_capacity
        )
        self.buffer_count = 0

    def quantile(self, q: float) -> float:
        """获取分位数 (0 <= q <= 1)"""
        # 查询前必须清空缓冲区
        if self.buffer_count > 0:
            self._flush()
        return _get_quantile(self.means, self.weights, self.n_centroids, q)

    def cdf(self, values: np.ndarray) -> np.ndarray:
        """
        批量计算 CDF (分位数)。
        :param values: 1D numpy array of scores
        :return: 1D numpy array of quantiles (0~1)
        """
        # 必须先合并缓冲区，否则新数据的分布未被统计
        if self.buffer_count > 0:
            self._flush()
            
        return _batch_cdf_kernel(
            values, 
            self.means, 
            self.weights, 
            self.n_centroids
        )
        
    def reset(self):
        self.n_centroids = 0
        self.buffer_count = 0

# ==========================================
# Part 3: Production Manager
# ==========================================

class QuantileTracker:
    """
    管理 10 个尺度的时间序列模型分位数统计。
    """
    def __init__(self, num_variables: int = 10, delta: float = 500.0):
        # Delta 设为 500 以保证极高的尾部精度 (Top 0.01%)
        self.num_variables = num_variables
        self.digests = [TDigestNumba(delta=delta) for _ in range(num_variables)]
        
    def update_batch(self, data: np.ndarray):
        """
        核心输入接口。
        :param data: Shape (N_samples, num_scales) 
                     例如 (5000股票 * 1分钟, 10尺度)
        """
        assert data.shape[1] == self.num_variables, "Data dimension mismatch with configured scales"
        
        # 按列遍历，分别更新每个尺度的 Digest
        # 由于是列式处理，numpy slice 是视图，效率很高
        for i in range(self.num_variables):
            # 过滤 NaN (量化数据常态)
            col_data = data[:, i]
            valid_mask = ~np.isnan(col_data)
            if np.any(valid_mask):
                self.digests[i].update(col_data[valid_mask])

    def get_cdf_matrix(self, scores: np.ndarray) -> np.ndarray:
        """
        将 (N, D) 的分数矩阵转换为对应的分位数矩阵。
        
        :param scores: Shape (N, D), N samples, D scales.
        :return: Shape (N, D), values between 0.0 and 1.0.
        """
        N, D = scores.shape
        assert D == self.num_variables, "Input scales dim must match tracker scales"
        
        output = np.empty((N, D), dtype=np.float64)
        
        for i in range(D):
            col_scores = scores[:, i]
            output[:, i] = self.digests[i].cdf(col_scores)
        return output

    def get_tail_statistics(self) -> Dict[int, Dict[str, float]]:
        """
        获取每日报告所需的头部高精度分位数。
        """
        stats = {}
        target_quantiles = [0.9, 0.95, 0.99, 0.999, 0.9999] # Top 10%, 1%, 0.1%, 0.01%
        
        for i in range(self.num_variables):
            scale_stats = {}
            for q in target_quantiles:
                # 获取右尾 (Top X%)
                val_high = self.digests[i].quantile(q)
                # 获取左尾 (Bottom X%)
                val_low = self.digests[i].quantile(1.0 - q)
                
                scale_stats[f"top_{q:.4f}"] = val_high
                scale_stats[f"bot_{1-q:.4f}"] = val_low
            
            stats[i] = scale_stats
        return stats

    def reset_daily(self):
        """每天收盘后调用"""
        for d in self.digests:
            d.reset()

# ==========================================
# Part 4: Usage Example (Simulation)
# ==========================================

if __name__ == "__main__":
    import time
    
    # 1. 初始化
    tracker = MultiScaleQuantileTracker(num_scales=10, delta=500)
    
    # 2. 模拟生产数据：每分钟 5000 个股票，10 个尺度
    n_stocks = 5000
    n_scales = 10
    drift = np.random.randn(n_scales) * 0.1
    noise = np.random.randn(n_stocks, n_scales) + drift
    tracker.update_batch(noise)

    print("Starting simulation...")
    t0 = time.time()
    
    # 模拟 240 分钟的数据流
    for minute in range(10): # 演示仅跑10分钟，生产环境跑240
        # 模拟数据：假设均值方差随机漂移 (Non-stationary)
        drift = np.random.randn(n_scales) * 0.1
        noise = np.random.randn(n_stocks * 40, n_scales) + drift
        
        # 更新
        tracker.update_batch(noise)
        
    t1 = time.time()
    print(f"Processed 10 mins of data (Total {10 * n_stocks * n_scales} points) in {t1-t0:.4f}s")
    
    # 3. 获取高精度分位数报告
    report = tracker.get_tail_statistics()
    print("\n--- Scale 0 Statistics (Tail Precision Check) ---")
    for k, v in report[0].items():
        print(f"{k}: {v:.6f}")