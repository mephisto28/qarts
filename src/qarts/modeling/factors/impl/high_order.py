import numpy as np
from numba import njit, prange

from ..ops import ContextOps
from ..context import ContextSrc
from ..base import register_factor, FactorFromDailyAndIntraday
from ..constants import FactorNames

__all__ = [
    'DailySkewness',
    'DailyKurtosis'
]

@register_factor(FactorNames.DAILY_SKEWNESS)
class DailySkewness(FactorFromDailyAndIntraday):
    num_daily_fields = 1 # daily_return
    num_intraday_fields = 0
    num_factor_cache_fields = 0 # daily_mom_1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 5, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_return_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.return_field = self.input_fields[ContextSrc.FACTOR_CACHE][0] \
            if ContextSrc.FACTOR_CACHE in self.input_fields else self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        default_field = 'daily_return'
        if self.history_return_field != default_field:
            return f'{self.history_return_field}_{FactorNames.DAILY_SKEWNESS}_{self.window}'
        return f'{FactorNames.DAILY_SKEWNESS}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):

        hist_window = self.window - 1
        h_s1 = ops.history_window_pow_sum(self.history_return_field, 1, window=hist_window)
        h_s2 = ops.history_window_pow_sum(self.history_return_field, 2, window=hist_window)
        h_s3 = ops.history_window_pow_sum(self.history_return_field, 3, window=hist_window)
        h_cnt = ops.history_window_pow_sum(self.history_return_field, 0, window=hist_window)
        
        today_ret = ops.now(self.return_field)
        
        # 3. 使用 Numba 并行计算最终因子
        _calc_skewness_kernel(
            h_s1, h_s2, h_s3, h_cnt, 
            today_ret, 
            out, 
            min_periods=3
        )


@register_factor(FactorNames.DAILY_KURTOSIS)
class DailyKurtosis(FactorFromDailyAndIntraday):
    num_daily_fields = 1 # daily_return
    num_intraday_fields = 0
    num_factor_cache_fields = 0 # intraday_mom

    def __init__(self, input_fields: dict[str, list[str]], window: int = 5, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_return_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.return_field = self.input_fields[ContextSrc.FACTOR_CACHE][0] \
            if ContextSrc.FACTOR_CACHE in self.input_fields else self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        default_field = 'daily_return'
        if self.history_return_field != default_field:
            return f'{self.history_return_field}_{FactorNames.DAILY_KURTOSIS}_{self.window}'
        return f'{FactorNames.DAILY_KURTOSIS}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        today_ret = ops.now(self.return_field)
        hist_window = self.window - 1
        h_s1 = ops.history_window_pow_sum(self.history_return_field, 1, window=hist_window)
        h_s2 = ops.history_window_pow_sum(self.history_return_field, 2, window=hist_window)
        h_s3 = ops.history_window_pow_sum(self.history_return_field, 3, window=hist_window)
        h_s4 = ops.history_window_pow_sum(self.history_return_field, 4, window=hist_window)
        h_cnt = ops.history_window_pow_sum(self.history_return_field, 0, window=hist_window)

        _compute_kurtosis_kernel(
            out, 
            today_ret,
            h_s1, h_s2, h_s3, h_s4, h_cnt,
            self.window
        )


@njit(parallel=True, fastmath=True, cache=True)
def _calc_skewness_kernel(
    h_s1: np.ndarray, h_s2: np.ndarray, h_s3: np.ndarray, h_cnt: np.ndarray, 
    today_ret: np.ndarray, out: np.ndarray, min_periods: int
):
    """
    Kernel Function: 结合历史统计量与今日分时数据计算偏度
    复杂度: O(N * T_intra)
    """
    n_stocks, n_ticks = today_ret.shape
    
    for i in prange(n_stocks):
        # 加载该标的历史窗口的统计量 (Scalar)
        base_s1 = h_s1[i, 0]
        base_s2 = h_s2[i, 0]
        base_s3 = h_s3[i, 0]
        base_n = h_cnt[i, 0]
        
        for j in range(n_ticks):
            r = today_ret[i, j]
            
            # 处理 NaN 数据
            if np.isnan(r):
                out[i, j] = np.nan
                continue
            
            # 增量更新统计量 (History + Today)
            cur_n = base_n + 1
            
            # 窗口有效性检查
            if cur_n < min_periods:
                out[i, j] = np.nan
                continue

            cur_s1 = base_s1 + r
            cur_s2 = base_s2 + r * r
            cur_s3 = base_s3 + r * r * r
            
            # 计算矩 (Moments)
            # Mean = S1 / N
            mean = cur_s1 / cur_n
            
            # Variance = E[x^2] - (E[x])^2
            # 使用 maximum 防止浮点误差导致负数
            var = np.maximum((cur_s2 / cur_n) - (mean * mean), 0.0)
            
            if var < 1e-12:  # 处理极小方差/除零
                out[i, j] = np.nan
                continue
            
            std = np.sqrt(var)
            
            # 偏度公式 (Population Skewness): 
            # M3 = E[x^3] - 3*mean*E[x^2] + 2*mean^3
            # Skew = M3 / std^3
            term1 = cur_s3 / cur_n
            term2 = 3 * mean * (cur_s2 / cur_n)
            term3 = 2 * (mean * mean * mean)
            
            m3 = term1 - term2 + term3
            out[i, j] = m3 / (std * std * std)


@njit(parallel=True, fastmath=True, cache=True)
def _compute_kurtosis_kernel(
    out: np.ndarray,
    today_ret: np.ndarray,
    h_s1: np.ndarray, 
    h_s2: np.ndarray, 
    h_s3: np.ndarray, 
    h_s4: np.ndarray,
    h_count: np.ndarray,
    window: int
):
    """
    核心计算逻辑：利用历史 Sum 和 今日 Value 合成新的矩，并计算峰度。
    复杂度: O(N * T)
    """
    N, T_intra = today_ret.shape
    
    # 预计算系数 (基于固定窗口 n)
    # 若 n < 4，公式分母为0，需要在循环中处理
    n = window
    
    # 边界检查常量
    min_periods = 4 
    
    if n < min_periods:
        out[:] = np.nan
        return

    # 峰度公式系数
    # G2 = C1 * (M4 / s^4) - C2
    # s^2 = SSD / (n-1) -> s^4 = SSD^2 / (n-1)^2
    # -> term = M4 * (n-1)^2 / SSD^2
    
    c1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    c2 = (3 * (n - 1)**2) / ((n - 2) * (n - 3))
    
    for i in prange(N):
        # 读取该标的历史累积值 (Last W-1 days)
        # 如果历史数据不足 (NaN)，则 h_count < window - 1
        # 这里简化处理：要求历史必须完整，否则输出 NaN
        if h_count[i] < (window - 1):
            out[i, :] = np.nan
            continue
            
        _h_s1 = h_s1[i, 0]
        _h_s2 = h_s2[i, 0]
        _h_s3 = h_s3[i, 0]
        _h_s4 = h_s4[i, 0]

        for t in range(T_intra):
            val = today_ret[i, t]
            
            if np.isnan(val):
                out[i, t] = np.nan
                continue

            # 增量计算当前窗口的 Sum (History + Current)
            val_sq = val * val
            s1 = _h_s1 + val
            s2 = _h_s2 + val_sq
            s3 = _h_s3 + val_sq * val
            s4 = _h_s4 + val_sq * val_sq
            
            # 1. 计算均值
            mu = s1 / n
            
            # 2. 计算 SSD (Sum of Squared Deviations) 用于方差
            # SSD = S2 - n * mu^2 
            # (或者使用 S2 - S1^2/n 可能会有精度损失，但在这里为了性能折衷)
            ssd = s2 - (s1 * s1) / n
            
            if ssd <= 1e-12: # 极小方差保护
                out[i, t] = np.nan
                continue
                
            # 3. 计算 4阶中心矩 M4
            # M4 = S4 - 4*mu*S3 + 6*mu^2*S2 - 3*n*mu^4
            mu2 = mu * mu
            mu3 = mu2 * mu
            mu4 = mu2 * mu2
            
            m4 = s4 - 4.0 * mu * s3 + 6.0 * mu2 * s2 - 3.0 * n * mu4
            
            # 4. 组合最终结果
            # Excess Kurtosis
            term = (m4 * (n - 1)**2) / (ssd * ssd)
            res = c1 * term - c2
            
            out[i, t] = res
