import numpy as np
from numba import njit, prange

from .ops import ContextOps
from .context import ContextSrc
from .base import register_factor, FactorFromDailyAndIntraday
from .constants import FactorNames


@register_factor(FactorNames.DAILY_SKEWNESS)
class DailySkewness(FactorFromDailyAndIntraday):
    num_daily_fields = 1 # daily_return
    num_intraday_fields = 0
    num_factor_cache_fields = 1 # daily_mom_1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 5, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_return_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.return_field = self.input_fields[ContextSrc.FACTOR_CACHE][0]

    @property
    def name(self) -> str:
        return f'{FactorNames.DAILY_SKEWNESS}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):

        hist_window = self.window - 1
        h_s1 = ops.history_window_pow_sum(self.history_return_field, 1, window=hist_window)
        h_s2 = ops.history_window_pow_sum(self.history_return_field, 2, window=hist_window)
        h_s3 = ops.history_window_pow_sum(self.history_return_field, 3, window=hist_window)
        h_cnt = ops.history_window_pow_sum(self.history_return_field, 0, window=hist_window)
        
        today_ret = ops.now_factor(self.return_field)
        
        # 3. 使用 Numba 并行计算最终因子
        _calc_skewness_kernel(
            h_s1, h_s2, h_s3, h_cnt, 
            today_ret, 
            out, 
            min_periods=3
        )
        breakpoint()


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