import numpy as np
import numba as nb

from ..ops import ContextOps
from ..context import ContextSrc
from ..base import register_factor, FactorFromDailyAndIntraday
from ..constants import FactorNames

__all__ = [
    'TodayVolatility',
    'TodaySkewness'
]


@register_factor(FactorNames.TODAY_MOM)
class TodayMomentum(FactorFromDailyAndIntraday):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.today_momentum_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0] if ContextSrc.INTRADAY_QUOTATION in self.input_fields \
            else self.input_fields[ContextSrc.FACTOR_CACHE][0]

    @property
    def name(self) -> str:
        default_field = 'intraday_mom'
        if self.today_momentum_field != default_field:
            return f'{self.today_momentum_field}_{FactorNames.TODAY_MOM}'
        return f'{FactorNames.TODAY_MOM}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        out[:] = ops.now(self.today_momentum_field) - ops.today_open(self.today_momentum_field)[:, :1]



@register_factor(FactorNames.TODAY_VOLATILITY)
class TodayVolatility(FactorFromDailyAndIntraday):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.today_momentum_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0] if ContextSrc.INTRADAY_QUOTATION in self.input_fields \
            else self.input_fields[ContextSrc.FACTOR_CACHE][0]

    @property
    def name(self) -> str:
        default_field = 'intraday_mom'
        if self.today_momentum_field != default_field:
            return f'{self.today_momentum_field}_{FactorNames.TODAY_VOLATILITY}'
        return f'{FactorNames.TODAY_VOLATILITY}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        diff_values = ops.today_diff(self.today_momentum_field)
        cum_realized_vol(diff_values, out)


@register_factor(FactorNames.TODAY_SKEWNESS)
class TodaySkewness(FactorFromDailyAndIntraday):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.today_momentum_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0] if ContextSrc.INTRADAY_QUOTATION in self.input_fields \
            else self.input_fields[ContextSrc.FACTOR_CACHE][0]

    @property
    def name(self) -> str:
        default_field = 'intraday_mom'
        if self.today_momentum_field != default_field:
            return f'{self.today_momentum_field}_{FactorNames.TODAY_SKEWNESS}'
        return f'{FactorNames.TODAY_SKEWNESS}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        diff_values = ops.today_diff(self.today_momentum_field)
        cum_skewness(diff_values, out)


@nb.njit(parallel=True)
def cum_realized_vol(x: np.ndarray, out: np.ndarray):
    N, T = x.shape
    for i in nb.prange(N):
        s = 0.0
        c = 0
        for t in range(T):
            v = x[i, t]
            if v == v:
                s += v * v
                c += 1

            if c > 1:
                out[i, t] = np.sqrt(s / (c - 1))
            else:
                out[i, t] = s * 0.5
    return out


@nb.njit(parallel=True)
def cum_skewness(returns: np.ndarray, out: np.ndarray):
    """
    Numba implementation for expanding window skewness (Fisher-Pearson adjusted).
    Complexity: O(N*T)
    """
    n_assets, n_times = returns.shape
    
    for i in nb.prange(n_assets):
        count = 0
        sum_x = 0.0
        sum_sq = 0.0
        sum_cub = 0.0
        
        for t in range(n_times):
            val = returns[i, t]
            
            # Handle NaNs in input (skip update, output nan)
            if val != val:
                out[i, t] = np.nan
                continue
            
            count += 1
            sum_x += val
            sum_sq += val * val
            sum_cub += val * val * val
            
            # Skewness requires at least 3 samples for Fisher-Pearson correction
            if count < 3:
                out[i, t] = 0
                continue
            
            # Calculate raw moments centered components
            mean = sum_x / count
            m2 = sum_sq - sum_x * mean
            
            # Handle constant value / zero variance
            if m2 <= 1e-18:
                out[i, t] = 0
                continue
            
            m3 = sum_cub - 3 * mean * sum_sq + 2 * count * (mean * mean * mean)
            bias_correction = (count * np.sqrt(count - 1)) / (count - 2)
            skew = bias_correction * (m3 / np.power(m2, 1.5))
            
            out[i, t] = skew