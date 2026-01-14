import numpy as np
from numba import njit, prange

from ..ops import ContextOps
from ..ops.utils import expand_tdim
from ..context import ContextSrc
from ..base import register_factor, FactorFromDailyAndIntraday
from ..constants import FactorNames

__all__ = [
    'ReturnVolumeCorr',
]



@register_factor(FactorNames.RV_CORR)
class ReturnVolumeCorr(FactorFromDailyAndIntraday):
    
    def __init__(self, input_fields: dict[str, list[str]], window: int = 10, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        # Daily fields
        self.d_ret = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.d_vol = self.input_fields[ContextSrc.DAILY_QUOTATION][1]
        # Intraday fields
        if ContextSrc.FACTOR_CACHE in self.input_fields:
            self.i_ret = self.input_fields[ContextSrc.FACTOR_CACHE][0]
            self.i_vol = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]
        else:
            self.i_ret = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]
            self.i_vol = self.input_fields[ContextSrc.INTRADAY_QUOTATION][1]

    @property
    def name(self) -> str:
        if self.d_ret != 'daily_return':
            return f'{self.d_ret}_{FactorNames.RV_CORR}_{self.window}'
        return f'rv_corr_{self.window}'

    @staticmethod
    @expand_tdim 
    def _get_hist_stats(ops: ContextOps, f_ret: str, f_vol: str, window: int):
        # We need historical stats for the past (window - 1) days
        # Since history arrays exclude today, index -(window-1) gives the suffix sum of relevant past days.
        hist_win = window - 1
        if hist_win <= 0:
            # Special case: window=1 (correlation undefined usually, but logic should hold for N=1 if kernel handles it)
            # Return zeros using ops shape reference
            dummy = ops.history_pow_cumsum(f_ret, power=0)[:, -1]
            return tuple(np.zeros_like(dummy) for _ in range(6))

        # 1. Sum X, Sum X^2, Count
        h_cnt_cum = ops.history_pow_cumsum(f_ret, power=0)
        h_sx_cum = ops.history_pow_cumsum(f_ret, power=1)
        h_sx2_cum = ops.history_pow_cumsum(f_ret, power=2)
        
        # 2. Sum Y, Sum Y^2
        h_sy_cum = ops.history_pow_cumsum(f_vol, power=1)
        h_sy2_cum = ops.history_pow_cumsum(f_vol, power=2)
        
        # 3. Sum XY
        h_sxy_cum = ops.history_prod_cumsum((f_ret, f_vol))

        # Extract slices (Using negative indexing for suffix sums)
        # Suffix sum at index -k represents sum of last k elements
        idx = -hist_win
        return (
            h_cnt_cum[:, idx],
            h_sx_cum[:, idx], h_sx2_cum[:, idx],
            h_sy_cum[:, idx], h_sy2_cum[:, idx],
            h_sxy_cum[:, idx]
        )

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        # 1. Prepare Historical Statistics (Cached)
        # (N, 1) expanded due to @expand_tdim
        h_cnt, h_sx, h_sx2, h_sy, h_sy2, h_sxy = self._get_hist_stats(
            ops, self.d_ret, self.d_vol, self.window
        )
        
        # 2. Prepare Today's Data
        # (N, T)
        t_ret = ops.now(self.i_ret)
        t_vol = ops.get_field(ContextSrc.INTRADAY_QUOTATION, self.i_vol)
        current_time_fraction = ops.time_fraction()
        t_vol = t_vol / current_time_fraction

        # 3. Compute via Numba
        _calc_rolling_corr_intraday(
            out,
            h_cnt.flatten(), # Flatten to (N,) for numba kernel efficiency
            h_sx.flatten(), h_sx2.flatten(),
            h_sy.flatten(), h_sy2.flatten(),
            h_sxy.flatten(),
            t_ret, t_vol,
            min_periods=max(2, self.window // 2)
        )


@register_factor(FactorNames.DAILY_ON_BALANCE_VOLUME_RATIO)
class DailyOBVRatio(FactorFromDailyAndIntraday):
    num_daily_fields = 2

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_return_field, self.history_volume_field = self.input_fields[ContextSrc.DAILY_QUOTATION]
        if ContextSrc.FACTOR_CACHE in self.input_fields:
            self.intraday_return_field = self.input_fields[ContextSrc.FACTOR_CACHE][0]
            self.intraday_volume_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]
        else:
            self.intraday_return_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]
            self.intraday_volume_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][1]

    @property
    def name(self) -> str:
        if self.history_return_field != 'daily_return':
            return f'{self.history_return_field}_{FactorNames.DAILY_ON_BALANCE_VOLUME_RATIO}_{self.window}'
        return f'{FactorNames.DAILY_ON_BALANCE_VOLUME_RATIO}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        history_volume_ma = ops.history_window_ma(self.history_volume_field, window=self.window)
        history_volume = ops.get_field(ContextSrc.DAILY_QUOTATION, self.history_volume_field)
        history_return = ops.get_field(ContextSrc.DAILY_QUOTATION, self.history_return_field)
        history_return_sign = np.sign(history_return)
        obv_volume_suffix_sum = ops.history_prod_cumsum((self.history_return_field, self.history_volume_field), (history_return_sign, history_volume))
        obv_volume_mean = obv_volume_suffix_sum[:, -self.window+1] / self.window
        current_volume = ops.now(self.intraday_volume_field)
        current_return = ops.now(self.intraday_return_field)
        current_time_fraction = ops.time_fraction()
        out[:] = (current_volume * np.sign(current_return)) / current_time_fraction
        out /= self.window
        out += obv_volume_mean[:, None]
        out /= history_volume_ma + 1


@register_factor(FactorNames.TODAY_AMOUNT_RATIO)
class TodayAmountRatio(FactorFromDailyAndIntraday):
    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.certain_amount_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]
        self.total_amount_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][1]

    @property
    def name(self) -> str:
        return f'{FactorNames.TODAY_AMOUNT_RATIO}_{self.certain_amount_field}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        certain_amount = ops.today_cumsum(self.certain_amount_field)
        total_amount = ops.today_cumsum(self.total_amount_field)
        np.nan_to_num(certain_amount, copy=False, nan=0)
        out[:] = certain_amount / (total_amount + 1)


@njit(parallel=True, fastmath=True, cache=True)
def _calc_rolling_corr_intraday(
    out: np.ndarray,
    h_cnt: np.ndarray,
    h_sx: np.ndarray, h_sx2: np.ndarray,
    h_sy: np.ndarray, h_sy2: np.ndarray,
    h_sxy: np.ndarray,
    t_x: np.ndarray, t_y: np.ndarray,
    min_periods: int
):
    """
    Core correlation calculation kernel using Pearson formula.
    Combined online update of sums and correlation computation.
    """
    n_assets, n_ticks = out.shape
    
    for i in prange(n_assets):
        # Cache historical scalars for this asset to registers
        hist_c = h_cnt[i]
        hist_x = h_sx[i]
        hist_x2 = h_sx2[i]
        hist_y = h_sy[i]
        hist_y2 = h_sy2[i]
        hist_xy = h_sxy[i]
        
        for t in range(n_ticks):
            vx = t_x[i, t]
            vy = t_y[i, t]
            
            # Check data validity
            if np.isnan(vx) or np.isnan(vy):
                out[i, t] = np.nan
                continue
            
            # Update sums with today's value
            n = hist_c + 1
            if n < min_periods:
                out[i, t] = np.nan
                continue
                
            sum_x = hist_x + vx
            sum_x2 = hist_x2 + vx * vx
            sum_y = hist_y + vy
            sum_y2 = hist_y2 + vy * vy
            sum_xy = hist_xy + vx * vy
            
            # Calculate Correlation: (N*SumXY - SumX*SumY) / sqrt(...)
            # Numerator
            cov_n = n * sum_xy - sum_x * sum_y
            
            # Denominator terms
            var_x_n = n * sum_x2 - sum_x * sum_x
            var_y_n = n * sum_y2 - sum_y * sum_y
            
            if var_x_n <= 1e-12 or var_y_n <= 1e-12:
                out[i, t] = np.nan
            else:
                out[i, t] = cov_n / np.sqrt(var_x_n * var_y_n)