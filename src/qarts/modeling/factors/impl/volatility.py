import numpy as np
from numba import njit, prange

from ..ops import ContextOps
from ..ops.utils import expand_tdim
from ..context import ContextSrc
from ..constants import FactorNames
from ..base import register_factor, FactorFromDailyAndIntraday
from .. import kernels as kns

__all__ = [
    'DailyVolatility',
    'DailyVolVol'
]

@register_factor(FactorNames.DAILY_VOLATILITY)
class DailyVolatility(FactorFromDailyAndIntraday):
    num_daily_fields = 1 # daily_return
    num_intraday_fields = -1
    num_factor_cache_fields = -1 # daily_mom_1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 5, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_return_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.return_field = self.input_fields[ContextSrc.FACTOR_CACHE][0] \
            if ContextSrc.FACTOR_CACHE in self.input_fields else self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        default_field = 'daily_return'
        if self.history_return_field != default_field:
            return f'{self.history_return_field}_{FactorNames.DAILY_VOLATILITY}_{self.window}'
        return f'{FactorNames.DAILY_VOLATILITY}_{self.window}'

    @staticmethod
    @expand_tdim
    def history_sq_cumsum_with_count(ops: ContextOps, field: str, window: int = 20) -> np.ndarray:
        hist_rsq_cumsum = ops.history_pow_cumsum(field, power=2)
        hist_count_cumsum = ops.history_pow_cumsum(field, power=0) # non-nan count
        hist_ss = hist_rsq_cumsum[:, -window]
        valid_count = hist_count_cumsum[:, -window]
        return hist_ss, valid_count

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        today_ret = ops.now(self.return_field)
        hist_ss, valid_count = self.history_sq_cumsum_with_count(ops, self.history_return_field, self.window - 1)
        np.square(today_ret, out=out) # inplace op to reduce memory allocation cost
        out += hist_ss
        out /= (valid_count + 1)
        np.sqrt(out, out=out)


@register_factor(FactorNames.DAILY_VOLVOL)
class DailyVolVol(FactorFromDailyAndIntraday):

    num_daily_fields = 1 
    num_intraday_fields = -1
    num_factor_cache_fields = -1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 20, window2: int = 20, **kwargs):
        super().__init__(input_fields=input_fields, window=window+window2, **kwargs)
        self.vol_window = window
        self.volvol_window = window2
        self.daily_ret_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.intraday_ret_field = self.input_fields[ContextSrc.FACTOR_CACHE][0] \
            if ContextSrc.FACTOR_CACHE in self.input_fields else self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        default_field = 'daily_return'
        if self.daily_ret_field != default_field:
            return f'{self.daily_ret_field}_{FactorNames.DAILY_VOLVOL}_{self.vol_window}_{self.volvol_window}'
        return f'{FactorNames.DAILY_VOLVOL}_{self.vol_window}_{self.volvol_window}'
    
    @staticmethod
    def history_vol_sequence(ops: ContextOps, field: str, window: int, 
                             input_value = None, 
                             name: str = 'history_vol_seq') -> np.ndarray:

        r2_mean = ops.history_rolling_moment_sequence(field, power=2, window=window, input_value=input_value)
        vol_seq = np.sqrt(r2_mean)
        return vol_seq
    
    @classmethod
    def history_volvol_prepared_stats(cls, ops: ContextOps, field: str, vol_window: int, name: str = 'hist_volvol_stats'):
        vol_seq = cls.history_vol_sequence(ops, field, window=vol_window)
        vol_sum_suffix = vol_seq.copy()
        kns.reverse_cumsum_2d(vol_sum_suffix, vol_sum_suffix)

        vol_seq_origin = cls.history_vol_sequence(ops, field, window=vol_window) # 命中缓存
        vol_sq = np.square(vol_seq_origin, out=vol_seq_origin)
        vol_sq_sum_suffix = kns.reverse_cumsum_2d(a=vol_sq, out=vol_sq)
        return vol_sum_suffix, vol_sq_sum_suffix

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        today_ret = ops.now(self.intraday_ret_field) 
        history_r2_suffix = ops.history_pow_cumsum(self.daily_ret_field, power=2)
        hist_vol_sum, hist_vol_sq_sum = self.history_volvol_prepared_stats(ops, self.daily_ret_field, self.vol_window)

        w1 = self.vol_window
        w2 = self.volvol_window
        
        hist_sum_r2 = history_r2_suffix[:, -(w1-1)].copy() 
        current_vol = np.square(today_ret) 
        current_vol += hist_sum_r2[:, None] 
        current_vol /= w1
        np.sqrt(current_vol, out=current_vol) # current_vol 现在是 Vol_today
        
        h_v_sum = hist_vol_sum[:, -(w2-1)][:, None]   # (N, 1)
        h_v_sq_sum = hist_vol_sq_sum[:, -(w2-1)][:, None] # (N, 1)
        np.square(current_vol, out=out) # out = Vol_today^2
        out += h_v_sq_sum
        out /= w2 # out = E[X^2]
        
        vol_today = np.sqrt((np.square(today_ret) + hist_sum_r2[:, None]) / w1) # (N, T) 
        mean_sq = (np.square(vol_today) + h_v_sq_sum) / w2
        
        mean_val = (vol_today + h_v_sum) / w2
        np.square(mean_val, out=mean_val)
        
        np.subtract(mean_sq, mean_val, out=out)
        out[out < 0] = 0.0
        np.sqrt(out, out=out)
