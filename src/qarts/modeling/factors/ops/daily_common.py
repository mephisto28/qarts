import inspect
import functools
import typing as T
from enum import Enum, auto
from dataclasses import dataclass, field

import numpy as np
from .base import BaseOps
from .utils import context_static_cache, expand_tdim
from .. import kernels as kns
from ..context import ContextSrc


class DailyOps(BaseOps):

    @expand_tdim
    def yesterday(self, field: str) -> np.ndarray:
        assert ContextSrc.DAILY_QUOTATION in self.context.blocks
        block = self.context.blocks[ContextSrc.DAILY_QUOTATION]
        return block.get_view(field)[:, -1]

    @context_static_cache
    def _history_pow_cumsum(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'history_pow_cumsum', power: int = 1) -> np.ndarray:
        if input_value is None:
            data = self.context.get_field(ContextSrc.DAILY_QUOTATION, field=field)
        else:
            data = input_value
        if power == 0:
            pow_data = (~np.isnan(data)).astype(np.float32)
        elif power == 1:
            pow_data = data
        else:
            pow_data = np.power(data, power)
        return np.nancumsum(pow_data, axis=1)

    @context_static_cache
    def _history_prod_cumsum(self, field: T.Tuple[str, ...], input_value: T.Optional[np.ndarray] = None, name: str = 'history_prod_cumsum') -> np.ndarray:
        if input_value is None:
            values = [
                self.context.get_field(ContextSrc.DAILY_QUOTATION, field=f) 
                for f in field
            ]
        else:
            values = input_value
        out = values[0].astype(np.float64, copy=True)
        for value in values[1:]:
            out *= value
        return np.nancumsum(out, axis=1).astype(np.float32)

    @context_static_cache
    def _history_suffix_high(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'history_suffix_high') -> np.ndarray:
        if input_value is None:
            values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field) # N, T
        else:
            values = input_value
        rev = values[:, ::-1]
        return np.fmax.accumulate(rev, axis=-1)[::-1]

    @context_static_cache
    def _history_suffix_low(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'history_suffix_low') -> np.ndarray:
        if input_value is None:
            values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field) # N, T
        else:
            values = input_value
        rev = values[:, ::-1]
        return np.fmin.accumulate(rev, axis=-1)[::-1]

    @context_static_cache
    def _history_recent_value_wo_nan(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'history_values_ffilled') -> np.ndarray:
        if input_value is None:
            values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field) # N, T
        else:
            values = input_value
        out = np.empty_like(values)
        kns.ffill2d(values, out)
        return out

    @expand_tdim
    def history_ma(self, field: str, name: str = 'history_ma', window: int = 20) -> np.ndarray:
        prefix_sum = self._history_pow_cumsum(field, power=1)
        prefix_count = self._history_pow_cumsum(field, power=0)
        ma = (prefix_sum[:, -1] - prefix_sum[:, -window-1]) / (prefix_count[:, -1] - prefix_count[:, -window-1])
        return ma
    
    @expand_tdim
    def history_valid_ratio(self, field: str, name: str = 'history_valid_ratio', window: int = 20) -> np.ndarray:
        prefix_count = self._history_pow_cumsum(field, power=0)
        valid_ratio = (prefix_count[:, -1] - prefix_count[:, -window]) / window
        return valid_ratio
    
    @expand_tdim
    def hisotry_vwap(self, fields: T.Tuple[str, ...], name: str = 'history_vwap', window: int = 20) -> np.ndarray:
        price_field, volume_field = fields
        weighted_prefix_sum = self._history_prod_cumsum((price_field, volume_field))
        volume_prefix_sum = self._history_pow_cumsum(volume_field, power=1)
        weighted_sum = weighted_prefix_sum[:, -1] - weighted_prefix_sum[:, -window-1]
        normalizer = volume_prefix_sum[:, -1] - volume_prefix_sum[:, -window-1]
        return (weighted_sum / normalizer).astype(np.float32)

    @expand_tdim
    def history_high(self, field: str, name: str = 'history_high', window: int = 20) -> np.ndarray:
        suffix_high = self._history_suffix_high(field)
        return suffix_high[:, -window]
    
    @expand_tdim
    def history_low(self, field: str, name: str = 'history_low', window: int = 20) -> np.ndarray:
        suffix_low = self._history_suffix_low(field)
        return suffix_low[:, -window]

    @expand_tdim
    def history_lag(self, field: str, name: str = 'history_lag', window: int = 1) -> np.ndarray:
        values = self._history_recent_value_wo_nan(field)
        return values[:, -window]
    
    @expand_tdim
    def today_open(self, field: str, name: str = 'today_open') -> np.ndarray:
        assert ContextSrc.INTRADAY_QUOTATION in self.context.blocks
        block = self.context.blocks[ContextSrc.INTRADAY_QUOTATION]
        field_values = block.get_view(field)
        return field_values[:, 0]

    @expand_tdim
    def history_sq_cumsum_with_count(self, field: str, window: int = 20) -> np.ndarray:
        hist_rsq_cumsum = self._history_pow_cumsum(field, power=2)
        hist_count_cumsum = self._history_pow_cumsum(field, power=0) # non-nan count
        hist_ss = hist_rsq_cumsum[:, -1] - hist_rsq_cumsum[:, -window]
        valid_count = hist_count_cumsum[:, -1] - hist_count_cumsum[:, -window]
        return hist_ss, valid_count
