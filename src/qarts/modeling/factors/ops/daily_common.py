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
    def history_pow_cumsum(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'history_pow_cumsum', power: int = 1) -> np.ndarray:
        if input_value is None:
            data = self.context.get_field(ContextSrc.DAILY_QUOTATION, field=field)
        else:
            data = input_value
        if power == 0:
            pow_data = (~np.isnan(data)).astype(np.float32)
        elif power == 1:
            pow_data = data.copy()
        else:
            pow_data = np.power(data, power)
        return kns.reverse_cumsum_2d(a=pow_data, out=pow_data)

    @context_static_cache
    def history_prod_cumsum(self, field: T.Tuple[str, ...], input_value: T.Optional[np.ndarray] = None, name: str = 'history_prod_cumsum') -> np.ndarray:
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
        return kns.reverse_cumsum_2d(a=out, out=out)

    @context_static_cache
    def history_suffix_high(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'history_suffix_high') -> np.ndarray:
        if input_value is None:
            values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field) # N, T
        else:
            values = input_value
        rev = values[:, ::-1]
        return np.fmax.accumulate(rev, axis=-1)[::-1]

    @context_static_cache
    def history_suffix_low(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'history_suffix_low') -> np.ndarray:
        if input_value is None:
            values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field) # N, T
        else:
            values = input_value
        rev = values[:, ::-1]
        return np.fmin.accumulate(rev, axis=-1)[::-1]

    @context_static_cache
    def history_recent_value_wo_nan(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'history_values_ffilled') -> np.ndarray:
        if input_value is None:
            values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field) # N, T
        else:
            values = input_value
        out = np.empty_like(values)
        kns.ffill2d(values, out)
        return out

    @context_static_cache
    def history_winsorized_value(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'history_winsorized_value', window: int = 126) -> np.ndarray:
        if input_value is None:
            values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field) # N, T
        else:
            values = input_value
        std = ((values ** 2).mean(axis=1) ** 0.5)[:, np.newaxis]
        return np.clip(values, -std * 3, std * 3)

    @expand_tdim
    def history_window_pow_sum(self, field: str, power: int, window: int) -> np.ndarray:
        cumsum = self.history_pow_cumsum(field, power=power)
        return cumsum[:, -window]

    @expand_tdim
    def history_window_ma(self, field: str, name: str = 'history_ma', window: int = 20) -> np.ndarray:
        prefix_sum = self.history_pow_cumsum(field, power=1)
        prefix_count = self.history_pow_cumsum(field, power=0)
        ma = prefix_sum[:, -window] / prefix_count[:, -window]
        return ma
    
    @expand_tdim
    def history_window_valid_ratio(self, field: str, name: str = 'history_valid_ratio', window: int = 20) -> np.ndarray:
        prefix_count = self.history_pow_cumsum(field, power=0)
        valid_ratio = prefix_count[:, -window] / window
        return valid_ratio
    
    @expand_tdim
    def hisotry_window_vwap(self, fields: T.Tuple[str, ...], name: str = 'history_vwap', window: int = 20) -> np.ndarray:
        price_field, volume_field = fields
        weighted_prefix_sum = self.history_prod_cumsum((price_field, volume_field))
        volume_prefix_sum = self.history_pow_cumsum(volume_field, power=1)
        weighted_sum = weighted_prefix_sum[:, -window]
        normalizer = volume_prefix_sum[:, -window]
        vwap = (weighted_sum / (normalizer + 1e-6)).astype(np.float32)
        vwap = np.where(normalizer == 0, self.history_window_ma(price_field, window=window)[:, 0], vwap)
        return vwap

    @expand_tdim
    def history_window_high(self, field: str, name: str = 'history_high', window: int = 20) -> np.ndarray:
        suffix_high = self.history_suffix_high(field)
        return suffix_high[:, -window]
    
    @expand_tdim
    def history_window_low(self, field: str, name: str = 'history_low', window: int = 20) -> np.ndarray:
        suffix_low = self.history_suffix_low(field)
        return suffix_low[:, -window]

    @expand_tdim
    def history_lag(self, field: str, name: str = 'history_lag', window: int = 1) -> np.ndarray:
        values = self.history_recent_value_wo_nan(field)
        return values[:, -window]
    
    @expand_tdim
    def today_open(self, field: str, name: str = 'today_open') -> np.ndarray:
        assert ContextSrc.INTRADAY_QUOTATION in self.context.blocks
        block = self.context.blocks[ContextSrc.INTRADAY_QUOTATION]
        field_values = block.get_view(field)
        return field_values[:, 0]

    def history_rolling_moment_sequence(self, field: str, power: int, window: int, 
                                        input_value: T.Optional[np.ndarray] = None, 
                                        name: str = 'history_rolling_moment') -> np.ndarray:
       
        suffix_sum = self.history_pow_cumsum(field, input_value=input_value, power=power)
        N, T_daily = suffix_sum.shape
        out = np.full((N, T_daily), np.nan, dtype=np.float64)
        s_head = suffix_sum[:, :-window]
        s_tail = suffix_sum[:, window:]
        out[:, window-1 : -1] = (s_head - s_tail) / window
        out[:, -1] = suffix_sum[:, -window] / window
        return out