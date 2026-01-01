import inspect
import functools
import typing as T
from enum import Enum, auto
from dataclasses import dataclass, field

import numpy as np
from .base import BaseOps
from .utils import context_cache, expand_tdim_on_batch
from .. import kernels as kns
from ..context import ContextSrc


class DailyOps(BaseOps):

    @expand_tdim_on_batch
    def yesterday(self, field: str) -> np.ndarray:
        assert ContextSrc.DAILY_QUOTATION in self.context.blocks
        block = self.context.blocks[ContextSrc.DAILY_QUOTATION]
        return block.get_view(field)[:, -1]

    @context_cache
    def _history_prefix_sum(self, field: str, name: str = 'history_prefix_sum') -> np.ndarray:
        values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field)
        return np.nancumsum(values, axis=1)

    @context_cache
    def _history_weighted_prefix_sum(self, fields: T.Tuple[str, ...], name: str = 'history_weighted_prefix_sum') -> np.ndarray:
        value1 = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=fields[0]).astype(np.float64)
        value2 = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=fields[1]).astype(np.float64)
        return np.nancumsum(value1 * value2, axis=1)

    @context_cache
    def _history_prefix_count(self, field: str, name: str = 'history_prefix_count') -> np.ndarray:
        values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field)
        return np.nancumsum((~np.isnan(values)).astype(np.float32), axis=1)

    @context_cache
    def _history_suffix_high(self, field: str, name: str = 'history_suffix_high') -> np.ndarray:
        values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field) # N, T
        rev = values[:, ::-1]
        return np.fmax.accumulate(rev, axis=-1)[::-1]

    @context_cache
    def _history_suffix_low(self, field: str, name: str = 'history_suffix_low') -> np.ndarray:
        values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field) # N, T
        rev = values[:, ::-1]
        return np.fmin.accumulate(rev, axis=-1)[::-1]

    @context_cache
    def _history_recent_value_wo_nan(self, field: str, name: str = 'history_values_ffilled') -> np.ndarray:
        values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field) # N, T
        out = np.empty_like(values)
        kns.ffill2d(values, out)
        return out

    @expand_tdim_on_batch
    def history_ma(self, field: str, name: str = 'history_ma', window: int = 20) -> np.ndarray:
        prefix_sum = self._history_prefix_sum(field)
        prefix_count = self._history_prefix_count(field)
        ma = (prefix_sum[:, -1] - prefix_sum[:, -window]) / (prefix_count[:, -1] - prefix_count[:, -window])
        return ma
    
    @expand_tdim_on_batch
    def history_valid_ratio(self, field: str, name: str = 'history_valid_ratio', window: int = 20) -> np.ndarray:
        prefix_count = self._history_prefix_count(field)
        valid_ratio = (prefix_count[:, -1] - prefix_count[:, -window]) / window
        return valid_ratio
    
    @expand_tdim_on_batch
    def hisotry_vwap(self, fields: T.Tuple[str, ...], name: str = 'history_vwap', window: int = 20) -> np.ndarray:
        price_field, volume_field = fields
        weighted_prefix_sum = self._history_weighted_prefix_sum((price_field, volume_field))
        volume_prefix_sum = self._history_prefix_sum(volume_field)
        weighted_sum = weighted_prefix_sum[:, -1] - weighted_prefix_sum[:, -window]
        normalizer = volume_prefix_sum[:, -1] - volume_prefix_sum[:, -window]
        return (weighted_sum / normalizer).astype(np.float32)

    @expand_tdim_on_batch
    def history_high(self, field: str, name: str = 'history_high', window: int = 20) -> np.ndarray:
        suffix_high = self._history_suffix_high(field)
        return suffix_high[:, -window]
    
    @expand_tdim_on_batch
    def history_low(self, field: str, name: str = 'history_low', window: int = 20) -> np.ndarray:
        suffix_low = self._history_suffix_low(field)
        return suffix_low[:, -window]

    @expand_tdim_on_batch
    def history_lag(self, field: str, name: str = 'history_lag', window: int = 1) -> np.ndarray:
        values = self._history_recent_value_wo_nan(field)
        return values[:, -window]
    
    @expand_tdim_on_batch
    def today_open(self, field: str, name: str = 'today_open') -> np.ndarray:
        assert ContextSrc.INTRADAY_QUOTATION in self.context.blocks
        block = self.context.blocks[ContextSrc.INTRADAY_QUOTATION]
        field_values = block.get_view(field)
        return field_values[:, 0]

    @context_cache
    def _history_sq_cumsum(self, field: str, name: str = 'history_sq_cumsum') -> np.ndarray:
        hist_ret = self.context.get_field(ContextSrc.DAILY_QUOTATION, field=field)
        ret_sq = np.square(hist_ret)
        return np.nancumsum(ret_sq, axis=1)

    @expand_tdim_on_batch
    def history_sq_cumsum_with_count(self, field: str, window: int = 20) -> np.ndarray:
        hist_rsq_cumsum = self._history_sq_cumsum(field)
        hist_count_cumsum = self._history_prefix_count(field)
        hist_ss = hist_rsq_cumsum[:, -1] - hist_rsq_cumsum[:, -window]
        valid_count = hist_count_cumsum[:, -1] - hist_count_cumsum[:, -window]
        return hist_ss, valid_count
