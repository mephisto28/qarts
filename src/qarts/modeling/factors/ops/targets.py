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


class TargetsOps(BaseOps):

    def future_n_days(self, field: str, n: int) -> np.ndarray:
        assert ContextSrc.FUTURE_DAILY_QUOTATION in self.context.blocks
        block = self.context.blocks[ContextSrc.FUTURE_DAILY_QUOTATION]
        return block.get_view(field)[:, n]

    def today(self, field: str) -> np.ndarray:
        return self.future_n_days(field, n=0)

    @context_static_cache
    def _future_prefix_high(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'future_prefix_high') -> np.ndarray:
        data = self.context.get_field(src=ContextSrc.FUTURE_DAILY_QUOTATION, field=field)
        out = data[1:].copy()
        np.fmax.accumulate(data[1:], axis=1, out=out)
        return out
    
    @context_static_cache
    def _future_prefix_low(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'future_prefix_low') -> np.ndarray:
        data = self.context.get_field(src=ContextSrc.FUTURE_DAILY_QUOTATION, field=field)
        out = data[1:].copy()
        np.fmin.accumulate(data[1:], axis=1, out=out)
        return out
    
    @expand_tdim
    def future_sum(self, field: str, window: int = 1) -> np.ndarray:
        cumsum = self._future_cumsum(field)
        return cumsum[:, window] - cumsum[:, 0]

    @expand_tdim
    def future_price_over_today_close(self, field: str, window: int = 1) -> np.ndarray:
        future_price = self.future_n_days(field, window)
        today_close = self.today(field)
        return future_price / today_close
      
    @expand_tdim
    def future_highest_over_today_close(self, field: str, window: int = 1) -> np.ndarray:
        high = self._future_prefix_high(field)[window - 1]
        today_close = self.today(field)
        return high / today_close

    def intraday_targets(self, field: str, window: int = 10) -> np.ndarray:
        price = self.now(field) # (N, T)
        out = np.empty((price, ), dtype=np.float32)
        if window == -1:
            out[:] = np.log(price[:, -1:]) - np.log(price)
        elif window > 0:
            out[:, :-window] = np.log(price[:, window:]) - np.log(price[:, :-window])
            out[:, -window:] = np.log(price[:, -1:]) - np.log(price[:, -window:])
        return out