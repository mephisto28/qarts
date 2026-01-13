import typing as T

import numpy as np
from .base import BaseOps
from .utils import context_timestep_cache
from ..context import ContextSrc


class IntradayOps(BaseOps):

    @context_timestep_cache
    def today_diff(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'today_diff') -> np.ndarray:
        if input_value is None:
            data = self.now(field) # N, T
        else:
            data = input_value
        data_diff = data.copy()
        data_diff[:, 1:] -= data[:, :-1]
        return data_diff

    @context_timestep_cache
    def today_cumsum(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'cumsum') -> np.ndarray:
        if input_value is None:
            data = self.now(field)
        else:
            data = input_value
        return np.nancumsum(data, axis=1)

    @context_timestep_cache
    def today_high(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'today_high') -> np.ndarray:
        if input_value is None:
            data = self.now(field)
        else:
            data = input_value
        return np.maximum.accumulate(data, axis=1)

    @context_timestep_cache
    def today_low(self, field: str, input_value: T.Optional[np.ndarray] = None, name: str = 'today_low') -> np.ndarray:
        if input_value is None:
            data = self.now(field)
        else:
            data = input_value
        return np.minimum.accumulate(data, axis=1)

    def time_fraction(self) -> np.ndarray:
        if getattr(self, '_time_fraction', None) is None:
            T = len(self.context.blocks[ContextSrc.INTRADAY_QUOTATION].timestamps)
            self._time_fraction = (np.arange(T) + 1) / T
        return self._time_fraction