import typing as T

import numpy as np
from .base import BaseOps
from .utils import context_timestep_cache


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
