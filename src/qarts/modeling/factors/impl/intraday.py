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


@register_factor(FactorNames.TODAY_VOLATILITY)
class TodayVolatility(FactorFromDailyAndIntraday):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.today_momentum_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0] if ContextSrc.INTRADAY_QUOTATION in self.input_fields \
            else self.input_fields[ContextSrc.FACTOR_CACHE][0]

    @property
    def name(self) -> str:
        return f'{FactorNames.INTRADAY_MOM}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        diff_values = ops.today_diff(self.today_momentum_field)
        cum_realized_vol(diff_values, out)


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

