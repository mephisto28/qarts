import numpy as np

from qarts.modeling.factors.ops import ContextOps
from qarts.modeling.factors.context import ContextSrc
from qarts.modeling.factors.base import register_factor, FactorFromDailyAndIntraday


@register_factor('daily_volatility')
class DailyVolatility(FactorFromDailyAndIntraday):
    num_daily_fields = 1 # daily_return
    num_intraday_fields = 0
    num_factor_cache_fields = 1 # daily_mom_1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 5, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_return_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.return_field = self.input_fields[ContextSrc.FACTOR_CACHE][0]

    @property
    def name(self) -> str:
        return f'daily_volatility_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        today_ret = ops.now_factor(self.return_field)
        hist_ss, valid_count = ops.history_sq_cumsum_with_count(self.history_return_field, self.window)
        current_ss = np.square(today_ret)
        out[:] = np.sqrt((hist_ss + current_ss) / (valid_count + 1))