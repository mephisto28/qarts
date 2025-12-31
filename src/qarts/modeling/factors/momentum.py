import numpy as np

from qarts.modeling.factors.context import ContextOps, ContextSrc
from qarts.modeling.factors.base import register_factor, FactorFromDailyAndIntraday


@register_factor('mom')
class Momentum(FactorFromDailyAndIntraday):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return f'mom_{self.window}'

    def _check_inputs_valid(self):
        assert len(self.input_fields[ContextSrc.DAILY_QUOTATION]) == 1
        assert len(self.input_fields[ContextSrc.INTRADAY_QUOTATION]) == 1

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        history_values = ops.history_lag(self.history_price_field, window=self.window)
        current_price = ops.now(self.price_field)
        out[:] = np.log(current_price) - np.log(history_values)


@register_factor('intraday_mom')
class IntradayMomentum(FactorFromDailyAndIntraday):
    num_daily_fields = 0
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return f'intraday_mom_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        open_price = ops.today_open(self.price_field)
        current_price = ops.now(self.price_field)
        out[:] = np.log(current_price) - np.log(open_price)