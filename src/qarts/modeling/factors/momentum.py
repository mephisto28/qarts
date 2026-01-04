import numpy as np

from .ops import ContextOps
from .context import ContextSrc
from .base import register_factor, FactorFromDailyAndIntraday
from .constants import FactorNames

__all__ = [
    'DailyMomentum',
    'DailyMomentumSum',
    'IntradayMomentum'
]


@register_factor(FactorNames.DAILY_MOM)
class DailyMomentum(FactorFromDailyAndIntraday):
    num_daily_fields = 1
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        default_field = 'adjusted_close'
        if self.history_price_field != default_field:
            return f'{self.history_price_field}_{FactorNames.DAILY_MOM}_{self.window}'
        return f'{FactorNames.DAILY_MOM}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        history_values = ops.history_lag(self.history_price_field, window=self.window)
        current_price = ops.now(self.price_field)
        np.log(current_price, out=out)
        out -= np.log(history_values)
        out /= np.sqrt(self.window)


@register_factor(FactorNames.DAILY_MOM_SUM)
class DailyMomentumSum(FactorFromDailyAndIntraday):
    num_daily_fields = 1
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_return_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.return_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return f'{FactorNames.DAILY_MOM_SUM}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        cumsum = ops.history_pow_cumsum(self.history_return_field, power=1)
        sum_before_today = cumsum[:, -self.window+1]
        out[:] = ops.now(self.return_field)
        out += sum_before_today[:, None]
        out /= np.sqrt(self.window)


@register_factor(FactorNames.INTRADAY_MOM)
class IntradayMomentum(FactorFromDailyAndIntraday):
    num_daily_fields = 0
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return f'{FactorNames.INTRADAY_MOM}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        open_price = ops.today_open(self.price_field)
        current_price = ops.now(self.price_field)
        np.log(current_price, out=out)
        out -= np.log(open_price)
        out /= np.sqrt(self.window)