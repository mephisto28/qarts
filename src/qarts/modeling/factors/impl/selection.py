import numpy as np
import numba as nb

from ..ops import ContextOps
from ..context import ContextSrc
from ..base import register_factor, FactorFromDailyAndIntraday
from ..constants import FactorNames

__all__ = [
    'IsUpLimit',
    'IsDownLimit',
    'DailyRecentVacancy'
]


@register_factor(FactorNames.DAILY_RECENT_VACANCY)
class DailyRecentVacancy(FactorFromDailyAndIntraday):
    num_daily_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]

    @property
    def name(self) -> str:
        default_field = 'adjusted_close'
        if self.history_price_field != default_field:
            return f'{self.history_price_field}_{FactorNames.DAILY_RECENT_VACANCY}_{self.window}'
        return f'{FactorNames.DAILY_RECENT_VACANCY}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        history_values = ops.history_pow_cumsum(self.history_price_field, power=0)
        valid_count = history_values[:, -self.window]
        out[:] = valid_count[:, None] / self.window
        # breakpoint()


@register_factor(FactorNames.IS_UP_LIMIT)
class IsUpLimit(FactorFromDailyAndIntraday):
    num_daily_fields = 1
    num_intraday_fields = 3

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.bid_p1_field, self.ask_v1_field , self.bid_v1_field = \
            self.input_fields[ContextSrc.INTRADAY_QUOTATION]

    @property
    def name(self) -> str:
        return f'{FactorNames.IS_UP_LIMIT}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        bid1_price = ops.now(self.bid_p1_field)
        ask1_volume = ops.now(self.ask_v1_field)
        bid1_volume = ops.now(self.bid_v1_field)
        yest_price = ops.yesterday(self.history_price_field)
        is_up_limit(bid1_price, ask1_volume, bid1_volume, yest_price, out)


@register_factor(FactorNames.IS_DOWN_LIMIT)
class IsDownLimit(FactorFromDailyAndIntraday):
    num_daily_fields = 1
    num_intraday_fields = 3

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.ask_p1_field, self.ask_v1_field , self.bid_v1_field = \
            self.input_fields[ContextSrc.INTRADAY_QUOTATION]

    @property
    def name(self) -> str:
        return f'{FactorNames.IS_UP_LIMIT}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        ask1_price = ops.now(self.ask_p1_field)
        ask1_volume = ops.now(self.ask_v1_field)
        bid1_volume = ops.now(self.bid_v1_field)
        yest_price = ops.yesterday(self.history_price_field)
        is_down_limit(ask1_price, ask1_volume, bid1_volume, yest_price, out)


@nb.njit
def is_up_limit(
    bid1_price: np.ndarray, 
    ask1_volume: np.ndarray,
    bid1_volume: np.ndarray,
    yest_price: np.ndarray,
    out: np.ndarray
):
    N, T = bid1_price.shape
    for i in nb.prange(N):
        for t in range(T):
            today_return = np.round(bid1_price[i, t] / yest_price[i, t] - 1, 2)
            if bid1_volume[i, t] != 0 and ask1_volume[i, t] == 0 and (today_return == 0.05 or today_return == 0.1 or today_return == 0.2):
                out[i, t] = 1
            else:
                out[i, t] = 0

@nb.njit
def is_down_limit(
    ask1_price: np.ndarray,
    ask1_volume: np.ndarray,
    bid1_volume: np.ndarray,
    yest_price: np.ndarray,
    out: np.ndarray
):
    N, T = ask1_price.shape
    for i in nb.prange(N):
        for t in range(T):
            today_return = np.round(ask1_price[i, t] / yest_price[i, t] - 1, 2)
            if ask1_volume[i, t] != 0 and bid1_volume[i, t] == 0 and (today_return == -0.05 or today_return == -0.1 or today_return == -0.2):
                out[i, t] = 1
            else:
                out[i, t] = 0
