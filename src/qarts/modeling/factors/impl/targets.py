import numpy as np
import numba as nb

from ..ops import ContextOps
from ..context import ContextSrc
from ..base import register_factor, FactorFromDailyAndIntraday
from ..constants import FactorNames
from .. import kernels as kns

__all__ = [
    'FutureDayTargets',
    'TodayTargets',
]


@register_factor(FactorNames.FUTURE_DAY_TARGETS)
class FutureDayTargets(FactorFromDailyAndIntraday):
    num_intraday_fields = 2

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.future_price_field = self.input_fields[ContextSrc.FUTURE_DAILY_QUOTATION][0]
        self.buy_price_field, self.mid_price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION]

    @property
    def name(self) -> str:
        default_field = 'adjusted_close'
        if self.future_price_field != default_field:
            return f'{self.future_price_field}_{FactorNames.FUTURE_DAY_TARGETS}_{self.window}'
        return f'{FactorNames.FUTURE_DAY_TARGETS}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        future_values = ops.future_n_days(self.future_price_field, window=self.window)
        buy_price = ops.now(self.buy_price_field)
        mid_price = ops.now(self.mid_price_field)
        cur_price = np.where(buy_price == 0, mid_price, buy_price)
        out[:] = np.log(future_values) - np.log(cur_price)
        # out /= np.sqrt(self.window + 1)


@register_factor(FactorNames.INTER_DAY_TARGETS)
class InterDayTargets(FactorFromDailyAndIntraday):
    num_intraday_fields = 2

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.future_price_field = self.input_fields[ContextSrc.FUTURE_DAILY_QUOTATION][0]
        self.buy_price_field, self.mid_price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION]

    @property
    def name(self) -> str:
        default_field = 'adjusted_close'
        if self.future_price_field != default_field:
            return f'{self.future_price_field}_{FactorNames.INTER_DAY_TARGETS}_{self.window}'
        return f'{FactorNames.INTER_DAY_TARGETS}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        future_values = ops.future_n_days(self.future_price_field, window=self.window)
        buy_price = ops.now(self.buy_price_field)[:, -1:]
        mid_price = ops.now(self.mid_price_field)[:, -1:]
        cur_price = np.where(buy_price == 0, mid_price, buy_price)
        out[:] = np.log(future_values) - np.log(cur_price)


@register_factor(FactorNames.FUTURE_DAY_RANGE_TARGETS)
class FutureDayRangeTargets(FactorFromDailyAndIntraday):
    num_intraday_fields = 2

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.future_high_field, self.future_low_field = self.input_fields[ContextSrc.FUTURE_DAILY_QUOTATION]
        self.buy_price_field, self.mid_price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION]

    @property
    def name(self) -> str:
        default_field = 'adjusted_high'
        if self.future_high_field != default_field:
            return f'{self.future_high_field}_{self.future_low_field}_{FactorNames.FUTURE_DAY_RANGE_TARGETS}_{self.window}'
        return f'{FactorNames.FUTURE_DAY_RANGE_TARGETS}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        future_high = ops.future_highest(self.future_high_field, window=self.window)
        future_low = ops.future_lowest(self.future_low_field, window=self.window)
        out[:] = np.log(future_high) - np.log(future_low)


@register_factor(FactorNames.FUTURE_DAY_UP_RANGE_TARGETS)
class FutureDayUpRangeTargets(FactorFromDailyAndIntraday):
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.future_high_field = self.input_fields[ContextSrc.FUTURE_DAILY_QUOTATION][0]
        self.mid_price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        default_field = 'adjusted_high'
        if self.future_high_field != default_field:
            return f'{self.future_high_field}_{FactorNames.FUTURE_DAY_UP_RANGE_TARGETS}_{self.window}'
        return f'{FactorNames.FUTURE_DAY_UP_RANGE_TARGETS}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        future_high = ops.future_highest(self.future_high_field, window=self.window)
        now_price = ops.now(self.mid_price_field)
        out[:] = np.log(future_high) - np.log(now_price)


@register_factor(FactorNames.FUTURE_DAY_DOWN_RANGE_TARGETS)
class FutureDayDownRangeTargets(FactorFromDailyAndIntraday):
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.future_low_field = self.input_fields[ContextSrc.FUTURE_DAILY_QUOTATION][0]
        self.mid_price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        default_field = 'adjusted_low'
        if self.future_low_field != default_field:
            return f'{self.future_low_field}_{FactorNames.FUTURE_DAY_DOWN_RANGE_TARGETS}_{self.window}'
        return f'{FactorNames.FUTURE_DAY_DOWN_RANGE_TARGETS}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        future_low = ops.future_lowest(self.future_low_field, window=self.window)
        now_price = ops.now(self.mid_price_field)
        out[:] = np.log(future_low) - np.log(now_price)


@register_factor(FactorNames.TODAY_TARGETS)
class TodayTargets(FactorFromDailyAndIntraday):
    num_intraday_fields = 3

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.buy_field, self.sell_field, self.mid_price = self.input_fields[ContextSrc.INTRADAY_QUOTATION]

    @property
    def name(self) -> str:
        return f'{FactorNames.TODAY_TARGETS}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        buy_price = ops.now(self.buy_field)
        sell_price = ops.now(self.sell_field)
        mid_price = ops.now(self.mid_price)
        compute_targets(buy_price, sell_price, mid_price, out, self.window)
        # out[:] *= 15 # sqrt(240)
        # out[:] /= np.sqrt(self.window)


@register_factor(FactorNames.RANK_TARGETS)
class RankTargets(FactorFromDailyAndIntraday):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.target_field = self.input_fields[ContextSrc.FACTOR_CACHE][0]
    
    @property
    def name(self) -> str:
        return f'rank_{self.target_field}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        target = ops.now(self.target_field) # N, T
        kns.fast_binned_percentile_2d(target.T, n_bins=2000, sigma_clip=3.5, out=out.T)


@nb.jit
def compute_targets(
    buy_price: np.ndarray,
    sell_price: np.ndarray,
    mid_price: np.ndarray,
    out: np.ndarray,
    future_window: int
):
    N, T = buy_price.shape
    for i in nb.prange(N):
        for t in range(T):
            t_future = min(t + future_window, T - 1)
            mid_price_now = mid_price[i, t]
            buy_price_now = buy_price[i, t]
            if buy_price_now == 0:
                buy_price_now = mid_price_now
            mid_price_future = mid_price[i, t_future]
            sell_price_future = sell_price[i, t_future]
            if sell_price_future == 0:
                sell_price_future = mid_price_future
            out[i, t] = np.log(sell_price_future) - np.log(buy_price_now)