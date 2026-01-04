import typing as T

import numpy as np

from .ops import ContextOps
from .context import ContextSrc
from .constants import FactorNames
from .base import register_factor, Factor, FactorFromDailyAndIntraday


@register_factor(FactorNames.PRICE_DEV_FROM_MA)
class MAPriceDeviation(Factor):
    num_daily_fields = 1
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return f'{FactorNames.PRICE_DEV_FROM_MA}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        ma = ops.history_window_ma(self.history_price_field, window=self.window)
        current_price = ops.now(self.price_field)
        np.log(current_price, out=out)
        out -= np.log(ma)
        out /= np.sqrt(self.window)


@register_factor(FactorNames.PRICE_DEV_FROM_VWAP)
class VWAPPriceDeviation(Factor):
    num_daily_fields = 2
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field, self.history_volume_field = self.input_fields[ContextSrc.DAILY_QUOTATION]
        self.current_price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return f'{FactorNames.PRICE_DEV_FROM_VWAP}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        vwap = ops.hisotry_window_vwap((self.history_price_field, self.history_volume_field), window=self.window)
        current_price = ops.now(self.current_price_field)
        np.log(current_price, out=out)
        out -= np.log(vwap)
        out /= np.sqrt(self.window)


@register_factor(FactorNames.PRICE_DEV_FROM_YEST_VWAP)
class YestVWAPPriceDeviation(Factor):
    num_daily_fields = 1
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.yesterday_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.current_price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return FactorNames.PRICE_DEV_FROM_YEST_VWAP

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        vwap = ops.yesterday(self.yesterday_price_field)
        current_price = ops.now(self.current_price_field)
        np.log(current_price, out=out)
        out -= np.log(vwap)


@register_factor(FactorNames.PRICE_POSITION)
class PricePosition(FactorFromDailyAndIntraday):
    num_daily_fields = 2
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]
        self.high_field, self.low_field = self.input_fields[ContextSrc.DAILY_QUOTATION]

    @property
    def name(self) -> str:
        return f'{FactorNames.PRICE_POSITION}_{self.window}'
    
    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        price = ops.now(self.price_field)
        history_high = ops.history_window_high(self.high_field, window=self.window)
        history_low = ops.history_window_low(self.low_field, window=self.window)
        high = np.maximum(history_high, price)
        low = np.minimum(history_low, price)
        out[:] = price - low
        out /= high - low
