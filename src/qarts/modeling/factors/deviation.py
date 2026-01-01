from dataclasses import dataclass
import re
import typing as T

import numpy as np

from qarts.modeling.factors.ops import ContextOps
from qarts.modeling.factors.context import ContextSrc
from qarts.modeling.factors.base import register_factor, Factor, FactorFromDailyAndIntraday


@register_factor('price_dev_from_ma')
class MAPriceDeviation(Factor):
    num_daily_fields = 1
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return f'price_dev_from_ma_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        ma = ops.history_ma(self.history_price_field, window=self.window)
        current_price = ops.now(self.price_field)
        np.log(current_price, out=out)
        out -= np.log(ma)


@register_factor('price_dev_from_vwap')
class VWAPPriceDeviation(Factor):
    num_daily_fields = 2
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field, self.history_volume_field = self.input_fields[ContextSrc.DAILY_QUOTATION]
        self.current_price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return f'price_dev_from_vwap_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        vwap = ops.hisotry_vwap((self.history_price_field, self.history_volume_field), self.window)
        current_price = ops.now(self.current_price_field)
        np.log(current_price, out=out)
        out -= np.log(vwap)


@register_factor('price_dev_from_yest_vwap')
class YestVWAPPriceDeviation(Factor):
    num_daily_fields = 1
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.yesterday_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.current_price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return 'price_dev_from_yest_vwap'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        vwap = ops.yesterday(self.yesterday_price_field)
        current_price = ops.now(self.current_price_field)
        np.log(current_price, out=out)
        out -= np.log(vwap)


@register_factor('price_position')
class PricePosition(FactorFromDailyAndIntraday):
    num_daily_fields = 2
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]
        self.high_field, self.low_field = self.input_fields[ContextSrc.DAILY_QUOTATION]

    @property
    def name(self) -> str:
        return f'price_position_{self.window}'
    
    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        price = ops.now(self.price_field)
        history_high = ops.history_high(self.high_field, window=self.window)
        history_low = ops.history_low(self.low_field, window=self.window)
        out[:] = price - history_low
        out /= (history_high - history_low)
