from dataclasses import dataclass
import re
import typing as T

import numpy as np

from qarts.modeling.factors.context import ContextOps, ContextSrc
from qarts.modeling.factors.base import register_factor, Factor


@register_factor('price_dev_from_ma')
class MAPriceDeviation(Factor):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return f'price_dev_from_ma_{self.window}'

    def _check_inputs_valid(self):
        assert len(self.input_fields[ContextSrc.DAILY_QUOTATION]) == 1
        assert len(self.input_fields[ContextSrc.INTRADAY_QUOTATION]) == 1

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        ma = ops.history_ma(self.history_price_field, window=self.window)
        current_price = ops.now(self.price_field)
        out[:] = np.log(current_price) - np.log(ma)


@register_factor('price_dev_from_vwap')
class VWAPPriceDeviation(Factor):

    def __init__(self, input_fields: dict[str, list[str]], window: int, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_price_field, self.history_volume_field = self.input_fields['daily_quotation']
        self.current_price_field = self.input_fields['intraday_quotation'][0]

    @property
    def name(self) -> str:
        return f'price_dev_from_vwap_{self.window}'

    def _check_inputs_valid(self):
        assert len(self.input_fields[ContextSrc.DAILY_QUOTATION]) == 2
        assert len(self.input_fields[ContextSrc.INTRADAY_QUOTATION]) == 1

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        vwap = ops.hisotry_vwap((self.history_price_field, self.history_volume_field), self.window)
        current_price = ops.now(self.current_price_field)
        out[:] = np.log(current_price) - np.log(vwap)


@register_factor('price_dev_from_yest_vwap')
class YestVWAPPriceDeviation(Factor):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.yesterday_price_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.current_price_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        return 'price_dev_from_yest_vwap'

    def _check_inputs_valid(self):
        assert len(self.input_fields[ContextSrc.DAILY_QUOTATION]) == 1, f"Invalid input fields: {self.input_fields}"
        assert len(self.input_fields[ContextSrc.INTRADAY_QUOTATION]) == 1, f"Invalid input fields: {self.input_fields}"

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        vwap = ops.yesterday(self.yesterday_price_field)
        current_price = ops.now(self.current_price_field)
        out[:] = np.log(current_price) - np.log(vwap)

