import numpy as np

from ..ops import ContextOps
from ..context import ContextSrc
from ..base import register_factor, FactorFromDailyAndIntraday
from ..constants import FactorNames

__all__ = [
    'AbsTransform',
]


@register_factor(FactorNames.ABS_TRANSFORM)
class AbsTransform(FactorFromDailyAndIntraday):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.factor_field = self.input_fields[ContextSrc.FACTOR_CACHE][0]

    @property
    def name(self) -> str:
        return f'{FactorNames.ABS_TRANSFORM}_{self.factor_field}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        current_price = ops.now(self.factor_field)
        np.abs(current_price, out=out)
