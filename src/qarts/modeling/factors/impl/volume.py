import numpy as np

from ..ops import ContextOps
from ..context import ContextSrc
from ..base import register_factor, FactorFromDailyAndIntraday
from ..constants import FactorNames

__all__ = [
    'VolumeRatio',
]


@register_factor(FactorNames.VOLUME_RATIO)
class VolumeRatio(FactorFromDailyAndIntraday):
    num_daily_fields = 1
    num_intraday_fields = 1

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        super().__init__(input_fields=input_fields, window=window, **kwargs)
        self.history_volume_field = self.input_fields[ContextSrc.DAILY_QUOTATION][0]
        self.intraday_volume_field = self.input_fields[ContextSrc.INTRADAY_QUOTATION][0]

    @property
    def name(self) -> str:
        default_field = 'adjusted_volume'
        if self.history_volume_field != default_field:
            return f'{self.history_volume_field}_{FactorNames.VOLUME_RATIO}_{self.window}'
        return f'{FactorNames.VOLUME_RATIO}_{self.window}'

    def compute_from_context(self, ops: ContextOps, out: np.ndarray):
        history_volume = ops.history_window_ma(self.history_volume_field, window=self.window)
        current_volume = ops.today_cumsum(self.intraday_volume_field)
        current_time_fraction = ops.time_fraction()
        today_volume_estimated = (current_volume + 1) / current_time_fraction
        np.log(today_volume_estimated, out=out)
        out -= np.log(history_volume)

