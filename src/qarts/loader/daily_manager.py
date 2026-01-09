import datetime
import typing as T

from qarts.core.panel import DailyPanelBlockDense, DailyPanelBlockIndexed
from .dataloader import PanelLoader


class DailyDataManager:

    def __init__(self, loader: PanelLoader, fields: list[str], recent_days: int = -1) -> None:
        self.loader = loader
        self.fields = fields
        self.daily_block, self.daily_fields_require_adjustment = self._load_daily_block(recent_days=recent_days)

    def _load_daily_block(self, recent_days: int = -1) -> tuple[DailyPanelBlockIndexed, list[str]]:
        today = datetime.datetime.now()
        if recent_days < 0:
            start_date = None
            end_date = None
        else:
            start_date = today - datetime.timedelta(days=recent_days)
            end_date = today
        required_daily_fields_before_adjustment = list(set([field.replace('adjusted_', '') for field in self.fields]))
        daily_block = self.loader.load_daily_quotation(
            start_date=start_date, end_date=end_date,
            fields=required_daily_fields_before_adjustment + ['instrument', 'factor']
        )
        daily_fields_require_adjustment = [field.replace('adjusted_', '') for field in self.fields if field.startswith('adjusted_')]
        daily_block.ensure_order('datetime-first')
        return daily_block, daily_fields_require_adjustment

    def load_daily_block_before(self, date, num_period: int = 365, include: bool = False):
        start_date = date - datetime.timedelta(days=num_period)
        daily_block = self.daily_block.between(start_date=start_date, end_date=date)
        daily_block.adjust_field_by_last(fields=self.daily_fields_require_adjustment)
        if not include:
            end_date = date - datetime.timedelta(days=1)
            daily_block = daily_block.between(start_date=start_date, end_date=end_date)
        daily_block.ensure_order('instrument-first')
        return daily_block
