import abc
import datetime
import typing as T
from dataclasses import dataclass, field

from qarts.core import PanelBlockDense, PanelBlockIndexed
from qarts.loader import PanelLoader, DailyDataManager, VariableLoadSpec
from qarts.modeling.factors import FactorContext, ContextSrc, get_fill_method
from .base import Processor, GlobalContext


class DailyAndIntradayProvider(Processor):
    name: str = 'daily_and_intraday'
    provide_keys: T.List[str] = ['daily', 'intraday']

    def __init__(
        self, 
        loader: PanelLoader, 
        daily_fields: list[str], 
        intraday_fields: list[str],
        target_specs: list[VariableLoadSpec] = [],
        return_factor_context: bool = False
    ):
        self.loader = loader
        self.daily_fields = daily_fields + ['instrument']
        if 'factor' not in self.daily_fields:
            self.daily_fields.append('factor')
        self.intraday_fields = intraday_fields + ['instrument']
        self.target_specs = target_specs
        self.return_factor_context = return_factor_context
        self.daily_manager = DailyDataManager(loader, daily_fields, recent_days=-1) # load all daily data

    def generate_tasks(self) -> T.Generator[tuple[datetime.datetime, T.Any], None, None]:
        load_spec = VariableLoadSpec(var_type='quotation', load_kwargs={})
        available_dates = self.loader.list_available_dates([load_spec])
        existing_dates = self.loader.list_available_dates(self.target_specs)
        required_dates = sorted(set(available_dates) - set(existing_dates))

        for date in required_dates:
            yield datetime.datetime.combine(date, datetime.time(0, 0, 0)), None

    def create_factor_context(self, daily_block: PanelBlockIndexed, intraday_block: PanelBlockIndexed) -> FactorContext:
        daily_block = PanelBlockDense.from_indexed_block(
            daily_block,
            required_columns=list(daily_block.data.columns),
            fill_methods=[get_fill_method(c) for c in daily_block.data.columns],
            frequency='1D'
        )
        context = FactorContext.from_daily_block(daily_block)
        intraday_block = PanelBlockDense.from_indexed_block(
            intraday_block,
            required_columns=list(intraday_block.data.columns),
            fill_methods=[get_fill_method(c) for c in intraday_block.data.columns],
            frequency='1min',
            inst_cats=daily_block.instruments,
            is_intraday=True,
            max_nan_count=100,
            backward_fill=True
        )
        context.register_block(ContextSrc.INTRADAY_QUOTATION, intraday_block)
        return context

    def process(self, context: GlobalContext) -> T.Any:
        date = context.current_datetime
        daily_block = self.daily_manager.load_daily_block_before(date, num_period=365, include=False)
        intraday_block = self.loader.load_intraday_quotation(date, instruments=daily_block.instruments, fields=self.intraday_fields)
        result = {
            'daily': daily_block, 
            'intraday': intraday_block,
        }
        if self.return_factor_context:
            factor_context = self.create_factor_context(daily_block, intraday_block)
            result['factor_context'] = factor_context
        return result