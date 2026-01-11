import datetime
import typing as T
from collections import defaultdict

import numpy as np
from loguru import logger

from qarts.core import FactorPanelBlockDense, IntradayPanelBlockDense, DailyPanelBlockIndexed
from qarts.core.panel import PanelBlockDense
from qarts.utils.profiler import TimerProfiler
from qarts.loader.dataloader import PanelLoader, VariableLoadSpec
from .base import Factor, FactorSpec, get_factor
from .context import ContextSrc, FactorContext
from .ops import ContextOps


class PipelineFactory:

    def __init__(self, factor_specs: list[FactorSpec]):
        self.factors = [get_factor(spec) for spec in factor_specs]
        input_fields = defaultdict(set)
        for factor in self.factors:
            for src in factor.input_fields:
                input_fields[src].update(factor.input_fields[src])
        self.input_fields = {src: list(fields) for src, fields in input_fields.items()}

    def create_batch_pipeline(self, src: ContextSrc) -> T.Callable:
        return FactorsProcessor(
            factors=self.factors, 
            input_fields=self.input_fields,
            src=src, is_online=False,
        )

    def create_online_engine(self, src: ContextSrc) -> T.Callable:
        return FactorsProcessor(
            factors=self.factors,
            input_fields=self.input_fields,
            src=src, is_online=True
        )


class FactorsProcessor:
    def __init__(
        self, 
        factors: list[Factor], 
        input_fields: dict[str, list[str]],
        src: ContextSrc = ContextSrc.INTRADAY_QUOTATION,
        is_online: bool = False
    ):
        self.factors = factors
        self.input_fields = input_fields
        self.src = src
        self.is_online = is_online
        self.has_future_data = ContextSrc.FUTURE_DAILY_QUOTATION in input_fields

    def get_daily_fields_before_adjustment(self) -> list[str]:
        required_daily_fields = self.get_daily_fields()
        required_daily_fields_before_adjustment = list(set([field.replace('adjusted_', '') for field in required_daily_fields]))
        return required_daily_fields_before_adjustment

    def get_daily_fields(self) -> list[str]:
        return list(set(self.input_fields.get(ContextSrc.DAILY_QUOTATION, []) + self.input_fields.get(ContextSrc.FUTURE_DAILY_QUOTATION, [])))

    def get_intraday_fields(self) -> list[str]:
        return self.input_fields[ContextSrc.INTRADAY_QUOTATION]

    def process_batch(self, context: FactorContext) -> FactorPanelBlockDense:
        factors_block = FactorPanelBlockDense.empty_like(context.blocks[self.src], F=len(self.factors), fields=[f.name for f in self.factors])
        context.register_block(ContextSrc.FACTOR_CACHE, factors_block)

        context_ops = ContextOps(context, is_online=False)
        for i, factor in enumerate(self.factors):
            placeholder = factors_block.data[i]
            factor.compute_from_context(context_ops, placeholder)
        for i, factor in enumerate(self.factors):
            value = factors_block.data[i]
            value -= factor.shift
            value *= factor.scale
        return factors_block

    def process_online(self, context: FactorContext) -> FactorPanelBlockDense:
        # TODO
        raise NotImplementedError

    def __call__(self, context: FactorContext) -> FactorPanelBlockDense:
        if self.is_online:
            return self.process_online(context)
        else:
            return self.process_batch(context)



class IntradayBatchProcessingEngine:

    def __init__(self, loader: PanelLoader, factor_specs: list[FactorSpec]):        
        self.loader = loader
        self.factor_factory = PipelineFactory(factor_specs)
        self.factors = self.factor_factory.factors
        self.daily_block, self.daily_fields_require_adjustment = self._load_daily_block()
        self.intraday_fields = self.factor_factory.input_fields[ContextSrc.INTRADAY_QUOTATION]
        self.profiler = TimerProfiler()
    
    def _load_daily_block(self) -> tuple[DailyPanelBlockIndexed, list[str]]:
        required_daily_fields = self.factor_factory.input_fields[ContextSrc.DAILY_QUOTATION]
        required_daily_fields_before_adjustment = list(set([field.replace('adjusted_', '') for field in required_daily_fields]))
        logger.info("Loading daily quotation...")
        daily_block = self.loader.load_daily_quotation(fields=required_daily_fields_before_adjustment + ['instrument', 'factor'])

        logger.info('preprocessing daily quotation...')
        daily_fields_require_adjustment = [field.replace('adjusted_', '') for field in required_daily_fields if field.startswith('adjusted_')]
        daily_block.ensure_order('datetime-first')
        return daily_block, daily_fields_require_adjustment

    def generate_date_tasks(self) -> T.Generator[datetime.datetime, None, None]:
        load_spec = VariableLoadSpec(var_type='quotation', load_kwargs={})
        save_specs = [VariableLoadSpec(var_type='factor', load_kwargs={'factor': f.name}) for f in self.factors]
        available_dates = self.loader.list_available_dates([load_spec])
        existing_dates = self.loader.list_available_dates(save_specs)
        required_dates = sorted(set(available_dates) - set(existing_dates))

        for date in required_dates:
            yield datetime.datetime.combine(date, datetime.time(0, 0, 0))

    def process_targets(self, date: datetime.datetime) -> FactorPanelBlockDense:
        logger.info(f"Processing targets for date: {date}")
        start_date = date
        end_date = date + datetime.timedelta(days=21)
        daily_block = self.daily_block.between(start_date=start_date, end_date=end_date)
        daily_block.adjust_field_by_last(fields=self.daily_fields_require_adjustment)
        daily_block.ensure_order('instrument-first')
        context = FactorContext.from_daily_block(daily_block)
        context
    
    def process_factor(self, date: datetime.datetime) -> FactorPanelBlockDense:
        logger.info(f"Processing date: {date}")
        with self.profiler.section('daily_history'):
            start_date = date - datetime.timedelta(days=300) # 200
            yesterday = date - datetime.timedelta(days=1)
            daily_block = self.daily_block.between(start_date=start_date, end_date=date)
            daily_block.adjust_field_by_last(fields=self.daily_fields_require_adjustment)
            daily_block = daily_block.filter_instrument_by_count(min_count=180)
            daily_block = daily_block.between(start_date=start_date, end_date=yesterday)
            daily_block.ensure_order('instrument-first')
            context = FactorContext.from_daily_block(daily_block)
        
        with self.profiler.section('daily_future'):
            end_date = date + datetime.timedelta(days=21)
            daily_block_future: DailyPanelBlockIndexed = self.daily_block.between(start_date=date, end_date=end_date)
            daily_block_future.adjust_field_by_first(fields=self.daily_fields_require_adjustment)
            daily_block_future.ensure_order('instrument-first')
            columns = list(daily_block.data.columns)
            daily_block_future = PanelBlockDense.from_indexed_block(
                daily_block_future,
                required_columns=columns,
                fill_methods=[1 for _ in columns],
                frequency='1D',
                inst_cats=context.inst_categories
            )
            context.register_block(ContextSrc.FUTURE_DAILY_QUOTATION, daily_block_future)

        with self.profiler.section('intraday_load'):
            load_spec = VariableLoadSpec(var_type='quotation', load_kwargs={'date': date, 'fields': self.intraday_fields + ['instrument']})
            block = self.loader.load_intraday(load_spec)
            intraday_block = IntradayPanelBlockDense.from_indexed_block(
                block, 
                required_columns=self.intraday_fields, 
                fill_methods=[1 for _ in self.intraday_fields],
                frequency='1min',
                inst_cats=context.inst_categories,
                is_intraday=True,
                max_nan_count=100,
                backward_fill=True
            )
            context.register_block(ContextSrc.INTRADAY_QUOTATION, intraday_block)
        
        with self.profiler.section('factor_compute'):
            factor_processor = self.factor_factory.create_batch_pipeline(ContextSrc.INTRADAY_QUOTATION)
            factors_block = factor_processor(context)
        return factors_block

    def iterate_tasks(self) -> T.Generator[tuple[datetime.date, FactorPanelBlockDense], None, None]:
        for date in self.generate_date_tasks():
            yield date, self.process_factor(date)
        
