import datetime
import typing as T
from collections import defaultdict

import numpy as np
from loguru import logger

from qarts.core import FactorPanelBlockDense, IntradayPanelBlockDense, DailyPanelBlockIndexed
from qarts.loader.dataloader import PanelLoader, VariableLoadSpec
from .base import FactorSpec, get_factor
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
        def pipeline(context: FactorContext) -> FactorPanelBlockDense:
            factors_block = FactorPanelBlockDense.init_empty_from_context(
                context.inst_categories,
                timestamps=context.blocks[src].timestamps,
                fields=[f.name for f in self.factors],
                freq=context.blocks[src].frequency
            )
            context_ops = ContextOps(context, is_online=False)
            for i, factor in enumerate(self.factors):
                placeholder = factors_block.data[i]
                factor.compute_from_context(context_ops, placeholder)
            return factors_block

        return pipeline

    def create_online_engine(self) -> T.Callable:
        pass


class IntradayOnlineProcessingEngine:
    pass # TODO



class IntradayBatchProcessingEngine:

    def __init__(self, loader: PanelLoader, factor_specs: list[FactorSpec]):        
        self.loader = loader
        self.factor_factory = PipelineFactory(factor_specs)
        self.factors = self.factor_factory.factors
        self.daily_block, self.daily_fields_require_adjustment = self._load_daily_block()
        self.intraday_fields = self.factor_factory.input_fields[ContextSrc.INTRADAY_QUOTATION]
    
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
        required_dates = set(available_dates) - set(existing_dates)

        for date in required_dates:
            yield datetime.datetime.combine(date, datetime.time(0, 0, 0))

    def process_date(self, date: datetime.datetime) -> FactorPanelBlockDense:
        logger.info(f"Processing date: {date}")
        start_date = date - datetime.timedelta(days=365)
        yesterday = date - datetime.timedelta(days=1)
        daily_block = self.daily_block.between(start_date=start_date, end_date=yesterday)
        daily_block.adjust_field_by_last(fields=self.daily_fields_require_adjustment)
        daily_block.ensure_order('instrument-first')
        context = FactorContext.from_daily_block(daily_block)

        load_spec = VariableLoadSpec(var_type='quotation', load_kwargs={'date': date, 'fields': self.intraday_fields + ['instrument']})
        block = self.loader.load_intraday(load_spec)
        context.register_block(
            ContextSrc.INTRADAY_QUOTATION, 
            IntradayPanelBlockDense.from_indexed_block(
                block, 
                required_columns=self.intraday_fields, 
                fill_methods=[1],
                frequency='1min',
                inst_cats=context.inst_categories,
                is_intraday=True
            )
        )
        
        factor_compute = self.factor_factory.create_batch_pipeline(ContextSrc.INTRADAY_QUOTATION)
        factors_block = factor_compute(context)
        return factors_block

    def iterate_tasks(self) -> T.Generator[tuple[datetime.date, FactorPanelBlockDense], None, None]:
        for date in self.generate_date_tasks():
            yield date, self.process_date(date)
        
