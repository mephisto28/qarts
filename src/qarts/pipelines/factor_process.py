import typing as T

from qarts.loader import ParquetPanelLoader
from qarts.modeling.factors import FactorsProcessor, PipelineFactory as FactorsPipelineFactory, ContextSrc, get_factor_group
from qarts.pipelines import BatchProcessPipeline, Processor, GlobalContext
from .provider import DailyAndIntradayProvider


class FactorsProcessorWrapper(Processor):
    name: str = 'factor'

    def __init__(self, processor: FactorsProcessor):
        self.processor = processor
        self.daily_fields = self.processor

    def process(self, context: GlobalContext) -> T.Any:
        factor_context = context.get('factor_context')
        factors_block = self.processor(factor_context)
        return factors_block

    @classmethod
    def from_factor_group(cls, factor_group_name: str):
        factor_group = get_factor_group(factor_group_name)
        factor_factory = FactorsPipelineFactory(factor_group)
        factors_processor = factor_factory.create_batch_pipeline(ContextSrc.INTRADAY_QUOTATION)
        return cls(factors_processor)


def get_factor_process_pipeline(factor_group_name: str) -> BatchProcessPipeline:
    factors_processor = FactorsProcessorWrapper.from_factor_group(factor_group_name)
    provider = DailyAndIntradayProvider(
        loader=ParquetPanelLoader(), 
        daily_fields=['adjusted_close'], 
        intraday_fields=['1min_v4_barra4_total']
    )

    pipeline = BatchProcessPipeline()
    pipeline.register_tasks(provider.generate_tasks)
    pipeline.register_processor(provider)
    pipeline.register_processor(factors_processor)
    return pipeline
