import typing as T

from qarts.loader import ParquetPanelLoader
from qarts.modeling.factors import FactorsProcessor, PipelineFactory as FactorsPipelineFactory, ContextSrc, get_factor_group
from qarts.pipelines import BatchProcessPipeline, Processor, GlobalContext
from .provider import DailyAndIntradayProvider


class FactorsProcessorWrapper(Processor):
    _name: str = 'factors'

    def __init__(self, processor: FactorsProcessor, factor_group_name: str = 'default'):
        self.factor_group_name = factor_group_name
        self.processor = processor
        self.daily_fields = self.processor.get_daily_fields()
        self.intraday_fields = self.processor.get_intraday_fields()

    @property
    def input_fields(self) -> list[str]:
        if self.processor.has_future_data:
            return ['factor_context_future']
        else:
            return ['factor_context']

    @property
    def name(self) -> str:
        return f'factors_{self.factor_group_name}'

    def process(self, context: GlobalContext) -> T.Any:
        factor_context = context.get(self.input_fields[0])
        factors_block = self.processor(factor_context)
        return factors_block

    @classmethod
    def from_factor_group(cls, factor_group_name: str):
        factor_group = get_factor_group(factor_group_name)
        factor_factory = FactorsPipelineFactory(factor_group)
        factors_processor = factor_factory.create_batch_pipeline(ContextSrc.INTRADAY_QUOTATION)
        return cls(factors_processor, factor_group_name=factor_group_name)


def get_factor_process_pipeline(factor_group_name: str) -> BatchProcessPipeline:
    from qarts.custom.factor import get_factor_group
    factors_processor = FactorsProcessorWrapper.from_factor_group(factor_group_name)
    provider = DailyAndIntradayProvider(
        loader=ParquetPanelLoader(), 
        daily_fields=factors_processor.daily_fields, 
        intraday_fields=factors_processor.intraday_fields,
        return_factor_context=True
    )

    pipeline = BatchProcessPipeline()
    pipeline.register_tasks(provider.generate_tasks)
    pipeline.register_processor(provider)
    pipeline.register_processor(factors_processor)
    return pipeline
