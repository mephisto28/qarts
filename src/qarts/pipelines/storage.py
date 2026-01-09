
from loguru import logger

from qarts.core import PanelBlockIndexed, PanelBlockDense
from qarts.loader.dataloader import PanelLoader, VariableLoadSpec
from .base import Processor, GlobalContext


class DataStoreProcessor(Processor):

    def __init__(self, loader: PanelLoader, input_name: str, output_name: str):
        self.loader = loader
        self.input_name = input_name
        self.output_name = output_name

    def process(self, context: GlobalContext):
        date = context.current_datetime
        data: PanelBlockDense | PanelBlockIndexed = context.get(self.input_name)
        if isinstance(data, PanelBlockDense):
            data = PanelBlockIndexed.from_dense_block(data)
        
        save_spec = VariableLoadSpec(var_type='factor', load_kwargs={
            'factor': self.output_name, 
            'date': date.date(),
        })
        logger.info(f'Saving {self.output_name} on {date.date()} with {data.data.shape}')
        self.loader.save_intraday(data, save_spec)