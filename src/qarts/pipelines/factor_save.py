from qarts.loader.dataloader import PanelLoader, VariableLoadSpec
from qarts.modeling.factors import IntradayBatchProcessingEngine, FactorSpec


class FactorSave:

    def __init__(self, factor_specs: list[FactorSpec], loader: PanelLoader, output_group_name: str):
        self.loader = loader
        self.output_group_name = output_group_name
        self.pipeline = IntradayBatchProcessingEngine(loader, factor_specs)

    def run(self) -> None:
        for date, factors_block in self.pipeline.iterate_tasks():
            save_spec = VariableLoadSpec(var_type='factor', load_kwargs={
                'factor': self.output_group_name, 
                'date': date.date(),
            })
            self.loader.save_intraday_factor(factors_block, save_spec)