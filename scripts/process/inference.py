import os
import json
import datetime

import fire
from qarts.loader import ParquetPanelLoader, VariableLoadSpec
from qarts.pipelines import DataStoreProcessor, DailyAndIntradayProvider
from qarts.pipelines.inferencer import get_inference_pipeline

@fire.Fire
def main(name: str, epoch: int):
    output_name = f'models/{name}_e{epoch:02d}'
    d = os.path.dirname
    project_dir = d(d(d(os.path.abspath(__file__))))
    config_path = os.path.join(project_dir, 'experiments', 'config', f'{name}.json')
    config = json.load(open(config_path))
    loader = ParquetPanelLoader(compression='zstd')
    pipeline = get_inference_pipeline(config_path, epoch)
    provider = pipeline.get_processor_by_type(DailyAndIntradayProvider)
    provider.target_specs = [
        VariableLoadSpec(var_type='factor', load_kwargs={'factor': output_name})]
    provider.start_date = datetime.datetime.strptime(config.get('dataset', {}).get('test_start', '2022-01-01'), '%Y-%m-%d').date()
    provider.end_date = datetime.datetime.strptime(config.get('dataset', {}).get('test_end', '2026-01-01'), '%Y-%m-%d').date()
    storage = DataStoreProcessor(loader, input_name=pipeline.processors[-1].name, output_name=f'models/{name}_e{epoch:02d}')
    pipeline.register_processor(storage)
    pipeline.run()
