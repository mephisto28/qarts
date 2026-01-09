import os
import json
import fire
from qarts.loader import ParquetPanelLoader
from qarts.pipelines import DataStoreProcessor
from qarts.pipelines.inferencer import get_inference_pipeline

@fire.Fire
def main(name: str, epoch: int):
    d = os.path.dirname
    project_dir = d(d(d(os.path.abspath(__file__))))
    config_path = os.path.join(project_dir, 'experiments', 'config', f'{name}.json')
    loader = ParquetPanelLoader(compression='zstd')
    pipeline = get_inference_pipeline(config_path, epoch)
    storage = DataStoreProcessor(loader, input_name=pipeline.processors[-1].name, output_name=f'models/{name}_e{epoch:02d}')
    pipeline.register_processor(storage)
    pipeline.run()


if __name__ == "__main__":
    main()