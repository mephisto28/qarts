import fire
from qarts.loader import ParquetPanelLoader
from qarts.pipelines.factor_process import get_factor_process_pipeline
from qarts.pipelines import DataStoreProcessor

@fire.Fire
def main(factor_group_name: str):
    loader = ParquetPanelLoader(compression='zstd')
    pipeline = get_factor_process_pipeline(factor_group_name)
    storage = DataStoreProcessor(loader, input_name=pipeline.processors[-1].name, output_name=factor_group_name)
    pipeline.register_processor(storage)
    pipeline.run()


if __name__ == "__main__":
    main()