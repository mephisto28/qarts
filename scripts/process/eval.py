import os
import json
import datetime

import fire
from loguru import logger

from qarts.loader import ParquetPanelLoader
from qarts.custom.factor import get_factor_group
from qarts.custom.train.eval_plot import build_report
from qarts.pipelines import BatchProcessPipeline, DailyAndIntradayProvider, FactorsProcessorWrapper, ModelInferenceProcessor
from qarts.pipelines.evaluator import EvaluatorProcessor


@fire.Fire
def main(
    name: str, 
    epoch: int,
    start_date: str = '2023-01-01',
    end_date: str = '2025-07-01',
    overwrite: bool = False,
    eval_quantiles: bool = False
):
    d = os.path.dirname
    project_dir = d(d(d(os.path.abspath(__file__))))
    config_path = os.path.join(project_dir, 'experiments', 'config', f'{name}.json')
    config = json.load(open(config_path))
    compute_rank = config.get('dataset', {}).get('rank_features', False)

    factor_group_name = config.get('dataset', {}).get('factor_group', 'default')
    target_group_name = config.get('dataset', {}).get('target_group', 'targets_with_costs_10m_3D_with_rank')
    inference_processor = ModelInferenceProcessor(config=config_path, epoch=epoch)
    evaluator = EvaluatorProcessor(
        pred_name=f'inference_{name}',
        output_name=f'{name}/eval_epoch{epoch:02d}',
        targets_name=f'factors_{target_group_name}', 
        pred_fields=inference_processor.pred_fields, 
        target_fields=inference_processor.pred_fields,
        eval_quantiles=eval_quantiles
    )
    if os.path.exists(os.path.join(evaluator.output_dir, "eval_results.pkl")) and not overwrite:
        logger.info(f"Evaluation results already exist for {name} at epoch {epoch}")
    else:
        factors_processor = FactorsProcessorWrapper.from_factor_group(factor_group_name, compute_rank=compute_rank)
        targets_processor = FactorsProcessorWrapper.from_factor_group(target_group_name)
        provider = DailyAndIntradayProvider(
            loader=ParquetPanelLoader(), 
            daily_fields=list(set(factors_processor.daily_fields + targets_processor.daily_fields)), 
            intraday_fields=list(set(factors_processor.intraday_fields + targets_processor.intraday_fields)),
            return_factor_context=True,
            start_date=start_date,
            end_date=end_date
        )
        pipeline = BatchProcessPipeline()
        pipeline.register_tasks(provider.generate_tasks)
        pipeline.register_processor(provider)
        pipeline.register_processor(factors_processor)
        pipeline.register_processor(targets_processor)
        pipeline.register_processor(inference_processor)
        pipeline.register_processor(evaluator)
        pipeline.run()

        evaluator.finalize()

    build_report(evaluator.output_dir)
