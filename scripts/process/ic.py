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
    factor_group_name: str, 
    start_date: str = '2023-01-01',
    end_date: str = '2025-07-01',
    overwrite: bool = False,
):
    d = os.path.dirname
    project_dir = d(d(d(os.path.abspath(__file__))))

    target_group_name = 'targets_with_costs_10m_3D'
    factors_processor = FactorsProcessorWrapper.from_factor_group(factor_group_name)
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

    evaluators = []
    for f in factors_processor.processor.factors:
        evaluator = EvaluatorProcessor(
            pred_name=f'factors_{factor_group_name}',
            output_name=f'factors/{factor_group_name}/{f.name}',
            targets_name=f'factors_{target_group_name}', 
            pred_fields=[f.name], 
            target_fields=[f.name for f in targets_processor.processor.factors],
        )
        if os.path.exists(os.path.join(evaluator.output_dir, "eval_results.pkl")) and not overwrite:
            logger.info(f"Evaluation results already exist for {f.name}")
        else:
            pipeline.register_processor(evaluator)
        evaluators.append(evaluator)

    # pipeline.run()
    for evaluator in evaluators:
        # evaluator.finalize()
        build_report(evaluator.output_dir)
