import unittest

import numpy as np
import pandas as pd
from loguru import logger
from qarts.modeling.factors import FactorNames, FactorSpec, ContextSrc, IntradayBatchProcessingEngine
from qarts.loader import ParquetPanelLoader
from qarts.custom.factor.factor_group import get_factor_group


class TestFactorEngine(unittest.TestCase):

    def test_features_smoketest(self):
        factors = get_factor_group('fg260114') + get_factor_group('targets_with_costs_10m_3D_with_rank') + get_factor_group('filters')
        loader = ParquetPanelLoader()
        engine = IntradayBatchProcessingEngine(loader, factors)
        desc_msgs = []
        # for date, factors_block in tqdm(engine.iterate_tasks()):
        from datetime import date
        date = pd.Timestamp(date(2023, 1, 4))
        # date = pd.Timestamp(date(2018, 2, 22))
        factors_block = engine.process_factor(date)
        flt = None
        for i,factor in enumerate(engine.factors):
            value = factors_block.get_view(factor.name, return_valid=True)
            if factor.name.startswith(FactorNames.DAILY_RECENT_VACANCY):
                flt = flt & (value.mean(axis=1) > 0.9) if flt is not None else (value.mean(axis=1) > 0.9)
            mean = np.nanmean(value)
            std = np.nanstd(value)
            min = np.nanmin(value)
            max = np.nanmax(value)
            col_nan_count = (np.isnan(value).sum(axis=0) > 0).sum()
            row_nan_count = (np.isnan(value).sum(axis=1) > 0).sum()
            total_nan_count = np.isnan(value).sum()
            total_value = value.size
            factor_name = factor.name + ' ' * (40 - len(factor.name))
            desc_msgs.append(f'{i:03d} {factor_name} mean={mean:.4f}\tstd={std:.4f}\tmin={min:.3f}\tmax={max:.3f}\tcnan={col_nan_count:03d},\trnan={row_nan_count:03d},\tnan={total_nan_count}/{total_value}')
            
        msg = 'factor_result:\n' + '\n'.join(desc_msgs)
        logger.info(msg)

        # for i,factor in enumerate(engine.factors):
        #     value = factors_block.get_view(factor.name, return_valid=True)[flt]
        #     mean = np.nanmean(value)
        #     std = np.nanstd(value)
        #     min = np.nanmin(value)
        #     max = np.nanmax(value)
        #     col_nan_count = (np.isnan(value).sum(axis=0) > 0).sum()
        #     row_nan_count = (np.isnan(value).sum(axis=1) > 0).sum()
        #     total_nan_count = np.isnan(value).sum()
        #     total_value = value.size
        #     factor_name = factor.name + ' ' * (32 - len(factor.name))
        #     desc_msgs.append(f'{i:03d} {factor_name} mean={mean:.4f}\tstd={std:.4f}\tmin={min:.3f}\tmax={max:.3f}\tcnan={col_nan_count:03d},\trnan={row_nan_count:03d},\tnan={total_nan_count}/{total_value}')
            
        # msg = 'factor_result:\n' + '\n'.join(desc_msgs)
        # logger.info(msg)

    def test_targets_smoketest(self):
        pass

