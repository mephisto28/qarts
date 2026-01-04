import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from qarts.modeling.factors import FactorNames, FactorSpec, ContextSrc
from qarts.loader import ParquetPanelLoader
from qarts.modeling.factors.engine import IntradayBatchProcessingEngine, FactorSpec, ContextSrc


def generate_factor_specs():

    factors = []
    windows = [1, 2, 5, 10, 21, 63, 126]
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.PRICE_DEV_FROM_MA, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0, 'scale': 100})
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.PRICE_DEV_FROM_VWAP, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close', 'volume'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0, 'scale': 100})
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name='price_position', input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_high', 'adjusted_low'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0.5, 'scale': 5})
        factors.append(spec)
    for i in windows:
        spec = FactorSpec(name='daily_mom', input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0, 'scale': 50})
        factors.append(spec)
    for i in windows[1:]:
        spec = FactorSpec(name='daily_volatility', input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1']
        }, window=i, params={'shift': 0.025, 'scale': 100})
        factors.append(spec)
    return factors


class TestFactorEngine(unittest.TestCase):

    def test_factor_engine(self):
        factors = generate_factor_specs()
        loader = ParquetPanelLoader()
        engine = IntradayBatchProcessingEngine(loader, factors)
        desc_msgs = []
        # for date, factors_block in tqdm(engine.iterate_tasks()):
        from datetime import date
        date = pd.Timestamp(date(2023, 1, 4))
        factors_block = engine.process_factor(date)
        for factor in engine.factors:
            value = factors_block.get_view(factor.name)
            mean = np.nanmean(value)
            std = np.nanstd(value)
            min = np.nanmin(value)
            max = np.nanmax(value)
            nan_count = np.isnan(value).sum(axis=1)
            factor_name = factor.name + ' ' * (24 - len(factor.name))
            desc_msgs.append(f'{factor_name} mean={mean:.4f},\tstd={std:.4f},\tmin={min:.4f},\tmax={max:.4f}')
        msg = 'factor_result:\n' + '\n'.join(desc_msgs)
        logger.info(msg)

