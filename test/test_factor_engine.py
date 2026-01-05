import unittest

import numpy as np
import pandas as pd
from loguru import logger
from qarts.modeling.factors import FactorNames, FactorSpec, ContextSrc
from qarts.loader import ParquetPanelLoader
from qarts.modeling.factors.engine import IntradayBatchProcessingEngine, FactorSpec, ContextSrc
from qarts.modeling.factors.high_order import DailyKurtosis


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
        spec = FactorSpec(name=FactorNames.PRICE_POSITION, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_high', 'adjusted_low'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0.5, 'scale': 5})
        factors.append(spec)
    for i in windows:
        spec = FactorSpec(name=FactorNames.DAILY_MOM, input_fields={
            ContextSrc.DAILY_QUOTATION: ['adjusted_close'],
            ContextSrc.INTRADAY_QUOTATION: ['mid_price']
        }, window=i, params={'shift': 0, 'scale': 50})
        factors.append(spec)
    for i in windows[1:]:
        spec = FactorSpec(name=FactorNames.DAILY_VOLATILITY, input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1']
        }, window=i, params={'shift': 0.025, 'scale': 100})
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.DAILY_SKEWNESS, input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1']
        }, window=i, params={'shift': 0.0, 'scale': 1})
        factors.append(spec)
    for i in windows[3:]:
        spec = FactorSpec(name=FactorNames.DAILY_KURTOSIS, input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1']
        }, window=i, params={'shift': 0.0, 'scale': 0.5})
        factors.append(spec)
    for i in windows[1:5]:
        spec = FactorSpec(name=FactorNames.DAILY_VOLVOL, input_fields={
            ContextSrc.DAILY_QUOTATION: ['daily_return'],
            ContextSrc.FACTOR_CACHE: ['daily_mom_1']
        }, window=i, params={'shift': 0.01, 'scale': 200, 'window2': i*3})
        factors.append(spec)
    
    # ---- residual features ----
    for i in windows[1:]:
        spec = FactorSpec(name=FactorNames.DAILY_MOM_SUM, input_fields={
            ContextSrc.DAILY_QUOTATION: ['alpha'],
            ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total'],
        }, window=i, params={'shift': 0.0, 'scale': 50})
        factors.append(spec)
    for i in windows[1:]:
        spec = FactorSpec(name=FactorNames.DAILY_VOLATILITY, input_fields={
            ContextSrc.DAILY_QUOTATION: ['alpha'],
            ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total']
        }, window=i, params={'shift': 0.02, 'scale': 100})
        factors.append(spec)
    for i in windows[2:]:
        spec = FactorSpec(name=FactorNames.DAILY_SKEWNESS, input_fields={
            ContextSrc.DAILY_QUOTATION: ['alpha'],
            ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total']
        }, window=i, params={'shift': 0.0, 'scale': 1})
        factors.append(spec)
    for i in windows[3:]:
        spec = FactorSpec(name=FactorNames.DAILY_KURTOSIS, input_fields={
            ContextSrc.DAILY_QUOTATION: ['alpha'],
            ContextSrc.INTRADAY_QUOTATION: ['1min_v4_barra4_total']
        }, window=i, params={'shift': 0.0, 'scale': 0.5})
        factors.append(spec)

    # ---- targets ----
    for i in [0, 1, 2, 3]:
        spec = FactorSpec(name=FactorNames.FUTURE_DAY_TARGETS, input_fields={
            ContextSrc.FUTURE_DAILY_QUOTATION: ['adjusted_close'],
            ContextSrc.INTRADAY_QUOTATION: ['ask_price1', 'mid_price']
        }, window=i)
        factors.append(spec)
    for i in [10, 30, 60]:
        spec = FactorSpec(name=FactorNames.TODAY_TARGETS, input_fields={
            ContextSrc.INTRADAY_QUOTATION: ['ask_price1', 'bid_price1', 'mid_price']
        }, window=i)
        factors.append(spec)
    return factors


class TestFactorEngine(unittest.TestCase):

    def test_features_smoketest(self):
        factors = generate_factor_specs()
        loader = ParquetPanelLoader()
        engine = IntradayBatchProcessingEngine(loader, factors)
        desc_msgs = []
        # for date, factors_block in tqdm(engine.iterate_tasks()):
        from datetime import date
        date = pd.Timestamp(date(2023, 1, 4))
        factors_block = engine.process_factor(date)
        for i,factor in enumerate(engine.factors):
            value = factors_block.get_view(factor.name, return_valid=True)
            mean = np.nanmean(value)
            std = np.nanstd(value)
            min = np.nanmin(value)
            max = np.nanmax(value)
            col_nan_count = (np.isnan(value).sum(axis=0) > 0).sum()
            row_nan_count = (np.isnan(value).sum(axis=1) > 0).sum()
            total_nan_count = np.isnan(value).sum()
            total_value = value.size
            factor_name = factor.name + ' ' * (27 - len(factor.name))
            desc_msgs.append(f'{i:03d} {factor_name} mean={mean:.3f}\tstd={std:.3f}\tmin={min:.3f}\tmax={max:.3f}\tcnan={col_nan_count:03d},\trnan={row_nan_count:03d},\tnan={total_nan_count}/{total_value}')
        msg = 'factor_result:\n' + '\n'.join(desc_msgs)
        logger.info(msg)

    def test_targets_smoketest(self):
        pass

