import os
import time
import glob
import datetime

import numpy as np
import pandas as pd
from loguru import logger
import torch
import torch.utils.data as data

from qarts.core import DailyPanelBlockIndexed, PanelBlockDense, IntradayPanelBlockDense
from qarts.loader import ParquetPanelLoader, VariableLoadSpec
from qarts.modeling.factors import PipelineFactory, ContextSrc, FactorContext
from qarts.custom.factor import get_factor_group

from .registry import register_dataset, get_fill_method


@register_dataset('intraday_cs')
class IntradayDataset(data.Dataset):

    def __init__(self, config: dict, is_training: bool = False):
        self.is_training = is_training
        self.config = config 
        self.loader = ParquetPanelLoader()
        
        self.factor_group_name = config.get('factor_group', 'default')
        self.factor_group = get_factor_group(self.factor_group_name)
        self.target_group_name = config.get('target_group', 'default')
        self.target_group = get_factor_group(self.target_group_name)
        self.factor_factory = PipelineFactory(self.factor_group)
        self.target_factory = PipelineFactory(self.target_group)
        self.factor_pipeline = self.factor_factory.create_batch_pipeline(ContextSrc.INTRADAY_QUOTATION)
        self.target_pipeline = self.target_factory.create_batch_pipeline(ContextSrc.INTRADAY_QUOTATION)
        self.intraday_fields = list(set(
            self.factor_factory.input_fields[ContextSrc.INTRADAY_QUOTATION] + \
            self.target_factory.input_fields[ContextSrc.INTRADAY_QUOTATION]))

        self.intraday_prefix = config['intraday_prefix']
        self.sample_per_day = config.get('sample_per_day', 1)
        self.freq = config.get('freq', 5)
        self.file_pattern = os.path.join(self.intraday_prefix, config.get('file_pattern', '*.parquet'))
        self.all_files = sorted(glob.glob(self.file_pattern))

        if self.is_training:
            start_date = config.get('train_start', '2012-01-01')
            end_date = config.get('train_end', '2023-01-01')
        else:
            start_date = config.get('test_start', '2023-01-01')
            end_date = config.get('test_end', '2025-11-28')
        start_date, end_date = map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), (start_date, end_date))
        self.files = [
            file for file in self.all_files 
            if os.path.basename(file) >= start_date.strftime('%Y%m%d') and os.path.basename(file) <= end_date.strftime('%Y%m%d') 
        ]
        self.start_date = start_date
        self.end_date = end_date
        logger.info(f'Loading {len(self.files)}/{len(self.all_files)} files {start_date.date()}-{end_date.date()} from {self.intraday_prefix}')
        logger.info(f'Loaded data from {start_date.date()} to {end_date.date()} {len(self.files)} files, is_training: {self.is_training}')

        self.daily_block, self.daily_fields_require_adjustment = self.load_daily_data()
        logger.info(f'Dataset prepared.')

    def load_daily_data(self):
        required_daily_fields = list(set(
            self.factor_factory.input_fields[ContextSrc.DAILY_QUOTATION] + \
            self.target_factory.input_fields[ContextSrc.FUTURE_DAILY_QUOTATION]))
        required_daily_fields_before_adjustment = list(set([field.replace('adjusted_', '') for field in required_daily_fields]))
        daily_block = self.loader.load_daily_quotation(fields=required_daily_fields_before_adjustment + ['instrument', 'factor'])
        daily_fields_require_adjustment = [field.replace('adjusted_', '') for field in required_daily_fields if field.startswith('adjusted_')]
        daily_block.ensure_order('datetime-first')
        daily_block = daily_block.between(start_date=self.start_date, end_date=self.end_date + datetime.timedelta(days=7))
        if 'alpha' in daily_block.data.columns:
            daily_block.data['alpha'] = daily_block.data['alpha'].fillna(daily_block.data['daily_return'])
        return daily_block, daily_fields_require_adjustment

    def init_context_with_daily(self, date: pd.Timestamp):
        start_date = date - datetime.timedelta(days=300) # 200
        yesterday = date - datetime.timedelta(days=1)
        daily_block: DailyPanelBlockIndexed = self.daily_block.between(start_date=start_date, end_date=date)
        daily_block.adjust_field_by_last(fields=self.daily_fields_require_adjustment)
        daily_block = daily_block.filter_instrument_by_count(min_count=180)
        daily_block = daily_block.between(start_date=start_date, end_date=yesterday)
        daily_block.ensure_order('instrument-first')
        context = FactorContext.from_daily_block(daily_block)
        return context

    def init_future_context_with_daily(self, history_context: FactorContext, date: pd.Timestamp):
        end_date = date + datetime.timedelta(days=21)
        daily_block_future: DailyPanelBlockIndexed = self.daily_block.between(start_date=date, end_date=end_date)
        daily_block_future.adjust_field_by_first(fields=self.daily_fields_require_adjustment)
        daily_block_future.ensure_order('instrument-first')
        columns = list(daily_block_future.data.columns)
        daily_block_future = PanelBlockDense.from_indexed_block(
            daily_block_future,
            required_columns=columns,
            fill_methods=[get_fill_method(c) for c in columns],
            frequency='1D',
            inst_cats=history_context.inst_categories
        )
        context = FactorContext.from_daily_block(daily_block_future, is_future=True)
        return context

    def load_intraday_block(self, context: FactorContext, date: pd.Timestamp):
        load_spec = VariableLoadSpec(var_type='quotation', load_kwargs={'date': date, 'fields': self.intraday_fields + ['instrument']})
        block = self.loader.load_intraday(load_spec)
        intraday_block = IntradayPanelBlockDense.from_indexed_block(
            block, 
            required_columns=self.intraday_fields, 
            fill_methods=[get_fill_method(c) for c in self.intraday_fields],
            frequency='1min',
            inst_cats=context.inst_categories,
            is_intraday=True,
            max_nan_count=100
        )
        return intraday_block

    def __getitem__(self, index: int):
        f = self.files[index]
        # logger.debug(f"Loading file {os.path.basename(f)} (index {index})")
        date = pd.Timestamp(os.path.basename(f).split('.')[0])
        context = self.init_context_with_daily(date)
        context_future = self.init_future_context_with_daily(context, date)
        intraday_block = self.load_intraday_block(context, date)
        context.register_block(ContextSrc.INTRADAY_QUOTATION, intraday_block)
        context_future.register_block(ContextSrc.INTRADAY_QUOTATION, intraday_block)
        factors_block = self.factor_pipeline(context)
        targets_block = self.target_pipeline(context_future)
        return {
            "features": factors_block.data,
            "targets": targets_block.data,
            "timesteps": intraday_block.timestamps,
            "instruments": intraday_block.instruments,
            "is_valid_instruments": intraday_block.is_valid_instruments,
            "feature_names": factors_block.fields,
            "target_names": targets_block.fields,
        }

    def __len__(self):
        return len(self.files)


