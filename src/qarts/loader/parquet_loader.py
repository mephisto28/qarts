import os
import json
import datetime
import typing as T

import pandas as pd
from loguru import logger
from qarts.core import IntradayPanelBlockIndexed, DailyPanelBlockIndexed
from .dataloader import PanelLoader, PanelBlockIndexed, VariableLoadSpec


class ParquetPanelLoader(PanelLoader):
    
    def __init__(self, prefix: str = '/', config: T.Optional[dict] = None, compression: str = 'zstd'):
        self.prefix = prefix
        if config is None:
            d = os.path.dirname
            proj_root = d(d(d(d(os.path.abspath(__file__)))))
            config_path = os.path.join(proj_root, 'config', 'files.json')
            self.config = json.load(open(config_path))
        else:
            self.config = config
        self.date_format = self.config.get('date_format', '%Y%m%d')
        self.compression = compression

    def get_dir(self, type: str, **load_kwargs) -> str:
        dir = os.path.join(self.prefix, self.config[type + '_prefix'])
        if type == 'quotation':
            return dir
        elif type == 'prediction':
            assert 'model' in load_kwargs and 'epoch' in load_kwargs, "model and epoch must be provided for prediction"
            model = load_kwargs['model']
            epoch = load_kwargs['epoch']
            use_ema = load_kwargs.get('use_ema', False)
            model_str = f'{model}/predictions_e{epoch:02d}{"_ema" if use_ema else ""}' 
            return os.path.join(dir, model_str)
        elif type == 'factor':
            factor = load_kwargs['factor']
            return os.path.join(dir, factor)
        else:
            raise ValueError(f"Invalid type: {type}")

    def list_available_dates(self, specs: list[VariableLoadSpec] | VariableLoadSpec) -> T.List[datetime.date]:
        if isinstance(specs, VariableLoadSpec):
            specs = [specs]
            return self.list_available_dates(specs)

        if len(specs) == 0:
            return []

        available_files = None
        for spec in specs:
            dir = self.get_dir(spec.var_type, **spec.load_kwargs)
            if not os.path.exists(dir):
                return []

            files = os.listdir(dir)
            available_files = set(files) if available_files is None else available_files.intersection(files)
        
        filenames = [os.path.basename(f) for f in available_files if f.endswith('.parquet')]
        available_dates = [
            datetime.datetime.strptime(f.split('.')[0], self.date_format).date()
            for f in filenames if f.startswith('20')
        ]
        return sorted(available_dates)

    def load_indexed_block_from_src(self, src: str) -> PanelBlockIndexed:
        return self.load_indexed_block_from_path(src)

    def save_intraday_to_dst(self, block: PanelBlockIndexed, dst: str, overwrite: bool = False) -> None:
        if os.path.exists(dst) and not overwrite:
            raise FileExistsError(f"File {dst} already exists, to save set overwrite=True")
        dir = os.path.dirname(dst)
        if not os.path.exists(dir):
            logger.info(f"Creating directory {dir} for saving block to {dst} ...")
            os.makedirs(dir, exist_ok=True)
        block.data.to_parquet(dst, compression=self.compression)

    def load_indexed_block_from_path(self, path: str, fields: T.Optional[list[str]] = None) -> PanelBlockIndexed:
        df = pd.read_parquet(path, columns=fields)
        return self.convert_dataframe_to_block(df, src=path)

    def load_intraday_quotation(self, date: datetime.date | str, instruments: T.Optional[str] = None, fields: T.Optional[list[str]] = None) -> PanelBlockIndexed:
        dir = self.get_dir('quotation')
        date_str = date.strftime('%Y%m%d') if isinstance(date, datetime.date) else date
        path = os.path.join(dir, f'{date_str}.parquet')
        return self.load_indexed_block_from_path(path, fields=fields)

    def save_intraday_quotation(self, block: PanelBlockIndexed, date: datetime.date | str) -> None:
        dir = self.get_dir('quotation')
        date_str = date.strftime('%Y%m%d') if isinstance(date, datetime.date) else date
        path = os.path.join(dir, f'{date_str}.parquet')
        self.save_intraday_to_dst(block, path)

    def load_intraday_factor(self, factor: str, date: datetime.date | str) -> PanelBlockIndexed:
        dir = self.get_dir('factor', factor=factor)
        date_str = date.strftime('%Y%m%d') if isinstance(date, datetime.date) else date
        path = os.path.join(dir, f'{date_str}.parquet')
        return self.load_indexed_block_from_path(path)

    def save_intraday_factor(self, block: PanelBlockIndexed, factor: str, date: datetime.date | str) -> None:
        dir = self.get_dir('factor', factor=factor)
        date_str = date.strftime('%Y%m%d') if isinstance(date, datetime.date) else date
        path = os.path.join(dir, f'{date_str}.parquet')
        self.save_intraday_to_dst(block, path)

    def load_intraday_prediction(self, model: str, epoch: int, date: datetime.date | str, use_ema: bool = False, fields: T.Optional[list[str]] = None) -> PanelBlockIndexed:
        dir = self.get_dir('prediction', model=model, epoch=epoch, use_ema=use_ema)
        date_str = date.strftime('%Y%m%d') if isinstance(date, datetime.date) else date
        path = os.path.join(dir, f'{date_str}.parquet')
        return self.load_indexed_block_from_path(path, fields=fields)
    
    def save_intraday_prediction(self, block: PanelBlockIndexed, model: str, epoch: int, date: datetime.date | str, use_ema: bool = False) -> None:
        dir = self.get_dir('prediction', model=model, epoch=epoch, use_ema=use_ema)
        date_str = date.strftime('%Y%m%d') if isinstance(date, datetime.date) else date
        path = os.path.join(dir, f'{date_str}.parquet')
        self.save_intraday_to_dst(block, path)

    def load_intraday_prediction_with_quotations(self, model: str, epoch: int, date: datetime.date | str, use_ema: bool = False) -> PanelBlockIndexed:
        prediction_block = self.load_intraday_prediction(model, epoch, date, use_ema=use_ema)
        quotation_block = self.load_intraday_quotation(date)
        return prediction_block.merge(quotation_block)

    def load_daily_quotation(self, start_date: datetime.date | str = None, end_date: datetime.date | str = None, instruments: T.Optional[str] = None, fields: T.Optional[list[str]] = None) -> PanelBlockIndexed:
        if any([f is not None for f in [start_date, end_date, instruments]]):
            logger.warning("load_daily_quotation with start_date, end_date, or instruments is not supported by parquet loader, will load all daily quotations")
        return DailyPanelBlockIndexed.from_base_block(
            self.load_indexed_block_from_path(self.config['daily_quotation_path'], fields=fields))
