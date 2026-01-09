import datetime
import typing as T
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from .convert_utils import build_ranges, make_time_grid, densify_features_from_df


@dataclass
class PanelDataIndexed:
    data: pd.DataFrame | pd.Series
    order: str  # 'datetime-first' or 'instrument-first' or None

    def __post_init__(self):
        assert self.data.index.names[0] == 'datetime'
        assert self.data.index.names[1] == 'instrument'

    @property
    def datetimes(self) -> pd.Index:
        return self.data.index.get_level_values('datetime')

    @property
    def instruments(self) -> pd.Index:
        return self.data.index.get_level_values('instrument')

    def between(self, start_date: datetime.date | str = None, end_date: datetime.date | str = None) -> 'PanelDataIndexed':
        self.ensure_order('datetime-first')
        df = self.data.loc[start_date:end_date].copy()
        return type(self)(df, order='datetime-first')

    def filter_instrument_by_count(self, min_count: int):
        df = self.data
        df = df[df.groupby(level='instrument').transform('size') > min_count]
        return type(self)(df, order=self.order)

    def ensure_order(self, order: str):
        self.order = order
        if order == 'datetime-first':
            datetime_idx = self.data.index.get_level_values('datetime')
            if not datetime_idx.is_monotonic_increasing:
                self.data.sort_index(level='datetime', inplace=True)
        elif order == 'instrument-first':
            instrument_idx = self.data.index.get_level_values('instrument')
            if not instrument_idx.is_monotonic_increasing:
                self.data.sort_index(level='instrument', inplace=True)
        elif order != 'none':
            raise ValueError(f"Invalid order: {order}")

    def intersect_index(self, other: 'PanelDataIndexed') -> pd.MultiIndex:
        common_index = self.data.index.intersection(other.data.index)
        return common_index

    def align_index(self, index: pd.MultiIndex) -> 'PanelDataIndexed':
        df = self.data.loc[index]
        block = PanelDataIndexed(df, order=self.order)
        block.ensure_order(self.order)
        return block


@dataclass
class PanelSeriesIndexed(PanelDataIndexed):
    data: pd.Series
    field: T.Optional[str] = None
    order: str = 'none'  # 'datetime-first' or 'instrument-first' or None


@dataclass
class PanelBlockIndexed(PanelDataIndexed):
    data: pd.DataFrame # (NxT, F) dataframe with multi-index (datetime, instrument)
    order: str = 'none' # 'datetime-first' or 'instrument-first' or None

    def __post_init__(self):
        super().__post_init__()
        self.fields = list(self.data.columns)

    def get_field(self, field: str) -> PanelSeriesIndexed:
        return PanelSeriesIndexed(self.data[field], field=field, order=self.order)
    
    def get_cross_sectional_groups(self) -> DataFrameGroupBy:
        return self.data.groupby(level='datetime')

    def get_instrument(self, instrument: str) -> pd.DataFrame:
        f = self.data.index.get_level_values('instrument') == instrument
        return self.data[f]

    def get_fields(self, fields: list[str]) -> 'PanelBlockIndexed':
        return type(self)(self.data[fields], order=self.order)
        
    def merge(self, other: T.Union['PanelBlockIndexed', 'PanelSeriesIndexed'], how: str = 'left') -> 'PanelBlockIndexed':
        merged_df = self.data.join(other.data, how=how)
        return PanelBlockIndexed(merged_df, order=self.order)

    @classmethod
    def from_dataframe(cls, df, src='df'):
        if isinstance(df.index, pd.MultiIndex):
            assert df.index.names[0] == 'datetime'
            assert df.index.names[1] == 'instrument'
            return PanelBlockIndexed(df, order='datetime-first')
        else:
            valid_datetime_cols = ['datetime', 'timestamp']
            if not df.index.name in valid_datetime_cols:
                valid_datetime_cols = [c for c in valid_datetime_cols if c in df.columns]
                if len(valid_datetime_cols) == 0:
                    raise ValueError(f"No valid datetime column found in {src}, available columns: {df.columns}")
                datetime_col = valid_datetime_cols[0]
                df.set_index(datetime_col, inplace=True)

            valid_instrument_cols = ['instrument', 'instruments', 'stock_code']
            valid_instrument_cols = [c for c in valid_instrument_cols if c in df.columns]
            if len(valid_instrument_cols) == 0:
                raise ValueError(f"No valid instrument column found in {src}, available columns: {df.columns}")
            instrument_col = valid_instrument_cols[0]
            df.set_index(instrument_col, append=True, inplace=True)
            df.index.names = ['datetime', 'instrument']
        return PanelBlockIndexed(df, order='datetime-first')

    @classmethod
    def from_series_list(cls, s_list: list[PanelSeriesIndexed], src='s_list') -> 'PanelBlockIndexed':
        names = ', '.join([s.field for s in s_list])
        df = pd.DataFrame({s.field: s.data for s in s_list})
        return cls.from_dataframe(df, src=names)

    @classmethod
    def from_dense_block(cls, block: 'PanelBlockDense') -> 'PanelBlockIndexed':
        df = block.to_dataframe()
        return cls.from_dataframe(df)


@dataclass
class PanelBlockDense:

    instruments: np.ndarray
    timestamps: np.ndarray
    data: np.ndarray # (F, N, T) to accelerate f(row-wise) access
    fields: list[str]
    frequency: str
    cursor: int = -1
    is_valid_instruments: np.ndarray = None

    def __post_init__(self):
        assert isinstance(self.instruments, np.ndarray)
        assert isinstance(self.timestamps, np.ndarray)
        assert isinstance(self.fields, list)
        if self.is_valid_instruments is None:
            self.is_valid_instruments = (~np.isnan(self.data[0])).sum(axis=-1) > 0
    
    def get_view(self, field: T.Optional[str], return_valid: bool = False) -> np.ndarray:
        if field is None:
            return self.data
        field_idx = self.fields.index(field)
        data = self.data[field_idx]
        if return_valid:
            return data[self.is_valid_instruments]
        return data

    def get_current_view(self, field: T.Optional[str] = None, window: int = 1) -> np.ndarray:
        block = self.get_view(field)
        if self.cursor < 0:
            return block

        if window == 1:
            return block[:, self.cursor]
        else:
            start_idx = max(0, self.cursor - window + 1)
            end_idx = min(self.cursor + 1, block.shape[2])
            return block[:, start_idx:end_idx]

    def get_copy(self, fields: T.Optional[list[str]] = None) -> 'PanelBlockDense':
        if fields is None:
            fields = self.fields
        fields_idx = [self.fields.index(field) for field in fields]
        return PanelBlockDense(
            instruments=self.instruments,
            timestamps=self.timestamps,
            data=self.data[fields_idx],
            fields=fields,
            frequency=self.frequency
        )

    def get_dataframe(self, instrument: str = None, timestamp: pd.Timestamp | int = None) -> pd.DataFrame:
        assert timestamp is not None or instrument is not None, "Either timestamp or instrument must be provided"
        if instrument is not None:
            data = self.data[instrument].T
            index = self.timestamps
        elif self.cursor < 0:
            return self.to_dataframe()
        else:
            cursor = self.cursor
            if timestamp is not None:
                cursor = np.searchsorted(self.timestamps, timestamp)
            data = self.data[..., cursor].T
            index = self.instruments
        return pd.DataFrame(data, index=index, columns=self.fields)

    def to_dataframe(self) -> pd.DataFrame:
        if self.is_valid_instruments is not None:
            data = self.data[:, self.is_valid_instruments]
            instruments = self.instruments[self.is_valid_instruments]
        else:
            data = self.data
            instruments = self.instruments
        F, N, T = data.shape
        data_2d = np.transpose(data, (2, 1, 0)).reshape(T * N, F)
        index = pd.MultiIndex.from_product([self.timestamps, instruments], names=['datetime', 'instrument'])
        columns = self.fields
        return pd.DataFrame(data_2d, index=index, columns=columns)

    @classmethod
    def init_empty_from_context(cls, instruments: np.ndarray, timestamps: np.ndarray, fields: list[str], freq: str) -> 'PanelBlockDense':
        return PanelBlockDense(
            instruments=instruments,
            timestamps=timestamps,
            data=np.empty((len(fields), len(instruments), len(timestamps)), dtype=np.float32),
            fields=fields,
            frequency=freq
        )
    
    @classmethod
    def from_indexed_block(
        cls, 
        block: PanelBlockIndexed,
        required_columns: list[str],
        fill_methods: list[str],
        frequency: str = '1min', 
        inst_cats: np.ndarray = None,
        is_intraday: bool = False,
        max_nan_count: int = 0,
        backward_fill: bool = False
    ) -> 'PanelBlockDense':
        block.ensure_order('instrument-first')
        if isinstance(block, IntradayPanelBlockIndexed) or is_intraday:
            trading_date = block.datetimes[0].date()
            timestamps = make_time_grid(trading_date, frequency=frequency)
        elif isinstance(block, DailyPanelBlockIndexed) or isinstance(block, PanelBlockIndexed):
            timestamps = block.datetimes.unique().sort_values().values
        else:
            raise ValueError(f"Invalid block type: {type(block)}")
        grid_ns = timestamps.astype("datetime64[ns]").view("int64")
        
        inst = block.instruments
        if inst_cats is None:
            inst_cats = inst.unique().sort_values()
        inst_code = pd.Categorical(inst, categories=inst_cats, ordered=True).codes.astype(np.int32)
        n_inst = len(inst_cats)
        starts, ends = build_ranges(inst_code, n_inst)
        values = densify_features_from_df(block.data, starts, ends, grid_ns, inst_cats, required_columns, fill_methods, backward_fill)
        inst_cats = np.array(inst_cats)
        if max_nan_count > 0:
            instrument_nan_counts = np.max(np.sum(np.isnan(values), axis=-1), axis=0)
            intrument_filter = instrument_nan_counts < max_nan_count
        else:
            intrument_filter = None
            # values = values[:, intrument_filter]
            # inst_cats = inst_cats[intrument_filter]
        
        return PanelBlockDense(
            instruments=np.array(inst_cats),
            timestamps=timestamps,
            data=values, # (F, N, T)
            fields=required_columns,
            frequency=frequency,
            is_valid_instruments=intrument_filter
        )


@dataclass
class IntradayPanelBlockIndexed(PanelBlockIndexed):
    pass


@dataclass
class DailyPanelBlockIndexed(PanelBlockIndexed):

    @classmethod
    def from_base_block(cls, block: PanelBlockIndexed) -> 'DailyPanelBlockIndexed':
        return DailyPanelBlockIndexed(block.data, order=block.order)
    
    def adjust_field_by_last(self, fields: str):
        daily_df = self.data
        last_factor = daily_df.groupby(level='instrument', sort=False)['factor'].transform('last')
        factor = daily_df['factor']
        for field in fields:
            new_field = 'adjusted_' + field
            if field in ('volume', 'turnover', 'amount'):
                daily_df[new_field] = daily_df[field] * factor / last_factor
            else:
                daily_df[new_field] = daily_df[field] / factor * last_factor
            if new_field not in self.fields:
                self.fields.append(new_field)

    def adjust_field_by_first(self, fields: str):
        daily_df = self.data
        first_factor = daily_df.groupby(level='instrument', sort=False)['factor'].transform('first')
        factor = daily_df['factor']
        for field in fields:
            new_field = 'adjusted_' + field
            daily_df[new_field] = daily_df[field] / factor * first_factor
            if new_field not in self.fields:
                self.fields.append(new_field)

@dataclass
class IntradayPanelBlockDense(PanelBlockDense):
    pass


@dataclass
class DailyPanelBlockDense(PanelBlockDense):
    pass


@dataclass
class FactorPanelBlockIndexed(PanelBlockIndexed):
    pass


@dataclass
class FactorPanelBlockDense(PanelBlockDense):
    pass
