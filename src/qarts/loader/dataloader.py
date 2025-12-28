import datetime
import typing as T
from dataclasses import dataclass, field

import pandas as pd
from qarts.core import PanelBlockIndexed

@T.runtime_checkable
@dataclass
class VariableLoadSpec(T.Protocol):
    var_type: str
    load_kwargs: dict = field(default_factory=dict)


@T.runtime_checkable
class PanelLoader(T.Protocol):

    def save_intraday(self, block: PanelBlockIndexed, specs: VariableLoadSpec) -> None:
        t = specs.var_type
        if t == 'quotation':
            return self.save_intraday_quotation(block, **specs.load_kwargs)
        elif t == 'prediction':
            return self.save_intraday_prediction(block, **specs.load_kwargs)
        elif t == 'factor':
            return self.save_intraday_factor(block, **specs.load_kwargs)
        else:
            raise ValueError(f"Invalid type: {t}")

    def load_intraday(self, specs: VariableLoadSpec) -> PanelBlockIndexed:
        t = specs.var_type
        if t == 'quotation':
            return self.load_intraday_quotation(**specs.load_kwargs)
        elif t == 'prediction':
            return self.load_intraday_prediction(**specs.load_kwargs)
        elif t == 'factor':
            return self.load_intraday_factor(**specs.load_kwargs)
        else:
            raise ValueError(f"Invalid type: {t}")
    
    def list_available_dates(self, specs: list[VariableLoadSpec]) -> T.List[datetime.date]:
        raise NotImplementedError

    def load_indexed_block_from_src(self, src: str) -> PanelBlockIndexed:
        raise NotImplementedError

    def save_intraday_to_dst(self, block: PanelBlockIndexed, dst: str, overwrite: bool = False) -> None:
        raise NotImplementedError

    def save_intraday_quotation(self, block: PanelBlockIndexed, date: datetime.date | str) -> None:
        raise NotImplementedError
        
    def load_intraday_quotation(self, date: datetime.date | str, instruments: T.Optional[str] = None) -> PanelBlockIndexed:
        raise NotImplementedError

    def save_intraday_prediction(self, block: PanelBlockIndexed, model: str, epoch: int, date: datetime.date | str, use_ema: bool = False) -> None:
        raise NotImplementedError

    def load_intraday_prediction(self, model: str, epoch: int, date: datetime.date | str, use_ema: bool = False) -> PanelBlockIndexed:
        raise NotImplementedError

    def save_intraday_factor(self, block: PanelBlockIndexed, factor: str, date: datetime.date | str) -> None:
        raise NotImplementedError

    def load_intraday_factor(self, factor: str, date: datetime.date | str) -> PanelBlockIndexed:
        raise NotImplementedError

    def load_intraday_prediction_with_quotations(self, model: str, epoch: int, date: datetime.date | str, use_ema: bool = False) -> PanelBlockIndexed:
        raise NotImplementedError
    
    def convert_dataframe_to_block(self, df: pd.DataFrame, src='df') -> PanelBlockIndexed:
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
        return PanelBlockIndexed(df)





