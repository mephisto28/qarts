from dataclasses import dataclass
import pandas as pd

__all__ = ['CrossSectionalDataIndexed', 'CrossSectionalSeriesIndexed', 'CrossSectionalBlockIndexed']


@dataclass
class CrossSectionalDataIndexed:

    data: pd.DataFrame | pd.Series
    timestamp: pd.Timestamp
    
    def __post_init__(self):
        assert self.data.index.name == 'instrument'

    @property
    def instruments(self) -> pd.Index:
        return self.data.index


@dataclass
class CrossSectionalSeriesIndexed(CrossSectionalDataIndexed):
    data: pd.Series
    
    @staticmethod
    def from_series(s: pd.Series, timestamp: pd.Timestamp) -> 'CrossSectionalSeriesIndexed':
        return CrossSectionalSeriesIndexed(s, timestamp=timestamp)


@dataclass
class CrossSectionalBlockIndexed(CrossSectionalDataIndexed):
    data: pd.DataFrame

    @staticmethod
    def from_dataframe(df: pd.DataFrame, timestamp: pd.Timestamp) -> 'CrossSectionalBlockIndexed':
        return CrossSectionalBlockIndexed(df, timestamp=timestamp)

    