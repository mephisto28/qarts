import abc
import functools
import typing as T
from dataclasses import dataclass, field

import numpy as np
from qarts.modeling.factors.ops import ContextOps
from qarts.modeling.factors.context import ContextSrc


@dataclass
class FactorSpec:
    name: str
    input_fields: dict[str, list[str]]
    window: int = 1
    params: dict = field(default_factory=dict)
    need_cache: bool = False


class Factor(metaclass=abc.ABCMeta):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, scale: float = 1.0, shift: float = 0.0, need_cache: bool = False, lower: float = -5, upper: float = 5, **kwargs):
        self.window = window
        self.input_fields = input_fields
        self.scale = scale
        self.shift = shift
        self.need_cache = need_cache
        self.lower = lower
        self.upper = upper
        self.params = kwargs
        self._sources = list(self.input_fields.keys())
        self._check_inputs_valid()

    def _check_inputs_valid(self):
        pass

    @property
    def name(self) -> str:
        return self.params.get('name', self.__class__.__name__)

    @abc.abstractmethod
    def compute_from_context(self, context: ContextOps, out: np.ndarray):
        raise NotImplementedError

    @property
    def sources(self) -> list[str]:
        return self._sources
    

@dataclass
class DailyStatsSpec:
    name: str
    input_fields: dict[str, list[str]]
    params: dict = field(default_factory=dict)


class DailyStats(metaclass=abc.ABCMeta):
    
    def __init__(self, input_fields: dict[str, list[str]], **kwargs):
        self.input_fields = input_fields
        self._sources = list(self.input_fields.keys())

    @abc.abstractmethod
    def compute_from_context(self, context: ContextOps):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.params.get('name', self.__class__.__name__)

    @property
    def output_fields(self) -> list[str]:
        raise NotImplementedError

    @property
    def num_output_fields(self) -> int:
        return len(self.output_fields)
    
    @property
    def sources(self) -> list[str]:
        return self._sources
    

class FactorFromDailyAndIntraday(Factor):

    num_daily_fields: int = 0
    num_intraday_fields: int = 0
    num_factor_cache_fields: int = 0

    def _check_inputs_valid(self):
        if self.num_daily_fields > 0:
            assert len(self.input_fields[ContextSrc.DAILY_QUOTATION]) == self.num_daily_fields, \
                f"Expected  daily fields {self.num_daily_fields}!={len(self.input_fields[ContextSrc.DAILY_QUOTATION])}(actual)"
                
        if self.num_intraday_fields > 0:
            assert len(self.input_fields[ContextSrc.INTRADAY_QUOTATION]) == self.num_intraday_fields, \
                f"Expected intraday fields {self.num_intraday_fields}!={len(self.input_fields[ContextSrc.INTRADAY_QUOTATION])}(actual)"
        
        if self.num_factor_cache_fields > 0:
            assert len(self.input_fields[ContextSrc.FACTOR_CACHE]) == self.num_factor_cache_fields, \
                f"Expected factor cache fields {self.num_factor_cache_fields}!={len(self.input_fields[ContextSrc.FACTOR_CACHE])}(actual)"


_factors_registry: T.Dict[str, type['Factor']] = {}
_stats_registry: T.Dict[str, type['DailyStats']] = {}

def register_factor(name: str):
    def wrapper(factor_class: type['Factor']) -> type['Factor']:
        _factors_registry[name] = factor_class
        return factor_class
    return wrapper

@functools.singledispatch
def get_factor(factor, *args, **kwargs) -> Factor:
    raise TypeError(f"Invalid factor type: {type(factor)}")

@get_factor.register
def _(factor_name: str, input_fields: dict[str, list[str]], window: int = 1, **kwargs) -> Factor:
    F = _factors_registry[factor_name]
    # print(f"get_factor: {factor_name}, {input_fields}, {window}, {kwargs}")
    return F(input_fields=input_fields, window=window, **kwargs)

@get_factor.register
def _(spec: FactorSpec) -> Factor:
    return get_factor(spec.name, input_fields=spec.input_fields, window=spec.window, need_cache=spec.need_cache, **spec.params)


def register_stats(name: str):
    def wrapper(stats_class: type['DailyStats']) -> type['DailyStats']:
        _stats_registry[name] = stats_class
        return stats_class
    return wrapper

@functools.singledispatch
def get_stats(stats, *args, **kwargs) -> DailyStats:
    raise TypeError(f"Invalid stats type: {type(stats)}")

@get_stats.register
def _(stats_name: str, input_fields: dict[str, list[str]], **kwargs) -> DailyStats:
    S = _stats_registry[stats_name]
    return S(input_fields=input_fields, **kwargs)

@get_stats.register
def _(spec: DailyStatsSpec) -> DailyStats:
    return get_stats(spec.name, input_fields=spec.input_fields, **spec.params)