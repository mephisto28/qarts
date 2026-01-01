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

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, scale: float = 1.0, shift: float = 0.0, need_cache: bool = False, **kwargs):
        self.window = window
        self.input_fields = input_fields
        self.scale = scale
        self.shift = shift
        self.need_cache = need_cache
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


class FactorFromDailyAndIntraday(Factor):

    num_daily_fields: int = 0
    num_intraday_fields: int = 0
    num_factor_cache_fields: int = 0

    def _check_inputs_valid(self):
        if self.num_daily_fields > 0:
            assert len(self.input_fields[ContextSrc.DAILY_QUOTATION]) == self.num_daily_fields
        else:
            if ContextSrc.DAILY_QUOTATION in self.input_fields:
                assert len(self.input_fields[ContextSrc.DAILY_QUOTATION]) == 0
                
        if self.num_intraday_fields > 0:
            assert len(self.input_fields[ContextSrc.INTRADAY_QUOTATION]) == self.num_intraday_fields
        else:
            if ContextSrc.INTRADAY_QUOTATION in self.input_fields:
                assert len(self.input_fields[ContextSrc.INTRADAY_QUOTATION]) == 0
        
        if self.num_factor_cache_fields > 0:
            assert len(self.input_fields[ContextSrc.FACTOR_CACHE]) == self.num_factor_cache_fields
        else:
            if ContextSrc.FACTOR_CACHE in self.input_fields:
                assert len(self.input_fields[ContextSrc.FACTOR_CACHE]) == 0


_factors_registry: T.Dict[str, type['Factor']] = {}


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
    print(f"get_factor: {factor_name}, {input_fields}, {window}, {kwargs}")
    return F(input_fields=input_fields, window=window, **kwargs)

@get_factor.register
def _(spec: FactorSpec) -> Factor:
    return get_factor(spec.name, input_fields=spec.input_fields, window=spec.window, need_cache=spec.need_cache, **spec.params)

