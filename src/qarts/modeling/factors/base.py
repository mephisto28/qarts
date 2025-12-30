import abc
import functools
import typing as T
from dataclasses import dataclass, field

import numpy as np
from qarts.modeling.factors.context import FactorContext

@dataclass
class FactorSpec:
    name: str
    input_fields: dict[str, list[str]]
    window: int = 1
    params: dict = field(default_factory=dict)


class Factor(metaclass=abc.ABCMeta):

    def __init__(self, input_fields: dict[str, list[str]], window: int = 1, **kwargs):
        self.window = window
        self.input_fields = input_fields
        self.params = kwargs
        self._sources = list(self.input_fields.keys())
        self._check_inputs_valid()

    def _check_inputs_valid(self):
        pass

    @property
    def name(self) -> str:
        return self.params.get('name', self.__class__.__name__)

    @abc.abstractmethod
    def compute_from_context(self, context: FactorContext, out: np.ndarray):
        raise NotImplementedError

    @property
    def sources(self) -> list[str]:
        return self._sources


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
    return get_factor(spec.name, input_fields=spec.input_fields, window=spec.window, **spec.params)

