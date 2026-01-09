import functools
import typing as T

from torch import nn

_model_registry: T.Dict[str, type['nn.Module']] = {}


def register_model(name: str):
    @functools.wraps(name)
    def wrapper(model_class: type['nn.Module']) -> type['nn.Module']:
        _model_registry[name] = model_class
        return model_class
    return wrapper


def get_model(name: str):
    return _model_registry[name]