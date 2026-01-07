import functools
from dataclasses import dataclass, field

from torch import nn


_loss_fn = {
    'mae': nn.L1Loss,
    'mse': nn.MSELoss,
}

_metrics = {}


@dataclass
class Schema:
    name: str
    fields: list[str]
    weight: float
    loss: list[dict]
    metrics: list[dict]


def register_loss_fn(name: str):
    @functools.wraps(name)
    def wrapper(cls):
        _loss_fn[name] = cls
        return cls
    return wrapper


def get_loss_fn(name: str):
    if name not in _loss_fn:
        raise ValueError(f"Loss function {name} not found")
    return _loss_fn[name]


def register_metric(name: str):
    @functools.wraps(name)
    def wrapper(cls):
        _metrics[name] = cls
        return cls
    return wrapper


def get_metric_fn(name: str):
    if name not in _metrics:
        raise ValueError(f"Metric {name} not found")
    return _metrics[name]