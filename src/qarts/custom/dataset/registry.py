import functools

import torch
import numpy as np


_datasets = {}


def register_dataset(name: str):
    def wrapper(class_name: type):
        _datasets[name] = class_name
        return class_name
    return wrapper


def get_dataset(name: str) -> type:
    if name not in _datasets:
        raise ValueError(f"Dataset {name} not found")
    return _datasets[name]


def get_collate_fn(name: str, **params) -> callable:
    if name == 'sequence':
        return collate_sequence
    elif name == 'sample':
        return functools.partial(collate_sample_intraday, **params)
    else:
        raise ValueError(f"Collate function {name} not found")


def collate_sequence(batch):
    assert len(batch) == 1, "Batch size must be 1"
    X, y = batch[0]
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("NaN values found in batch")
    return torch.tensor(X), torch.tensor(y)


def collate_sample_intraday(batch, period: tuple[int, int], sample_num: int = 1):
    assert len(batch) == 1, "Batch size must be 1"
    batch = batch[0]
    X = batch["features"] # F, N, T
    y = batch["targets"] # F, N, T
    is_valid_instruments = batch["is_valid_instruments"]
    X = X[:, is_valid_instruments]
    y = y[:, is_valid_instruments]
    F, N, T = X.shape
    t0, t1 = period
    t_idx = np.random.randint(t0, t1, size=N)
    X = X[:, np.arange(N), t_idx].T
    y = y[:, np.arange(N), t_idx].T
    if 'selector' in batch:
        selector = batch['selector']
        selector = selector[is_valid_instruments]
        selector = selector[np.arange(N), t_idx]
        X = X[selector]
        y = y[selector]
    return {
        'features': torch.tensor(X), 
        'targets': torch.tensor(y), 
        'timesteps': torch.tensor(t_idx),
        'instruments': batch['instruments'][is_valid_instruments],    
        'feature_names': batch['feature_names'],
        'target_names': batch['target_names']
    }


def get_fill_method(c: str):
    if c in ('volume', 'turnover', 'amount', 'daily_return'):
        return 0
    elif c.endswith('diff'):
        return 0
    return 1