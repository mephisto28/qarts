import inspect
import functools
import typing as T
from enum import Enum, auto
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from qarts.core import PanelBlockDense
from qarts.core.panel import PanelBlockIndexed
from . import kernels as kns


class ContextSrc(Enum):
    INTRADAY_QUOTATION = auto()
    DAILY_QUOTATION = auto()
    FACTOR_CACHE = auto()

    FUTURE_DAILY_QUOTATION = auto() # for targets label / IC test


@dataclass
class FactorContext:
    n_inst: int
    inst_categories: np.ndarray
    blocks: T.Dict[str, PanelBlockDense] = field(default_factory=dict)
    intermediate_cache: T.Dict[str, np.ndarray] = field(default_factory=dict)
    current_cursor: int = -1

    def register_block(self, src: ContextSrc, block: PanelBlockDense):
        assert src in (ContextSrc.DAILY_QUOTATION, ContextSrc.INTRADAY_QUOTATION, ContextSrc.FACTOR_CACHE, ContextSrc.FUTURE_DAILY_QUOTATION), f"Invalid source: {src}"
        self.blocks[src] = block

    def register_cache(self, key, shape, dtype=np.float32, scope=None):
        # scope 标识来源，避免不同因子注册缓存冲突
        if key not in self.intermediate_cache:
            self.intermediate_cache[key] = (scope, np.empty(shape, dtype=dtype))
        scope_, buffer = self.intermediate_cache[key]
        if scope != scope_:
            logger.error(f"Conflict: {key} is already registered by {scope_}, cannot register by {scope}")

    def get_cache(self, key, scope=None, shape=None, dtype=np.float32):
        if key not in self.intermediate_cache:
            if shape is not None:
                self.register_cache(key, shape, dtype, scope)
            else:
                raise KeyError(f"Cache {key} not found, available keys: {list(self.intermediate_cache.keys())}")
        scope_, buffer = self.intermediate_cache[key]
        if scope is not None and scope != scope_:
            logger.error(f"Cache {key}(scope={scope_}) is not registered by {scope}, cannot get cache")
        return buffer

    def set_cache(self, key, value, scope=None):
        if key in self.intermediate_cache:
            scope_, buffer = self.intermediate_cache[key]
            if scope != scope_:
                logger.error(f"Conflict: {key} is already registered by {scope_}, cannot set cache by {scope}")
            buffer[:] = value
        else:
            buffer = value
        self.intermediate_cache[key] = (scope, value)

    def get_field(self, src: ContextSrc, field: str) -> np.ndarray:
        if not src in self.blocks:
            raise ValueError(f"Invalid source: {src}, available sources: {list(self.blocks.keys())}")
        block = self.blocks[src]
        return block.get_view(field)

    @classmethod
    def from_daily_block(cls, daily_block: PanelBlockDense | PanelBlockIndexed, is_future: bool = False) -> 'FactorContext':
        if isinstance(daily_block, PanelBlockIndexed):
            columns = list(daily_block.data.columns)
            daily_block = PanelBlockDense.from_indexed_block(
                daily_block,
                required_columns=columns,
                fill_methods=[get_fill_method(c) for c in columns],
                frequency='1D',
            )
        src = ContextSrc.FUTURE_DAILY_QUOTATION if is_future else ContextSrc.DAILY_QUOTATION
        return cls(
            n_inst=len(daily_block.instruments),
            inst_categories=daily_block.instruments,
            blocks={src: daily_block}
        )


def create_mock_context(size=10, seed=42):
    from qarts.utils.random_walk import simulate_random_walk, simulate_noisy_random_walk
    instruments = np.array([f'mk{i:06}' for i in range(size)])
    daily_len = 252
    dates = np.arange(daily_len + 1)

    daily_prices = close = simulate_random_walk(s0=100, mu=0.00, sigma=0.02, n_steps=daily_len, n_paths=size, seed=seed)
    open = daily_prices * (1 + np.random.uniform(-0.01, 0.01, size=(size, daily_len + 1)))
    high = daily_prices * (1 + np.random.uniform(0, 0.03, size=(size, daily_len + 1)))
    low = daily_prices * (1 - np.random.uniform(0, 0.03, size=(size, daily_len + 1)))
    volume = np.exp(np.random.normal(loc=12, scale=1, size=(size, daily_len + 1)))
    daily_return = np.concatenate([np.zeros((size, 1)), np.log(daily_prices[:, 1:] / daily_prices[:, :-1])], axis=1)
    block_data = np.stack([open, high, low, close, volume, daily_return], axis=0)
    fields = ['adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'volume', 'daily_return']
    
    intraday_len = 240
    minutes = np.arange(intraday_len + 1)
    intraday_prices = simulate_noisy_random_walk(s0=daily_prices[:, -1], mu=0.00, sigma=0.001, n_steps=intraday_len, n_paths=size, seed=seed)
    intraday_mom = np.log(intraday_prices / intraday_prices[:, 0:1])
    daily_block = PanelBlockDense(
        instruments=instruments,
        timestamps=dates,
        data=block_data,
        fields=fields,
        frequency='1D'
    )
    intraday_block = PanelBlockDense(
        instruments=instruments,
        timestamps=minutes,
        data=np.stack([intraday_prices], axis=0),
        fields=['mid_price', ],
        frequency='1min'
    )
    vacancy = np.zeros((10, len(instruments), len(minutes)), dtype=np.float32)
    vacancy_fields = ['vacancy_' + str(i) for i in range(10)]
    factor_cache_block = PanelBlockDense(
        instruments=instruments,
        timestamps=minutes,
        data=np.concatenate([intraday_mom[None, ...], vacancy], axis=0),
        fields=['intraday_mom'] + vacancy_fields,
        frequency='1min'
    )

    context = FactorContext.from_daily_block(daily_block)
    context.register_block(ContextSrc.INTRADAY_QUOTATION, intraday_block)
    context.register_block(ContextSrc.FACTOR_CACHE, factor_cache_block)
    return context


def get_fill_method(c: str):
    if c in ('volume', 'turnover', 'amount', 'daily_return'):
        return 0
    elif c.endswith('diff'):
        return 0
    return 1