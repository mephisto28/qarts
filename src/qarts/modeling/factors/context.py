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


@dataclass
class FactorContext:
    n_inst: int
    inst_categories: np.ndarray
    blocks: T.Dict[str, PanelBlockDense] = field(default_factory=dict)
    intermediate_cache: T.Dict[str, np.ndarray] = field(default_factory=dict)
    current_cursor: int = -1

    def register_block(self, src: ContextSrc, block: PanelBlockDense):
        assert src in (ContextSrc.DAILY_QUOTATION, ContextSrc.INTRADAY_QUOTATION, ContextSrc.FACTOR_CACHE), f"Invalid source: {src}"
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
    def from_daily_block(cls, daily_block: PanelBlockDense | PanelBlockIndexed) -> 'FactorContext':
        if isinstance(daily_block, PanelBlockIndexed):
            columns = list(daily_block.data.columns)
            daily_block = PanelBlockDense.from_indexed_block(
                daily_block,
                required_columns=columns,
                fill_methods=[1 for _ in columns],
                frequency='1D',
            )
        return cls(
            n_inst=len(daily_block.instruments),
            inst_categories=daily_block.instruments,
            blocks={ContextSrc.DAILY_QUOTATION: daily_block}
        )
