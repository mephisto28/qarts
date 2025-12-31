import inspect
import functools
import typing as T
from enum import Enum, auto
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from qarts.core import PanelBlockDense
from qarts.core.panel import PanelBlockIndexed


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
    factor_cache: T.Dict[str, np.ndarray] = field(default_factory=dict)
    current_cursor: int = -1

    def register_block(self, src: ContextSrc, block: PanelBlockDense):
        assert src in (ContextSrc.DAILY_QUOTATION, ContextSrc.INTRADAY_QUOTATION), f"Invalid source: {src}"
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
        self.intermediate_cache[key] = (scope, value)

    def get_field(self, src: ContextSrc, field: str) -> np.ndarray:
        if src == ContextSrc.FACTOR_CACHE:
            return self.factor_cache[field]
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


def query_or_set_history_cache(func):
    sig = inspect.signature(func)
    default_name = sig.parameters['name'].default

    @functools.wraps(func)
    def wrapper(self: 'ContextOps', field: T.Union[str, T.Tuple[str, ...]], name: str = default_name, **kwargs):
        if isinstance(field, str):
            field_key = (field,)
        else:
            field_key = field
        key = (name, *field_key, *kwargs.values())
        if key in self.context.intermediate_cache:
            return self.context.get_cache(key)
        else:
            values = func(self, field, name, **kwargs)
            cache = self.context.get_cache(key, shape=values.shape, dtype=np.float32, scope=None)
            cache[:] = values
            return values
    return wrapper


def expand_tdim_on_batch(func):
    @functools.wraps(func)
    def wrapper(self: 'ContextOps', *args, **kwargs):
        values = func(self, *args, **kwargs)
        if self.is_online:
            return values
        else:
            return values[..., np.newaxis]
    return wrapper


class ContextOps:

    def __init__(self, context: FactorContext, is_online: bool = False):
        self.context = context
        self.is_online = is_online # 
    
    def now(self, field: str, window: int = 1) -> np.ndarray:
        assert ContextSrc.INTRADAY_QUOTATION in self.context.blocks
        block = self.context.blocks[ContextSrc.INTRADAY_QUOTATION]
        return block.get_current_view(field, window)
    
    @expand_tdim_on_batch
    def yesterday(self, field: str) -> np.ndarray:
        assert ContextSrc.DAILY_QUOTATION in self.context.blocks
        block = self.context.blocks[ContextSrc.DAILY_QUOTATION]
        return block.get_view(field)[:, -1]

    @query_or_set_history_cache
    def _history_prefix_sum(self, field: str, name: str = 'history_prefix_sum') -> np.ndarray:
        values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field)
        return np.nancumsum(values, axis=1)

    @query_or_set_history_cache
    def _history_weighted_prefix_sum(self, fields: T.Tuple[str, ...], name: str = 'history_weighted_prefix_sum') -> np.ndarray:
        value1 = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=fields[0])
        value2 = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=fields[1])
        return np.nancumsum(value1 * value2, axis=1)

    @query_or_set_history_cache
    def _history_prefix_count(self, field: str, name: str = 'history_prefix_count') -> np.ndarray:
        values = self.context.get_field(src=ContextSrc.DAILY_QUOTATION, field=field)
        return np.nancumsum((~np.isnan(values)).astype(np.float32), axis=1)
    
    @expand_tdim_on_batch
    @query_or_set_history_cache
    def history_ma(self, field: str, name: str = 'history_ma', window: int = 20) -> np.ndarray:
        prefix_sum = self._history_prefix_sum(field)
        prefix_count = self._history_prefix_count(field)
        ma = (prefix_sum[:, -1] - prefix_sum[:, -window]) / (prefix_count[:, -1] - prefix_count[:, -window])
        return ma
    
    @expand_tdim_on_batch
    @query_or_set_history_cache
    def history_valid_ratio(self, field: str, name: str = 'history_valid_ratio', window: int = 20) -> np.ndarray:
        prefix_count = self._history_prefix_count(field)
        valid_ratio = (prefix_count[:, -1] - prefix_count[:, -window]) / window
        return valid_ratio
    
    @expand_tdim_on_batch
    @query_or_set_history_cache
    def hisotry_vwap(self, fields: T.Tuple[str, ...], name: str = 'history_vwap', window: int = 20) -> np.ndarray:
        price_field, volume_field = fields
        weighted_prefix_sum = self._history_weighted_prefix_sum((price_field, volume_field))
        volume_prefix_sum = self._history_prefix_sum(volume_field)
        weighted_sum = weighted_prefix_sum[:, -1] - weighted_prefix_sum[:, -window]
        normalizer = volume_prefix_sum[:, -1] - volume_prefix_sum[:, -window]
        return weighted_sum / normalizer
        
    