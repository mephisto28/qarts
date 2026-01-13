
import numpy as np
from qarts.core import PanelBlockDense
from ..context import FactorContext, ContextSrc


class BaseOps:

    def __init__(self, context: FactorContext, is_online: bool = False):
        self.context = context
        self.is_online = is_online
        self.current_step = -1

    def get_src(self, src: ContextSrc) -> PanelBlockDense:
        if src in self.context.blocks:
            return self.context.blocks[src]
        else:
            raise ValueError(f"Source {src} not found in context")

    def get_field(self, src: ContextSrc, field: str) -> np.ndarray:
        block = self.get_src(src)
        return block.get_view(field)

    def now(self, field: str, window: int = -1) -> np.ndarray:
        assert ContextSrc.INTRADAY_QUOTATION in self.context.blocks
        block = self.context.blocks[ContextSrc.INTRADAY_QUOTATION]
        if field in block.fields:
            return block.get_current_view(field, window)
        else:
            block = self.context.blocks[ContextSrc.FACTOR_CACHE]
            if field in block.fields:
                return block.get_current_view(field, window)
            else:
                raise ValueError(f"Field {field} not found in factor cache")
            

    def now_factor(self, field: str, window: int = -1) -> np.ndarray:
        assert ContextSrc.FACTOR_CACHE in self.context.blocks
        block = self.context.blocks[ContextSrc.FACTOR_CACHE]
        return block.get_current_view(field, window)