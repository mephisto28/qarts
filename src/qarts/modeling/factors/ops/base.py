
import numpy as np
from ..context import FactorContext, ContextSrc


class BaseOps:

    def __init__(self, context: FactorContext, is_online: bool = False):
        self.context = context
        self.is_online = is_online # 

    def now(self, field: str, window: int = 1) -> np.ndarray:
        assert ContextSrc.INTRADAY_QUOTATION in self.context.blocks
        block = self.context.blocks[ContextSrc.INTRADAY_QUOTATION]
        return block.get_current_view(field, window)