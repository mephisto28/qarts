import typing as T
from dataclasses import dataclass, field

import numpy as np
from qarts.core import PanelBlockDense


@dataclass
class FactorContext:
    n_inst: int
    inst_categories: np.ndarray
    blocks: T.Dict[str, PanelBlockDense]
    intermediate_cache: T.Dict[str, np.ndarray] = field(default_factory=dict)
    factor_cache: T.Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Factor:
    factor_name: str
    window: int
    input_fields: list[str]
    extra_params: T.Dict[str, any] = field(default_factory=dict)

    @property
    def name(self):
        return self.factor_spec.name

    def compute_from_context(self, context: FactorContext, out: np.ndarray):
        raise NotImplementedError
    