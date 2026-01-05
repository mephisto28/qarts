from .daily_common import DailyOps
from .targets import TargetsOps


class ContextOps(DailyOps, TargetsOps):
    pass


__all__ = ['ContextOps']