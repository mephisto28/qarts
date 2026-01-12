from .daily_common import DailyOps
from .targets import TargetsOps
from .intraday import IntradayOps

class ContextOps(DailyOps, TargetsOps, IntradayOps):
    pass


__all__ = ['ContextOps']