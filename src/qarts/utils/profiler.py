from __future__ import annotations

import time
import math
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, Iterable, Optional


_Z_95 = 1.96  # normal approx


@dataclass
class _WindowStats:
    window: int
    samples: Deque[float] = field(init=False)
    last: float = 0.0

    def __post_init__(self) -> None:
        self.samples = deque(maxlen=self.window)

    def add(self, x: float) -> None:
        self.last = x
        self.samples.append(x)

    def moving_mean(self) -> float:
        n = len(self.samples)
        return 0.0 if n == 0 else sum(self.samples) / n

    def sample_std(self, mean: float) -> float:
        n = len(self.samples)
        if n < 2:
            return 0.0
        sse = 0.0
        for v in self.samples:
            d = v - mean
            sse += d * d
        return math.sqrt(sse / (n - 1))

    def ci95_half_width(self, mean: float, std: float) -> float:
        n = len(self.samples)
        if n == 0:
            return 0.0
        return _Z_95 * std / math.sqrt(n)


class TimerProfiler:
    """
    Label-based context manager timer.
    Moving average and CI computed over last `window` runs per label.
    """

    class _Section:
        def __init__(self, owner: "TimerProfiler", label: str) -> None:
            self._owner = owner
            self._label = label
            self._t0: Optional[float] = None

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            t1 = time.perf_counter()
            if self._t0 is not None:
                self._owner._record(self._label, t1 - self._t0)
            return False  # do not suppress exceptions

    def __init__(self, *, window: int = 20) -> None:
        self._window = int(window)
        if self._window <= 0:
            raise ValueError("window must be > 0")

        self._lock = threading.Lock()
        self._data: Dict[str, _WindowStats] = {}

    def section(self, label: str) -> "_Section":
        return TimerProfiler._Section(self, label)

    def _get(self, label: str) -> _WindowStats:
        s = self._data.get(label)
        if s is None:
            s = _WindowStats(window=self._window)
            self._data[label] = s
        return s

    def _record(self, label: str, elapsed_s: float) -> None:
        with self._lock:
            self._get(label).add(elapsed_s)

    def report(
        self,
        labels: Optional[Iterable[str]] = None,
        *,
        unit: str = "s",
        precision: int = 3,
        sort_by_last: bool = False,
    ) -> str:
        """
        Output:
          [cost time] - label1: last (ma+/-ci), label2: ...
        where ma is moving average(window), ci is 95% CI half-width.
        """
        scale, suf = (1000.0, "ms") if unit == "ms" else (1.0, "s")

        with self._lock:
            items = list(self._data.items())

        if labels is not None:
            wanted = set(labels)
            items = [(k, v) for k, v in items if k in wanted]

        if sort_by_last:
            items.sort(key=lambda kv: kv[1].last, reverse=True)
        else:
            items.sort(key=lambda kv: kv[0])

        parts = []
        for label, st in items:
            n = len(st.samples)
            if n == 0:
                continue
            ma = st.moving_mean()
            sd = st.sample_std(ma)
            ci = st.ci95_half_width(ma, sd)

            parts.append(
                f"{label}: "
                f"{st.last*scale:.{precision}f}{suf} "
                f"({ma*scale:.{precision}f}+/-{ci*scale:.{precision}f}{suf})"
            )

        if not parts:
            return "[cost time] - <no data>"

        return "[cost time] - " + ", ".join(parts)
