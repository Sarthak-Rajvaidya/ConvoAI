"""
utils/timing.py
===============================================================================
Lightweight stage-timing helper used to build the performance metrics shown
in the UI ("Processing completed in 38.4s") and logged via utils.logging.

Usage:

    timer = StageTimer()
    with timer.stage("transcription"):
        ...
    timer.as_dict()  # {"transcription": 14.2, "total": 14.2}
===============================================================================
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    durations: dict[str, float] = field(default_factory=dict)
    _start: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = round(time.perf_counter() - started, 3)

    @property
    def total(self) -> float:
        return round(time.perf_counter() - self._start, 3)

    def as_dict(self) -> dict[str, float]:
        return {**self.durations, "total": self.total}
