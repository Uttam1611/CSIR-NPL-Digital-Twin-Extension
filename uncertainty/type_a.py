"""
Type A uncertainty: statistical uncertainty derived from repeated measurements
under the same conditions (GUM Section 4.2), estimated with a rolling window.

This module holds in-memory state (the rolling window itself) and is
therefore stateful per-process. It does not talk to the database, MQTT, or
the dashboard — it only tracks numbers it is given via `update()`.
"""
from collections import deque
from dataclasses import dataclass
from statistics import mean, stdev
from threading import Lock
from typing import Deque, Dict


@dataclass
class TypeAResult:
    n: int                      # number of samples currently in the window
    sample_mean: float
    sample_std_dev: float
    standard_uncertainty: float  # s / sqrt(n) -- the Type A standard uncertainty of the mean


class RollingWindowEstimator:
    """
    Maintains the last `window_size` readings for ONE channel and computes
    Type A uncertainty from them.

    Thread safety: inference_server.py serves requests from uvicorn's worker
    threads, so `update()` is guarded by a lock to avoid a torn read/write on
    the underlying deque under concurrent requests.
    """

    def __init__(self, window_size: int = 30):
        if window_size < 2:
            raise ValueError("window_size must be >= 2 to compute a standard deviation")
        self._window_size = window_size
        self._values: Deque[float] = deque(maxlen=window_size)
        self._lock = Lock()

    def update(self, value: float) -> TypeAResult:
        """Append a new reading and return the Type A result computed over the current window."""
        with self._lock:
            self._values.append(value)
            return self._compute_locked()

    def snapshot(self) -> TypeAResult:
        """Return the current Type A result without adding a new reading."""
        with self._lock:
            return self._compute_locked()

    def _compute_locked(self) -> TypeAResult:
        n = len(self._values)
        if n == 0:
            return TypeAResult(n=0, sample_mean=0.0, sample_std_dev=0.0, standard_uncertainty=0.0)
        if n < 2:
            # A standard deviation is undefined for a single sample; report
            # zero uncertainty rather than dividing by zero. The window will
            # start producing a real estimate from the 2nd reading onward.
            return TypeAResult(n=n, sample_mean=self._values[0], sample_std_dev=0.0,
                                standard_uncertainty=0.0)
        m = mean(self._values)
        s = stdev(self._values)
        return TypeAResult(n=n, sample_mean=m, sample_std_dev=s, standard_uncertainty=s / (n ** 0.5))


class MultiChannelTypeAEstimator:
    """Owns one RollingWindowEstimator per named channel (e.g. 'vibration', 'current', ...)."""

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._estimators: Dict[str, RollingWindowEstimator] = {}
        self._registry_lock = Lock()

    def _get_or_create(self, channel: str) -> RollingWindowEstimator:
        with self._registry_lock:
            estimator = self._estimators.get(channel)
            if estimator is None:
                estimator = RollingWindowEstimator(self._window_size)
                self._estimators[channel] = estimator
            return estimator

    def update(self, channel: str, value: float) -> TypeAResult:
        return self._get_or_create(channel).update(value)

    def snapshot(self, channel: str) -> TypeAResult:
        return self._get_or_create(channel).snapshot()
