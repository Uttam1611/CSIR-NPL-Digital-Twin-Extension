"""
uncertainty/ — Measurement uncertainty estimation, following metrology (GUM) principles.

This package is deliberately independent of FastAPI, MQTT, PostgreSQL, and the
dashboard. It only knows about numbers in, numbers out, so it can be reused
from inference_server.py, from a future edge-inference build on the Pi
(Section 8.3 of the project report), or from an offline analysis notebook,
without dragging in any of those dependencies.

Public entry point: `UncertaintyEngine`, which wraps:
  - type_a.MultiChannelTypeAEstimator  (statistical uncertainty, rolling window)
  - type_b.load_channel_specs          (instrument/sensor uncertainty, configurable)
  - combiner.combine                   (GUM combination -> combined + expanded uncertainty)

Terminology note (per the project's own requirement): this package computes
*measurement* uncertainty of sensor readings. It has nothing to do with, and
must not be confused with, the fault-classifier's `confidence` value returned
by the meta-learner in inference_server.py — that is model confidence, not
measurement uncertainty.
"""
from dataclasses import asdict
from typing import Dict, Optional

from .type_a import MultiChannelTypeAEstimator, TypeAResult
from .type_b import ChannelSpec, load_channel_specs
from .combiner import CombinedUncertainty, combine

__all__ = [
    "UncertaintyEngine",
    "MultiChannelTypeAEstimator",
    "TypeAResult",
    "ChannelSpec",
    "load_channel_specs",
    "CombinedUncertainty",
    "combine",
]


class UncertaintyEngine:
    """
    Convenience façade: given a dict of {channel_name: latest_reading}, returns
    a dict of {channel_name: CombinedUncertainty} using a persistent rolling
    window per channel for Type A and the configured instrument spec for Type B.
    """

    def __init__(self, window_size: int = 30, coverage_factor: float = 2.0,
                 channel_specs: Optional[Dict[str, ChannelSpec]] = None):
        self._type_a = MultiChannelTypeAEstimator(window_size=window_size)
        self._specs = channel_specs if channel_specs is not None else load_channel_specs()
        self._coverage_factor = coverage_factor

    def evaluate(self, readings: Dict[str, float]) -> Dict[str, CombinedUncertainty]:
        results: Dict[str, CombinedUncertainty] = {}
        for channel, value in readings.items():
            if value is None:
                continue
            type_a_result = self._type_a.update(channel, float(value))
            spec = self._specs.get(channel)
            type_b_u = spec.standard_uncertainty() if spec is not None else 0.0
            results[channel] = combine(channel, type_a_result, type_b_u, self._coverage_factor)
        return results

    @staticmethod
    def as_dict(results: Dict[str, CombinedUncertainty]) -> Dict[str, dict]:
        return {channel: asdict(result) for channel, result in results.items()}
