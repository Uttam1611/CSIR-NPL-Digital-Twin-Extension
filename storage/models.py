"""
InferenceCycleRecord — one row of the inference_cycles table.

Kept as a plain dataclass (not a psycopg2-specific or FastAPI-specific type)
so it can be constructed and passed around independently of both.
"""
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InferenceCycleRecord:
    cycle_uuid: uuid.UUID = field(default_factory=uuid.uuid4)

    # Core telemetry -- all optional, since inference_server.py's /predict
    # currently only guarantees a vibration `window`; rpm/current/voltage/
    # frequency/temperature are only present if the caller supplies them
    # (see the extended PredictRequest in inference_server.py).
    rpm: Optional[float] = None
    current_a: Optional[float] = None
    voltage_v: Optional[float] = None
    frequency_hz: Optional[float] = None
    temperature_c: Optional[float] = None
    vibration_rms_g: Optional[float] = None

    # Prediction
    fault_class: Optional[str] = None
    confidence: Optional[float] = None
    model_probabilities: Optional[Dict[str, float]] = None

    # Future placeholders -- always None until a future module computes them
    ohim: Optional[float] = None
    rul_hours: Optional[float] = None
    health_score: Optional[float] = None

    # Uncertainty (vibration channel, since that's the model's input signal)
    type_a_uncertainty: Optional[float] = None
    type_b_uncertainty: Optional[float] = None
    combined_uncertainty: Optional[float] = None
    uncertainty_detail: Optional[Dict[str, Any]] = None  # full per-channel breakdown

    # Extensibility for sensors/features not foreseen at schema-design time
    extra_sensors: Optional[Dict[str, Any]] = None
    preprocessed_features: Optional[Dict[str, Any]] = None
    feature_matrix_shape: Optional[List[int]] = None
    feature_matrix_bytes: Optional[bytes] = None

    inference_sr_hz: Optional[int] = None
    target_sr_hz: Optional[int] = None
    window_size: Optional[int] = None
