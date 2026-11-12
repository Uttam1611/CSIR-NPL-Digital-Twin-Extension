"""
InferenceCycleRepository — writes InferenceCycleRecord rows to PostgreSQL.

Every insert runs inside an explicit transaction: commit on success, rollback
on any failure, so a partially-written row can never land in the table.

A storage failure (DB down, network blip that survived the connection
manager's retries, etc.) is logged and returns False rather than raising --
losing one row of historical logging should not be able to break the live
/predict response path that the dashboard and control loop depend on.
"""
import gzip
import logging
from typing import Optional

from psycopg2.extras import Json

from .connection import PostgresConnectionManager
from .models import InferenceCycleRecord

logger = logging.getLogger("digital_twin.storage")

INSERT_SQL = """
INSERT INTO inference_cycles (
    cycle_uuid, rpm, current_a, voltage_v, frequency_hz, temperature_c, vibration_rms_g,
    fault_class, confidence, model_probabilities,
    ohim, rul_hours, health_score,
    type_a_uncertainty, type_b_uncertainty, combined_uncertainty, uncertainty_detail,
    extra_sensors, preprocessed_features, feature_matrix_shape, feature_matrix_bytes,
    inference_sr_hz, target_sr_hz, window_size
) VALUES (
    %(cycle_uuid)s, %(rpm)s, %(current_a)s, %(voltage_v)s, %(frequency_hz)s, %(temperature_c)s, %(vibration_rms_g)s,
    %(fault_class)s, %(confidence)s, %(model_probabilities)s,
    %(ohim)s, %(rul_hours)s, %(health_score)s,
    %(type_a_uncertainty)s, %(type_b_uncertainty)s, %(combined_uncertainty)s, %(uncertainty_detail)s,
    %(extra_sensors)s, %(preprocessed_features)s, %(feature_matrix_shape)s, %(feature_matrix_bytes)s,
    %(inference_sr_hz)s, %(target_sr_hz)s, %(window_size)s
)
"""


class InferenceCycleRepository:
    def __init__(self, conn_manager: PostgresConnectionManager, store_feature_matrix: bool = False):
        self._conn_manager = conn_manager
        self._store_feature_matrix = store_feature_matrix

    def insert(self, record: InferenceCycleRecord) -> bool:
        params = {
            "cycle_uuid": str(record.cycle_uuid),
            "rpm": record.rpm,
            "current_a": record.current_a,
            "voltage_v": record.voltage_v,
            "frequency_hz": record.frequency_hz,
            "temperature_c": record.temperature_c,
            "vibration_rms_g": record.vibration_rms_g,
            "fault_class": record.fault_class,
            "confidence": record.confidence,
            "model_probabilities": Json(record.model_probabilities) if record.model_probabilities is not None else None,
            "ohim": record.ohim,
            "rul_hours": record.rul_hours,
            "health_score": record.health_score,
            "type_a_uncertainty": record.type_a_uncertainty,
            "type_b_uncertainty": record.type_b_uncertainty,
            "combined_uncertainty": record.combined_uncertainty,
            "uncertainty_detail": Json(record.uncertainty_detail) if record.uncertainty_detail is not None else None,
            "extra_sensors": Json(record.extra_sensors) if record.extra_sensors is not None else None,
            "preprocessed_features": Json(record.preprocessed_features) if record.preprocessed_features is not None else None,
            "feature_matrix_shape": list(record.feature_matrix_shape) if record.feature_matrix_shape is not None else None,
            "feature_matrix_bytes": self._encode_feature_matrix(record.feature_matrix_bytes),
            "inference_sr_hz": record.inference_sr_hz,
            "target_sr_hz": record.target_sr_hz,
            "window_size": record.window_size,
        }
        try:
            with self._conn_manager.get_connection() as conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(INSERT_SQL, params)
                    conn.commit()
                    return True
                except Exception:
                    conn.rollback()
                    raise
        except Exception as e:
            logger.error("Failed to persist inference cycle %s: %s", record.cycle_uuid, e)
            return False

    def _encode_feature_matrix(self, raw: Optional[bytes]) -> Optional[bytes]:
        if not self._store_feature_matrix or raw is None:
            return None
        return gzip.compress(raw)
