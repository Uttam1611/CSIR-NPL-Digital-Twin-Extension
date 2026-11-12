"""
Automatic table creation for the dataset-generation pipeline.

`ensure_schema()` runs idempotent DDL (CREATE TABLE/INDEX IF NOT EXISTS) so it
is safe to call every time inference_server.py starts up, rather than
requiring a separate manual migration step. The identical DDL is also kept in
sql/001_create_inference_cycles.sql for teams that prefer to run schema
changes through a dedicated migration tool instead of on app startup.

Schema design notes (why it looks the way it does):

  - Frequently-queried, always-numeric fields (rpm, current_a, voltage_v,
    frequency_hz, temperature_c, vibration_rms_g, fault_class, confidence,
    the three uncertainty numbers) are typed columns, so trend queries and ML
    feature extraction can use normal indexed SQL instead of unpacking JSON.

  - `extra_sensors` (JSONB) exists specifically so that adding a new sensor
    later (e.g. the PZEM-004T power-quality module mentioned in the report's
    Future Work) does NOT require a schema migration -- it just becomes a new
    key in this JSON column until/unless it earns its own typed column
    because it's queried often enough to be worth indexing.

  - `uncertainty_detail` (JSONB) stores the full per-channel breakdown
    (Type A, Type B, combined, expanded, sample count) for every channel that
    was available on that cycle, while `type_a_uncertainty` /
    `type_b_uncertainty` / `combined_uncertainty` typed columns hold the
    *vibration* channel's numbers specifically, since vibration is the signal
    actually feeding the fault classifier.

  - `ohim`, `rul_hours`, `health_score` are nullable placeholder columns per
    the task's explicit instruction to reserve space for future work; nothing
    in the current codebase computes any of the three, so they are always
    NULL until a future module populates them. Nothing here invents a
    computation for them.

  - `feature_matrix_bytes` (BYTEA) is OFF by default (see repository.py /
    `DT_STORE_FEATURE_MATRIX`) because storing the full (17, 769) float32
    STFT+envelope feature matrix on every single cycle is ~52 KB/row, which
    adds up fast at a 1-second telemetry interval. `preprocessed_features`
    (JSONB) instead stores lightweight summary statistics of that matrix by
    default; the raw matrix can be turned on for research use when disk isn't
    a constraint.
"""
import logging

from .connection import PostgresConnectionManager

logger = logging.getLogger("digital_twin.storage")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inference_cycles (
    id                      BIGSERIAL PRIMARY KEY,
    cycle_uuid              UUID NOT NULL,
    recorded_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

    rpm                      DOUBLE PRECISION,
    current_a                DOUBLE PRECISION,
    voltage_v                DOUBLE PRECISION,
    frequency_hz             DOUBLE PRECISION,
    temperature_c            DOUBLE PRECISION,
    vibration_rms_g          DOUBLE PRECISION,

    fault_class              TEXT,
    confidence                DOUBLE PRECISION,
    model_probabilities       JSONB,

    ohim                      DOUBLE PRECISION,
    rul_hours                 DOUBLE PRECISION,
    health_score              DOUBLE PRECISION,

    type_a_uncertainty        DOUBLE PRECISION,
    type_b_uncertainty        DOUBLE PRECISION,
    combined_uncertainty      DOUBLE PRECISION,
    uncertainty_detail        JSONB,

    extra_sensors             JSONB,
    preprocessed_features     JSONB,
    feature_matrix_shape      SMALLINT[],
    feature_matrix_bytes      BYTEA,

    inference_sr_hz           INTEGER,
    target_sr_hz              INTEGER,
    window_size               INTEGER,

    created_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_inference_cycles_recorded_at ON inference_cycles (recorded_at);",
    "CREATE INDEX IF NOT EXISTS idx_inference_cycles_fault_class ON inference_cycles (fault_class);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_inference_cycles_cycle_uuid ON inference_cycles (cycle_uuid);",
    "CREATE INDEX IF NOT EXISTS idx_inference_cycles_extra_sensors ON inference_cycles USING GIN (extra_sensors);",
]


def ensure_schema(conn_manager: PostgresConnectionManager) -> None:
    with conn_manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            for stmt in CREATE_INDEXES_SQL:
                cur.execute(stmt)
        conn.commit()
    logger.info("PostgreSQL schema verified/created (inference_cycles).")
