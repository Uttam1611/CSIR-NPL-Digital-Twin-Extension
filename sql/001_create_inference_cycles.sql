-- Manual migration equivalent of storage/schema.py's ensure_schema().
-- Safe to run more than once (IF NOT EXISTS everywhere).
-- Use this if you'd rather run migrations through a dedicated tool
-- (alembic, sqitch, flyway, etc.) instead of on inference_server.py startup.

CREATE TABLE IF NOT EXISTS inference_cycles (
    id                      BIGSERIAL PRIMARY KEY,
    cycle_uuid              UUID NOT NULL,
    recorded_at             TIMESTAMPTZ NOT NULL DEFAULT now(),

    rpm                     DOUBLE PRECISION,
    current_a               DOUBLE PRECISION,
    voltage_v               DOUBLE PRECISION,
    frequency_hz            DOUBLE PRECISION,
    temperature_c           DOUBLE PRECISION,
    vibration_rms_g         DOUBLE PRECISION,

    fault_class             TEXT,
    confidence              DOUBLE PRECISION,
    model_probabilities     JSONB,

    ohim                    DOUBLE PRECISION,
    rul_hours               DOUBLE PRECISION,
    health_score            DOUBLE PRECISION,

    type_a_uncertainty      DOUBLE PRECISION,
    type_b_uncertainty      DOUBLE PRECISION,
    combined_uncertainty    DOUBLE PRECISION,
    uncertainty_detail      JSONB,

    extra_sensors           JSONB,
    preprocessed_features   JSONB,
    feature_matrix_shape    SMALLINT[],
    feature_matrix_bytes    BYTEA,

    inference_sr_hz         INTEGER,
    target_sr_hz            INTEGER,
    window_size             INTEGER,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_inference_cycles_recorded_at ON inference_cycles (recorded_at);
CREATE INDEX IF NOT EXISTS idx_inference_cycles_fault_class ON inference_cycles (fault_class);
CREATE UNIQUE INDEX IF NOT EXISTS idx_inference_cycles_cycle_uuid ON inference_cycles (cycle_uuid);
CREATE INDEX IF NOT EXISTS idx_inference_cycles_extra_sensors ON inference_cycles USING GIN (extra_sensors);
