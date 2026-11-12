# Digital Twin extension: automatic PostgreSQL dataset generation + GUM measurement uncertainty

## 0. What was actually in the codebase (read this first)

Before writing anything, I unzipped and read both archives. The report describes a fully
wired pipeline (MPU-6050 → feature extraction → ONNX ensemble → HTTPS telemetry → dashboard),
but the actual code does not match that in one important way:

- **`control_panel.py`** (identical in both uploads) reads VFD registers + DS18B20 temperature
  and publishes over **MQTT to a Railway broker**. It has **no MPU-6050 code**, hardcodes
  `vibration: 0.0`, and **never calls `inference_server.py`**. No fault prediction happens on
  the Pi side at all in the code as given.
- **`inference_server.py`** is real and matches the report closely: FastAPI, 4 ONNX models +
  a pickled meta-learner, STFT/Hilbert feature extraction, `/health` and `/predict`. This is the
  *only* place a prediction is actually produced anywhere in the codebase.
- **`web-dashboard/api/telemetry.js` / `api/command.js`** are real and match the report (Vercel
  KV, 10s offline detection, pending/ack command queue).
- **`mock-sensors/pi_controller.py`** and **`mock-sensors/motor_sensor_simulator.py`** are
  unintegrated reference/template scripts (different Modbus library, stubbed sensor collection,
  a third MQTT variant) — not part of the live pipeline.
- Model artifacts (`dt_model_artifacts/*.onnx`, `meta_learner.pkl`) are not included in either
  upload, so `inference_server.py` cannot actually be started/tested end-to-end in this
  environment; the new logic added below was unit-tested independently instead (Section 5).

**Per your instruction, this extension plugs into `inference_server.py` only** — the only place
predictions currently exist — and does not touch `control_panel.py`.

## 1. Directory structure (new files only)

```
inference_server.py        <- MODIFIED (additive only, see diff summary in Section 2)
uncertainty/                <- NEW, Task 2
    __init__.py             <- UncertaintyEngine facade
    type_a.py               <- Type A: rolling-window statistical uncertainty
    type_b.py               <- Type B: instrument/resolution uncertainty, configurable
    combiner.py             <- GUM combination -> combined + expanded uncertainty
storage/                    <- NEW, Task 1
    __init__.py
    config.py               <- Postgres config from env vars (no hardcoded credentials)
    connection.py           <- pooled connections + reconnection/retry
    schema.py               <- idempotent CREATE TABLE/INDEX (auto-migration on startup)
    models.py               <- InferenceCycleRecord dataclass (one row = one inference cycle)
    repository.py           <- transactional insert
sql/
    001_create_inference_cycles.sql   <- same DDL, for teams that prefer manual migrations
.env.example                <- Postgres credentials template
requirements-additions.txt  <- psycopg2-binary, python-dotenv
```

## 2. What changed in `inference_server.py` (additive only)

- New imports for `uncertainty` / `storage`, plus `uuid`, `logging`, `BackgroundTasks`.
- One new startup block: builds an `UncertaintyEngine`, then tries to build a
  `PostgresConnectionManager` + `ensure_schema()` + `InferenceCycleRepository`. If Postgres is
  unreachable at startup, this **logs a warning and continues** — dataset logging is disabled,
  but the inference server still starts and `/predict` still works. Nothing about the existing
  model-loading code was touched.
- `PredictRequest` gained optional fields: `rpm`, `current`, `voltage`, `frequency`,
  `temperature`, `extra_sensors`. All default to `None`, so any existing caller sending only
  `window` behaves exactly as before. These had to be added because the endpoint previously had
  *no* way to receive this telemetry at all, and Task 1 requires logging it per cycle.
- `PredictResponse` gained optional `uncertainty` (per-channel GUM breakdown) and `cycle_id`
  fields.
- Inside `predict()`: after the existing inference logic (unchanged), it computes vibration RMS
  from the raw window, evaluates uncertainty for every channel it has a reading for, builds an
  `InferenceCycleRecord`, and schedules the DB write as a `BackgroundTask` (runs after the HTTP
  response is already sent, so a slow/failing DB write adds zero latency to `/predict`).
- `/health` now also reports whether dataset logging is enabled.

## 3. Design decisions and why

**Why PostgreSQL logging lives in `inference_server.py` and not `control_panel.py`:**
Per your answer — this is the only place a prediction actually exists right now.

**Why `PredictRequest` needed new optional fields:** the endpoint had no path for
rpm/current/voltage/frequency/temperature at all. Making them optional (not required) keeps
every existing caller working unmodified — this is the "as little coupling/no rewriting" rule
in practice: add a door, don't move the walls.

**Why the schema mixes typed columns and JSONB:** `rpm`, `current_a`, `voltage_v`, `fault_class`,
`confidence`, and the three uncertainty numbers are typed columns because they're the fields
you'll actually filter/aggregate/join on for trend analysis and retraining datasets — indexing
JSON for that is painful. `extra_sensors` and `uncertainty_detail` are JSONB specifically so a
sensor you add next year (PZEM-004T power quality, per the report's Future Work) doesn't require
a migration; it just becomes a new key until it earns a typed column.

**Why `ohim`, `rul_hours`, `health_score` are nullable and always `NULL` right now:** nothing in
the codebase computes any of them. Per the "do not hallucinate" instruction, I did not invent a
health-score formula — the columns exist so the schema doesn't need to change once a real one is
built, and stay `NULL` until then.

**Why the DB write is a `BackgroundTask`, not inline:** the report's own Section 5.4 already
established the pattern of moving network I/O to background threads so it can't add latency to
the control loop's 1-second interval; the DB write follows the same principle for `/predict`'s
response time.

**Why Type B falls back to `resolution/√12`:** no manufacturer accuracy % is documented anywhere
in the uploaded report or code for the MPU-6050, DS18B20, or the VFD's ADC — only register
scaling / valid ranges are. Rather than invent accuracy figures, every channel defaults to the
standard GUM resolution-only estimate, and every default is documented with its actual source
(see `uncertainty/type_b.py`'s module docstring) so you can tell at a glance which numbers came
from this project's own report vs. which are generic placeholders to replace with real datasheet
or calibration figures.

**Why the combined uncertainty uses k=2 rather than a Welch-Satterthwaite effective-DOF
calculation:** doing the latter properly requires degrees-of-freedom information per input that
isn't available here; k=2 is flagged in the docstring as a conventional approximation, not
presented as more rigorous than it is.

**Why `feature_matrix_bytes` is off by default:** the full (17, 769) float32 STFT+envelope
matrix is ~52 KB per cycle; at a 1-second cadence that's ~4.5 GB/day if stored raw on every row.
`preprocessed_features` (JSONB summary stats) is always populated; the raw matrix is opt-in via
`DT_STORE_FEATURE_MATRIX=true` for when you specifically need it for retraining and have the
disk budget.

## 4. Installation

```bash
# from wherever inference_server.py already lives
pip install -r requirements-additions.txt   # psycopg2-binary, python-dotenv
# (existing requirements for fastapi/onnxruntime/etc. are untouched)

cp .env.example .env
# edit .env with real Postgres host/user/password
```

Create the database once (the app will create the table itself on first startup):

```bash
createdb digital_twin           # or: CREATE DATABASE digital_twin; via psql
```

## 5. Testing procedure

What I could verify in this sandbox (no live Postgres server was installable here — the
`postgresql-16` apt package is missing from the mirror snapshot currently served; no ONNX model
artifacts were included in either upload, so the full server can't be started end-to-end
regardless of these changes):

- `python3 -m py_compile inference_server.py uncertainty/*.py storage/*.py` — all modules
  compile cleanly.
- `uncertainty.UncertaintyEngine` — ran a 7-cycle simulation with noisy vibration/current
  readings; confirmed the rolling window caps at `window_size`, Type A shrinks as `n` grows,
  and combined/expanded uncertainty compute correctly.
- `storage` — with no Postgres reachable, confirmed `PostgresConnectionManager` logs and
  degrades gracefully (no exception escapes) and `InferenceCycleRepository.insert()` returns
  `False` rather than raising, so a DB outage can't take down `/predict`.

What you should verify once you have a real Postgres instance and the model artifacts:

1. Start `inference_server.py`; confirm the startup log shows
   `[SUCCESS] PostgreSQL dataset-generation pipeline ready (...)` and `GET /health` reports
   `"dataset_logging": "enabled"`.
2. `psql -d digital_twin -c '\d inference_cycles'` — confirm the table/indexes exist.
3. `curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" -d '{"window": [...], "current": 1.3, "voltage": 230.1, "temperature": 27.4}'`
   — confirm the response includes `uncertainty` and `cycle_id`.
4. `SELECT * FROM inference_cycles ORDER BY id DESC LIMIT 1;` — confirm the row matches.
5. Stop Postgres mid-run, call `/predict` again — confirm it still returns 200 (uncertainty
   still computed, just no DB row), and restart Postgres — confirm logging resumes without
   restarting `inference_server.py`.

## 6. Migration steps

New deployment: nothing to migrate, `ensure_schema()` creates everything on first run.
Existing deployment: this is the first version of the table, so there's no prior schema to
migrate from. Future column additions should extend `CREATE_TABLE_SQL`/`sql/001_...sql` with
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...` in a new `002_...sql` rather than editing
`001_create_inference_cycles.sql` in place.

## 7. Required Python packages

See `requirements-additions.txt`: `psycopg2-binary`, `python-dotenv`. Nothing else changes.
