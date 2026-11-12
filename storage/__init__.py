"""
storage/ — Automatic PostgreSQL dataset-generation pipeline for the digital twin.

Every inference cycle (one call to POST /predict in inference_server.py) can
be persisted here as one row, capturing telemetry, prediction, and
uncertainty together, for later use in trend analysis, OHIM/RUL research, and
model retraining.

This package depends only on psycopg2 and the standard library -- it does not
import FastAPI, MQTT, or anything from `uncertainty/`, so it can be reused
from any future component (e.g. a batch job, or an edge-inference build on
the Pi) that needs to write inference cycles to Postgres.
"""
