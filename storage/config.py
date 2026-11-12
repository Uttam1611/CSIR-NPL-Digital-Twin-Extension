"""
Configuration for the PostgreSQL dataset-generation pipeline.

All values come from environment variables so credentials are never
hardcoded in source (Task 1 requirement: "configurable credentials"). Locally
you can put these in a `.env` file next to inference_server.py (see
`.env.example`) and python-dotenv will load it automatically; in any real
deployment, set the same variables through the platform's secret manager
instead of committing a `.env` file.
"""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional -- env vars can also be set directly


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    minconn: int
    maxconn: int
    connect_timeout: int
    sslmode: str


def load_postgres_config() -> PostgresConfig:
    return PostgresConfig(
        host=os.environ.get("DT_PG_HOST", "localhost"),
        port=int(os.environ.get("DT_PG_PORT", "5432")),
        database=os.environ.get("DT_PG_DATABASE", "digital_twin"),
        user=os.environ.get("DT_PG_USER", "digital_twin"),
        password=os.environ.get("DT_PG_PASSWORD", ""),
        minconn=int(os.environ.get("DT_PG_MINCONN", "1")),
        maxconn=int(os.environ.get("DT_PG_MAXCONN", "5")),
        connect_timeout=int(os.environ.get("DT_PG_CONNECT_TIMEOUT", "5")),
        sslmode=os.environ.get("DT_PG_SSLMODE", "prefer"),
    )
