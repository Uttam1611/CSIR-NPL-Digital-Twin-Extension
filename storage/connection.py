"""
PostgreSQL connection pooling with robust reconnection.

Uses psycopg2's ThreadedConnectionPool, since FastAPI/uvicorn runs sync `def`
endpoint handlers (like the existing /predict) in a worker thread pool, not a
single event loop thread.

`get_connection()` validates every connection it hands out with `SELECT 1`
before use, and transparently rebuilds the pool if it's missing or a
connection has gone stale (e.g. the DB restarted, or a network blip dropped
the TCP connection) -- retrying with a short backoff rather than failing the
first time a connection turns out to be dead.
"""
import logging
import time
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2 import pool as pg_pool

from .config import PostgresConfig

logger = logging.getLogger("digital_twin.storage")


class PostgresConnectionManager:
    def __init__(self, config: PostgresConfig, max_retries: int = 3, retry_backoff_s: float = 0.5):
        self._config = config
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._pool: Optional[pg_pool.ThreadedConnectionPool] = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._pool = pg_pool.ThreadedConnectionPool(
                self._config.minconn,
                self._config.maxconn,
                host=self._config.host,
                port=self._config.port,
                dbname=self._config.database,
                user=self._config.user,
                password=self._config.password,
                connect_timeout=self._config.connect_timeout,
                sslmode=self._config.sslmode,
            )
            logger.info("PostgreSQL connection pool established (%s:%s/%s)",
                        self._config.host, self._config.port, self._config.database)
        except Exception as e:
            logger.error("Failed to establish PostgreSQL connection pool: %s", e)
            self._pool = None

    @contextmanager
    def get_connection(self):
        """Yield a validated connection from the pool, retrying/rebuilding on failure."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            if self._pool is None:
                self._connect()
            if self._pool is None:
                last_error = RuntimeError("PostgreSQL pool unavailable")
                time.sleep(self._retry_backoff_s * attempt)
                continue

            conn = None
            try:
                conn = self._pool.getconn()
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            except Exception as e:
                last_error = e
                logger.warning("PostgreSQL connection attempt %d/%d failed: %s",
                                attempt, self._max_retries, e)
                if conn is not None:
                    try:
                        self._pool.putconn(conn, close=True)
                    except Exception:
                        pass
                self._pool = None  # force a full rebuild next attempt
                time.sleep(self._retry_backoff_s * attempt)
                continue

            try:
                yield conn
            finally:
                self.release_connection(conn)
            return

        raise ConnectionError(
            f"Could not obtain a healthy PostgreSQL connection after {self._max_retries} attempts"
        ) from last_error

    def release_connection(self, conn) -> None:
        if self._pool is not None and conn is not None:
            try:
                self._pool.putconn(conn)
            except Exception as e:
                logger.warning("Failed to release connection back to pool: %s", e)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.closeall()
            logger.info("PostgreSQL connection pool closed.")
