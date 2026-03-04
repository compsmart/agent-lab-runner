"""Local SQLite state store for crash recovery.

Persists active job state so the runner can resume uploads and
status reporting after a restart.

Schema:
    jobs(queue_id, lease_token, attempt, work_dir, status, started_at, updated_at)

Status values:
    claimed   — lease obtained, tarball not yet downloaded
    executing — subprocess is running
    uploading — waiting to upload results
    done      — job fully reported; safe to delete
"""
import asyncio
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("runner.state_store")

DEFAULT_DB_PATH = os.path.expanduser("~/.agent-lab-runner/state.db")


class StateStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                queue_id    INTEGER PRIMARY KEY,
                lease_token TEXT    NOT NULL,
                attempt     INTEGER NOT NULL DEFAULT 1,
                work_dir    TEXT,
                status      TEXT    NOT NULL DEFAULT 'claimed',
                -- Use strftime for compatibility with older SQLite builds
                started_at  REAL    NOT NULL DEFAULT (strftime('%s','now')),
                updated_at  REAL    NOT NULL DEFAULT (strftime('%s','now'))
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        """)
        self._conn.commit()

    @contextmanager
    def _cursor(self):
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def upsert_job(self, queue_id: int, lease_token: str, attempt: int,
                   work_dir: Optional[str] = None, status: str = "claimed"):
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (queue_id, lease_token, attempt, work_dir, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(queue_id) DO UPDATE SET
                    lease_token = excluded.lease_token,
                    attempt     = excluded.attempt,
                    work_dir    = COALESCE(excluded.work_dir, jobs.work_dir),
                    status      = excluded.status,
                    updated_at  = excluded.updated_at
                """,
                (queue_id, lease_token, attempt, work_dir, status, time.time()),
            )

    def update_status(self, queue_id: int, status: str):
        with self._cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE queue_id = ?",
                (status, time.time(), queue_id),
            )

    def get_job(self, queue_id: int) -> Optional[dict]:
        cur = self._conn.execute("SELECT * FROM jobs WHERE queue_id = ?", (queue_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_incomplete(self) -> list[dict]:
        """Return all jobs not yet marked done — used for crash recovery."""
        cur = self._conn.execute(
            "SELECT * FROM jobs WHERE status != 'done' ORDER BY started_at ASC"
        )
        return [dict(r) for r in cur.fetchall()]

    def delete_job(self, queue_id: int):
        with self._cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE queue_id = ?", (queue_id,))

    def mark_done(self, queue_id: int):
        self.update_status(queue_id, "done")
        # Clean up after some time to avoid unbounded growth
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM jobs WHERE status = 'done' AND updated_at < ?",
                (time.time() - 3600,),
            )

    def close(self):
        self._conn.close()
