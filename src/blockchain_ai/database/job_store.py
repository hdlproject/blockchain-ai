import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class JobStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    address    TEXT PRIMARY KEY,
                    status     TEXT NOT NULL,
                    result     TEXT,
                    error      TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def get(self, address: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT address, status, result, error, created_at, updated_at FROM jobs WHERE address = ?",
                (address,),
            ).fetchone()
        if row is None:
            return None
        return {"address": row[0], "status": row[1], "result": row[2],
                "error": row[3], "created_at": row[4], "updated_at": row[5]}

    def create_pending(self, address: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO jobs (address, status, created_at, updated_at) VALUES (?, 'pending', ?, ?)",
                (address, now, now),
            )

    def mark_done(self, address: str, result: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'done', result = ?, updated_at = ? WHERE address = ?",
                (json.dumps(result), now, address),
            )

    def mark_failed(self, address: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE address = ?",
                (error, now, address),
            )
