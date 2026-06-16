import sqlite3
from pathlib import Path


class FunderLedger:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS funder_ledger (
                    funder TEXT NOT NULL,
                    wallet TEXT NOT NULL,
                    PRIMARY KEY (funder, wallet)
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def record(self, funder: str, wallet: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO funder_ledger (funder, wallet) VALUES (?, ?)",
                (funder.lower(), wallet.lower()),
            )

    def funded_count(self, funder: str, exclude_wallet: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT wallet) FROM funder_ledger WHERE funder = ? AND wallet != ?",
                (funder.lower(), exclude_wallet.lower()),
            ).fetchone()
        return int(row[0]) if row else 0
