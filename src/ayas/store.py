"""SQLite log of every inspection and every operator conversation."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    part_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    verdict TEXT NOT NULL,
    defect_count INTEGER NOT NULL,
    report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL REFERENCES inspections(id),
    ts TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def add_inspection(self, report: dict[str, Any]) -> int:
        cur = self.db.execute(
            "INSERT INTO inspections (part_id, ts, verdict, defect_count, report_json) VALUES (?, ?, ?, ?, ?)",
            (report["part_id"], report["timestamp"], report["verdict"], len(report["defects"]), json.dumps(report)),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def add_chat(self, inspection_id: int, role: str, text: str) -> None:
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        self.db.execute("INSERT INTO chats (inspection_id, ts, role, text) VALUES (?, ?, ?, ?)", (inspection_id, ts, role, text))
        self.db.commit()

    def recent(self, n: int = 10) -> list[tuple]:
        return self.db.execute(
            "SELECT id, part_id, ts, verdict, defect_count FROM inspections ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()

    def verdict_counts(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT verdict, COUNT(*) FROM inspections GROUP BY verdict").fetchall())

    def close(self) -> None:
        self.db.close()
