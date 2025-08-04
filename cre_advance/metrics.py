from __future__ import annotations

"""Simple metrics store backed by SQLite."""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List

_DB_PATH = Path(__file__).resolve().parent / "metrics.db"
_CONN: sqlite3.Connection | None = None
_LOCK = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    """Return a SQLite connection, initialising the DB if needed."""

    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _CONN.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value TEXT,
                tags TEXT,
                feedback TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _CONN.commit()
    return _CONN


def log_metric(
    name: str,
    value: Any,
    tags: Dict[str, Any] | None = None,
    feedback: Dict[str, Any] | None = None,
) -> None:
    """Persist a metric with optional tags and user feedback."""

    with _LOCK:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO metrics (name, value, tags, feedback) VALUES (?, ?, ?, ?)",
            (
                name,
                json.dumps(value),
                json.dumps(tags or {}),
                json.dumps(feedback or {}),
            ),
        )
        conn.commit()


def get_metrics(
    name: str | None = None, tags: Dict[str, Any] | None = None
) -> List[Dict[str, Any]]:
    """Return metrics optionally filtered by ``name`` and ``tags``."""

    with _LOCK:
        conn = _get_conn()
        cur = conn.execute(
            "SELECT name, value, tags, feedback, ts FROM metrics ORDER BY id"
        )
        rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for n, val, t, fb, ts in rows:
        if name and n != name:
            continue
        parsed_tags = json.loads(t) if t else {}
        if tags and not all(parsed_tags.get(k) == v for k, v in tags.items()):
            continue
        results.append(
            {
                "name": n,
                "value": json.loads(val),
                "tags": parsed_tags,
                "feedback": json.loads(fb) if fb else {},
                "ts": ts,
            }
        )
    return results


def log_feedback(
    name: str, corrections: Dict[str, Any], tags: Dict[str, Any] | None = None
) -> None:
    """Store user feedback for a given metric name."""

    log_metric(name, None, tags=tags, feedback=corrections)
