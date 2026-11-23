"""Saved Reports library (SQLite). Every query is scoped to the owner; deletes are
audited. Statements: config/sql.yaml (reports)."""

import json
import sqlite3
import threading
from pathlib import Path

from retail_agent.config import SQL

Q = SQL["reports"]


class ReportStore:
    def __init__(self, path: Path):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(Q["schema"])
        # LangGraph runs parallel tool calls in threads; one sqlite connection must not
        # be shared concurrently.
        # ponytail: one lock per store, fine for a CLI; production uses Postgres with a
        # connection pool.
        self.lock = threading.Lock()

    def save(self, owner: str, conversation_id: str, title: str, body: str) -> int:
        with self.lock, self.con:
            cur = self.con.execute(Q["insert"], (owner, conversation_id, title, body))
        return cur.lastrowid

    def find(
        self, owner: str, text: str | None = None, conversation_id: str | None = None
    ) -> list[dict]:
        sql, args = Q["find"], [owner]
        if text:
            sql, args = f"{sql} {Q['find_text']}", [*args, text, text]
        if conversation_id:
            sql, args = f"{sql} {Q['find_conversation']}", [*args, conversation_id]
        with self.lock:
            return [dict(r) for r in self.con.execute(f"{sql} {Q['order']}", args)]

    def get(self, owner: str, report_id: int) -> dict | None:
        with self.lock:
            row = self.con.execute(Q["get"], (owner, report_id)).fetchone()
        return dict(row) if row else None

    def delete(self, owner: str, ids: list[int]) -> int:
        """Delete exactly these ids, and only if owned. Caller must have confirmed them
        with the user."""
        with self.lock, self.con:
            sql = Q["delete"].format(ids=",".join("?" * len(ids)))
            cur = self.con.execute(sql, [owner, *ids])
            self.con.execute(Q["audit_delete"], (owner, json.dumps(ids)))
        return cur.rowcount
