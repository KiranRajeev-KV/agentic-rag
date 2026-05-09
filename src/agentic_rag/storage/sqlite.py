from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def initialize_schema(self, schema_path: Path) -> None:
        schema_sql = schema_path.read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        with self.connect() as conn:
            conn.execute(query, params)
            conn.commit()

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.connect() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
