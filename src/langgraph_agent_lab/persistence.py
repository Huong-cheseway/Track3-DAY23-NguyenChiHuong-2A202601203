"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def _sqlite_path(database_url: str | None) -> str:
    """Normalize a plain path or sqlite:/// URL and create its parent directory."""
    target = database_url or "checkpoints.db"
    if target.startswith("sqlite:///"):
        target = target.removeprefix("sqlite:///")
    if target == ":memory:":
        return target

    path = Path(target).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def build_checkpointer(
    kind: str = "memory",
    database_url: str | None = None,
) -> BaseCheckpointSaver[str] | None:
    """Return a LangGraph checkpointer.

    Memory is useful for tests; SQLite provides durable checkpoints with WAL enabled.
    PostgreSQL remains an optional extension.
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        connection = sqlite3.connect(
            _sqlite_path(database_url),
            check_same_thread=False,
        )
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        saver = SqliteSaver(conn=connection)
        saver.setup()
        return saver
    if kind == "postgres":
        raise NotImplementedError("Postgres checkpointer is an optional extension")
    raise ValueError(f"Unknown checkpointer kind: {kind}")
