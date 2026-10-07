"""
Database connection setup (SQLAlchemy 2.x).

Defaults to a local SQLite file so the project runs with zero setup.
Set SCREENER_DB_URL to point at Postgres (or anything SQLAlchemy
supports) later without touching any other code, e.g.
    postgresql+psycopg://user:pass@host/dbname
"""
from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def default_database_url() -> str:
    override = os.environ.get("SCREENER_DB_URL")
    if override:
        return override
    from config import DATA_DIR  # imported lazily so tests can pass an explicit URL
    return f"sqlite:///{DATA_DIR / 'screener.db'}"


def make_engine(url: Optional[str] = None) -> Engine:
    engine = create_engine(url or default_database_url())
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            # WAL lets the dashboard read while the trading loop writes.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine