"""
Tables. Every row that belongs to a person carries a user_id, so moving
from "one local user" to many users later is a data-model no-op.

Timestamps are stored as floats exactly as the engine produced them
(epoch seconds, or the simulated clock in demo mode).
"""
from __future__ import annotations

import time
from typing import Optional

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from storage.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class PaperPosition(Base):
    """One CLOSED paper trade."""
    __tablename__ = "paper_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    book: Mapped[str] = mapped_column(String(16))
    symbol: Mapped[str] = mapped_column(String(32))
    signal: Mapped[str] = mapped_column(String(8))
    qty: Mapped[int] = mapped_column(Integer)
    entry_price: Mapped[float] = mapped_column(Float)
    entry_ts: Mapped[float] = mapped_column(Float)
    entry_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float] = mapped_column(Float)
    exit_ts: Mapped[float] = mapped_column(Float)
    exit_reason: Mapped[str] = mapped_column(String(160))
    pnl: Mapped[float] = mapped_column(Float)
    pnl_pct: Mapped[float] = mapped_column(Float)


class OpenPosition(Base):
    """
    A position that is open right now, saved so that a crash or restart can
    resume it instead of losing it. Removed when the position closes.
    """
    __tablename__ = "open_positions"
    __table_args__ = (UniqueConstraint("user_id", "book", "symbol", name="uq_open_position"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    book: Mapped[str] = mapped_column(String(16))
    symbol: Mapped[str] = mapped_column(String(32))
    signal: Mapped[str] = mapped_column(String(8))
    qty: Mapped[int] = mapped_column(Integer)
    entry_price: Mapped[float] = mapped_column(Float)
    entry_ts: Mapped[float] = mapped_column(Float)
    entry_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    features: Mapped[str] = mapped_column(Text)  # JSON {feature name: value}


class TradeLogRow(Base):
    """Every decision event: opened, avoided, or closed (WIN/LOSS)."""
    __tablename__ = "trade_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    book: Mapped[str] = mapped_column(String(16))
    symbol: Mapped[str] = mapped_column(String(32))
    signal: Mapped[str] = mapped_column(String(8))
    decision: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reasons: Mapped[str] = mapped_column(Text)  # JSON-encoded list of strings
    outcome: Mapped[str] = mapped_column(String(16))
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ts: Mapped[float] = mapped_column(Float)


class LiveSnapshot(Base):
    """
    The latest live dashboard state, one row per user, overwritten about
    once a second by the trading process. Replaces polling a JSON file:
    the database write is atomic, so the dashboard never reads a
    half-written snapshot.
    """
    __tablename__ = "live_snapshots"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    updated_at: Mapped[float] = mapped_column(Float)
    payload: Mapped[str] = mapped_column(Text)  # JSON-encoded state dict


class CrossoverEvent(Base):
    """Every crossover seen, with the features the model scored (JSON)."""
    __tablename__ = "crossover_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32))
    signal: Mapped[str] = mapped_column(String(8))
    ts: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(16))
    probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    features: Mapped[str] = mapped_column(Text)  # JSON {feature name: value}


class ScoreSample(Base):
    """The model re-scored on a later tick of a crossover's setup."""
    __tablename__ = "score_samples"
    __table_args__ = (Index("ix_score_samples_event", "user_id", "symbol", "event_ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    symbol: Mapped[str] = mapped_column(String(32))
    event_ts: Mapped[float] = mapped_column(Float)   # ts of the crossover it belongs to
    ts: Mapped[float] = mapped_column(Float)
    seconds_since_event: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)