"""
All database reads and writes live here, so the rest of the codebase
never touches SQL. The trading engine stays database-agnostic: it only
exposes callbacks, and this class is what gets plugged into them.
"""
from __future__ import annotations

import json
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from storage.db import Base
from storage.models import PaperPosition, TradeLogRow, User


class Repository:
    def __init__(self, engine: Engine):
        Base.metadata.create_all(engine)
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    # ----------------------------------------------------------- users
    def ensure_user(self, username: str = "local") -> int:
        with self._sessions() as session:
            user = session.scalar(select(User).where(User.username == username))
            if user is None:
                user = User(username=username)
                session.add(user)
                session.commit()
            return user.id

    # ---------------------------------------------------------- writes
    def record_closed_position(self, user_id: int, pos) -> None:
        """`pos` is a trading.paper_engine.Position that has been closed."""
        if pos.pnl is None or pos.exit_price is None or pos.exit_ts is None:
            raise ValueError("record_closed_position needs a closed position")
        with self._sessions() as session:
            session.add(PaperPosition(
                user_id=user_id, book=_book_value(pos.book), symbol=pos.symbol,
                signal=pos.signal, qty=pos.qty, entry_price=pos.entry_price,
                entry_ts=pos.entry_ts, entry_prob=pos.entry_prob,
                exit_price=pos.exit_price, exit_ts=pos.exit_ts,
                exit_reason=pos.exit_reason or "", pnl=pos.pnl, pnl_pct=pos.pnl_pct or 0.0,
            ))
            session.commit()

    def record_trade_log(self, user_id: int, entry) -> None:
        """`entry` is a trading.paper_engine.TradeLogEntry."""
        with self._sessions() as session:
            session.add(TradeLogRow(
                user_id=user_id, book=_book_value(entry.book), symbol=entry.symbol,
                signal=entry.signal, decision=entry.decision, probability=entry.probability,
                reasons=json.dumps(list(entry.reasons)), outcome=entry.outcome,
                pnl=entry.pnl, ts=entry.ts,
            ))
            session.commit()

    # ----------------------------------------------------------- reads
    def closed_positions(self, user_id: int, book: Optional[str] = None) -> List[PaperPosition]:
        query = select(PaperPosition).where(PaperPosition.user_id == user_id)
        if book is not None:
            query = query.where(PaperPosition.book == book)
        with self._sessions() as session:
            return list(session.scalars(query.order_by(PaperPosition.id)))

    def trade_log(self, user_id: int, limit: Optional[int] = None) -> List[dict]:
        query = (select(TradeLogRow).where(TradeLogRow.user_id == user_id)
                 .order_by(TradeLogRow.id.desc()))
        if limit is not None:
            query = query.limit(limit)
        with self._sessions() as session:
            rows = list(session.scalars(query))
        return [
            {"book": r.book, "symbol": r.symbol, "signal": r.signal,
             "decision": r.decision, "probability": r.probability,
             "reasons": json.loads(r.reasons), "outcome": r.outcome,
             "pnl": r.pnl, "ts": r.ts}
            for r in rows
        ]

    def summary(self, user_id: int, book: str) -> dict:
        """Same win/loss/breakeven accounting as PaperTradingEngine.summary()."""
        pnls = [p.pnl for p in self.closed_positions(user_id, book)]
        wins = sum(1 for x in pnls if x > 0)
        losses = sum(1 for x in pnls if x < 0)
        breakeven = sum(1 for x in pnls if x == 0)
        return {
            "book": book,
            "total_trades": len(pnls),
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate": wins / len(pnls) if pnls else 0.0,
            "total_pnl": sum(pnls),
        }


def _book_value(book) -> str:
    return getattr(book, "value", book)