"""
Paper trading engine. No real orders are ever placed.

Runs TWO parallel books so the assignment's required A/B comparison is
always available from the same live run:
  A. "basic"    -- enters every SMMA crossover signal, no ML filter
  B. "filtered" -- enters only signals the ML model ACCEPTs, and exits
                   early on a DETERIORATING flag from the monitor

Exit rules (both books, once in a trade): stop-loss, target, max hold
time, or -- filtered book only -- a deterioration exit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from brokers.base import Tick
from config import CONFIG
from features.engineering import FeatureSnapshot
from ml.predict import CrossoverModel, Decision, check_deterioration


class Book(str, Enum):
    BASIC = "basic"
    FILTERED = "filtered"


@dataclass
class Position:
    book: Book
    symbol: str
    signal: str
    entry_price: float
    entry_ts: float
    qty: int
    entry_feat: FeatureSnapshot
    entry_prob: Optional[float] = None
    status: str = "OPEN"
    exit_price: Optional[float] = None
    exit_ts: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    monitor_flag: Optional[str] = None


@dataclass
class TradeLogEntry:
    book: Book
    symbol: str
    signal: str
    decision: Optional[str]
    probability: Optional[float]
    reasons: list
    outcome: str
    pnl: Optional[float] = None
    ts: float = field(default_factory=time.time)


class PaperTradingEngine:
    def __init__(self, model: CrossoverModel):
        self.model = model
        self.open_positions: dict[tuple[Book, str], Position] = {}
        self.closed_positions: list[Position] = []
        self.trade_log: list[TradeLogEntry] = []
        self.avoided_count = 0
        self.avoided_reasons: dict[str, int] = {}

    def on_crossover(self, event, decision: Decision, live_ltp: float,
                     ts: Optional[float] = None):
        # `ts` MUST be the triggering tick's timestamp. Entry and exit
        # timing (elapsed = tick.ts - entry_ts) have to share one clock;
        # with an accelerated simulated clock, mixing in time.time()
        # made every position look "held" for > max_hold_minutes and
        # close instantly at breakeven. Falls back to time.time() only
        # for callers that don't pass it (real-time feeds).
        symbol, signal = event.symbol, event.signal

        self._open(Book.BASIC, symbol, signal, live_ltp, event.features,
                   entry_prob=None, ts=ts)
        self.trade_log.append(TradeLogEntry(
            book=Book.BASIC, symbol=symbol, signal=signal,
            decision=None, probability=None, reasons=["Basic strategy: trade every crossover"],
            outcome="OPEN",
        ))

        if decision.decision == "ACCEPT":
            self._open(Book.FILTERED, symbol, signal, live_ltp, event.features,
                       entry_prob=decision.probability, ts=ts)
            self.trade_log.append(TradeLogEntry(
                book=Book.FILTERED, symbol=symbol, signal=signal,
                decision=decision.decision, probability=decision.probability,
                reasons=decision.reasons, outcome="OPEN",
            ))
        else:
            self.avoided_count += 1
            for r in decision.reasons:
                self.avoided_reasons[r] = self.avoided_reasons.get(r, 0) + 1
            self.trade_log.append(TradeLogEntry(
                book=Book.FILTERED, symbol=symbol, signal=signal,
                decision=decision.decision, probability=decision.probability,
                reasons=decision.reasons, outcome="AVOIDED",
            ))

    def _open(self, book: Book, symbol: str, signal: str, price: float,
              feat: FeatureSnapshot, entry_prob: Optional[float],
              ts: Optional[float] = None):
        key = (book, symbol)
        if key in self.open_positions:
            return
        if sum(1 for (b, _) in self.open_positions if b == book) >= CONFIG.max_concurrent_positions:
            return
        qty = max(1, int(CONFIG.capital_per_trade // price))
        self.open_positions[key] = Position(
            book=book, symbol=symbol, signal=signal, entry_price=price,
            entry_ts=ts if ts is not None else time.time(),
            qty=qty, entry_feat=feat, entry_prob=entry_prob,
        )

    def on_tick(self, tick: Tick, live_feat: Optional[FeatureSnapshot]):
        for book in (Book.BASIC, Book.FILTERED):
            key = (book, tick.symbol)
            pos = self.open_positions.get(key)
            if pos is None:
                continue

            direction = 1 if pos.signal == "BUY" else -1
            move_pct = direction * (tick.ltp - pos.entry_price) / pos.entry_price
            elapsed = tick.ts - pos.entry_ts

            reason = None
            if move_pct >= CONFIG.target_pct:
                reason = "Target hit"
            elif move_pct <= -CONFIG.stop_loss_pct:
                reason = "Stop-loss hit"
            elif elapsed >= CONFIG.max_hold_minutes * 60:
                reason = "Max hold time reached"
            elif book == Book.FILTERED and live_feat is not None and pos.entry_prob is not None:
                flag = check_deterioration(self.model, pos.entry_feat, pos.entry_prob, live_feat)
                if flag:
                    pos.monitor_flag = f"DETERIORATING: {flag}"
                    if move_pct <= 0:
                        reason = f"Deterioration exit ({flag})"
                else:
                    pos.monitor_flag = None

            if reason:
                self._close(pos, tick.ltp, tick.ts, reason)

    def _close(self, pos: Position, price: float, ts: float, reason: str):
        direction = 1 if pos.signal == "BUY" else -1
        pnl_pct = direction * (price - pos.entry_price) / pos.entry_price
        pnl = pnl_pct * pos.entry_price * pos.qty

        pos.status = "CLOSED"
        pos.exit_price = price
        pos.exit_ts = ts
        pos.exit_reason = reason
        pos.pnl = pnl
        pos.pnl_pct = pnl_pct

        key = (pos.book, pos.symbol)
        del self.open_positions[key]
        self.closed_positions.append(pos)
        # NOTE: outcome classification below explicitly checks
        # `pnl is not None` rather than truthy-checking `pnl` directly.
        # A prior version used `if pnl and pnl > 0` / `if pnl and pnl
        # <= 0`, which silently misclassified exact-breakeven trades
        # (pnl == 0.0 is falsy in Python) -- they were excluded from
        # BOTH win and loss buckets in summary() while still counting
        # toward total_trades, discovered via a real breakeven trade in
        # live Fyers data on 2026-08-31/09-01.
        self.trade_log.append(TradeLogEntry(
            book=pos.book, symbol=pos.symbol, signal=pos.signal,
            decision="ACCEPT" if pos.book == Book.FILTERED else None,
            probability=pos.entry_prob, reasons=[reason],
            outcome="WIN" if (pnl is not None and pnl > 0) else "LOSS", pnl=pnl,
        ))

    def summary(self, book: Book) -> dict:
        closed = [p for p in self.closed_positions if p.book == book]
        wins = [p for p in closed if p.pnl is not None and p.pnl > 0]
        losses = [p for p in closed if p.pnl is not None and p.pnl < 0]
        breakeven = [p for p in closed if p.pnl is not None and p.pnl == 0]
        total_pnl = sum(p.pnl for p in closed if p.pnl is not None)
        return {
            "book": book.value,
            "total_trades": len(closed),
            "open_trades": sum(1 for (b, _) in self.open_positions if b == book),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "total_pnl": total_pnl,
        }