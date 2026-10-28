"""
Paper trading engine. No real orders are ever placed.

Runs TWO parallel books so the assignment's required A/B comparison is
always available from the same live run:
  A. "basic"    -- enters every SMMA crossover signal, no ML filter
  B. "filtered" -- enters only signals the ML model ACCEPTs, and exits
                   early on a DETERIORATING flag from the monitor

Exit rules (both books, once in a trade): stop-loss, target, max hold
time, or -- filtered book only -- a deterioration exit.

The deterioration monitor re-scores the model on every tick, and a single
tick's LTQ/depth is noisy enough to swing the score by tens of points. So
it only starts judging after a grace period (monitor_grace_seconds) and
only exits when the warning has persisted for monitor_confirm_seconds.
Hard exits (stop-loss, target, max hold) are never delayed.

Positions still open when a session ends are closed by close_all(), at the
last price seen for each symbol, so they are recorded instead of vanishing.
Those exits carry the reason SESSION_END_REASON, so they can be told apart
from real exits (they are marks to the last price, not fills).

Optionally the engine knows the session close (session_close=(15, 30), in
Indian time). From the first tick at or after that time it opens nothing,
evaluates no exits, and closes every open position at the last price seen
before the close, with the reason SESSION_END_REASON. Without session_close
(the demo and the mock broker) nothing changes.

Optionally the engine records every crossover (with its features) and
re-scores each setup every few seconds for a while afterwards, so the
stability of the model's verdict can be measured offline.

Open positions can be saved as they open (on_open) and put back after a
crash or restart (restore_position), so a restart does not lose them.

Persistence is optional and pluggable: pass `on_log` and/or `on_close`
callbacks (see storage/repository.py) to record trade events. With no
callbacks (the demo), everything stays in memory exactly as before.
"""
from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from brokers.base import Tick
from config import CONFIG
from features.engineering import FEATURE_ORDER, FeatureSnapshot
from ml.predict import CrossoverModel, Decision, check_deterioration

logger = logging.getLogger(__name__)

DEFAULT_MONITOR_GRACE_SECONDS = 120.0
DEFAULT_MONITOR_CONFIRM_SECONDS = 30.0
SESSION_END_REASON = "Session end (marked at last price)"
DEFAULT_SAMPLE_INTERVAL_SECONDS = 5.0
DEFAULT_SAMPLE_WINDOW_SECONDS = 900.0
IST_OFFSET_SECONDS = 5 * 3600 + 30 * 60   # India has no daylight saving
SECONDS_PER_DAY = 86400


def _locked(method):
    """Serialise engine calls: ticks arrive on the broker thread while the
    main thread may call close_all() at shutdown."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class Book(str, Enum):
    BASIC = "basic"
    FILTERED = "filtered"


def _feature_dict(feat) -> dict:
    """The model's input vector keyed by feature name ({} if unavailable)."""
    if feat is None:
        return {}
    return dict(zip(FEATURE_ORDER, (float(v) for v in feat.to_feature_vector())))


class StoredFeatures:
    """
    Entry features restored from storage. It carries every model input by
    name, which is all the deterioration monitor reads from them. Features
    that were not saved default to 0.0.
    """

    def __init__(self, values: Optional[dict] = None):
        values = values or {}
        for name in FEATURE_ORDER:
            setattr(self, name, float(values.get(name, 0.0)))

    def to_feature_vector(self) -> list:
        return [getattr(self, name) for name in FEATURE_ORDER]


@dataclass
class OpenPositionRecord:
    """A position that has just opened, in a form that can be saved and restored."""
    book: str
    symbol: str
    signal: str
    qty: int
    entry_price: float
    entry_ts: float
    entry_prob: Optional[float]
    features: dict


@dataclass
class CrossoverRecord:
    """Every crossover the engine sees, with the features the model scored."""
    symbol: str
    signal: str
    ts: float
    price: float
    decision: str
    probability: Optional[float]
    features: dict


@dataclass
class ScoreSample:
    """The model re-scored on a later tick of the same setup."""
    symbol: str
    signal: str
    event_ts: float
    ts: float
    seconds_since_event: float
    score: float
    price: float


@dataclass
class _SamplingState:
    event_ts: float
    signal: str
    last_ts: float


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
    flag_since: Optional[float] = None  # when the current warning first appeared


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
    def __init__(self, model: CrossoverModel,
                 on_close: Optional[Callable[[Position], None]] = None,
                 on_log: Optional[Callable[[TradeLogEntry], None]] = None,
                 monitor_grace_seconds: float = DEFAULT_MONITOR_GRACE_SECONDS,
                 monitor_confirm_seconds: float = DEFAULT_MONITOR_CONFIRM_SECONDS,
                 on_crossover_event: Optional[Callable[[CrossoverRecord], None]] = None,
                 on_sample: Optional[Callable[[ScoreSample], None]] = None,
                 sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
                 sample_window_seconds: float = DEFAULT_SAMPLE_WINDOW_SECONDS,
                 on_open: Optional[Callable[[OpenPositionRecord], None]] = None,
                 session_close: Optional[tuple[int, int]] = None):
        self.model = model
        self.on_close = on_close
        self.on_log = on_log
        self.on_open = on_open
        self.on_crossover_event = on_crossover_event
        self.on_sample = on_sample
        self.sample_interval_seconds = sample_interval_seconds
        self.sample_window_seconds = sample_window_seconds
        self._sampling: dict[str, _SamplingState] = {}
        self._sampling_disabled = False
        self.monitor_grace_seconds = monitor_grace_seconds
        self.monitor_confirm_seconds = monitor_confirm_seconds
        self._lock = threading.RLock()
        self._last_tick: dict[str, tuple[float, float]] = {}  # symbol -> (price, ts)
        self._post_close_ts: Optional[float] = None   # newest tick seen after the close
        self.session_close = session_close
        self.open_positions: dict[tuple[Book, str], Position] = {}
        self.closed_positions: list[Position] = []
        self.trade_log: list[TradeLogEntry] = []
        self.avoided_count = 0
        self.avoided_reasons: dict[str, int] = {}

    @property
    def session_close(self) -> Optional[tuple[int, int]]:
        """(hour, minute) in Indian time after which the engine stops trading, or None."""
        return self._session_close

    @session_close.setter
    def session_close(self, value: Optional[tuple[int, int]]):
        self._session_close = value
        self._close_seconds = None if value is None else value[0] * 3600 + value[1] * 60

    def _after_close(self, ts: float) -> bool:
        """True if this timestamp is at or after the session close (never with no cutoff set)."""
        if self._close_seconds is None:
            return False
        return (ts + IST_OFFSET_SECONDS) % SECONDS_PER_DAY >= self._close_seconds

    def _log(self, entry: TradeLogEntry):
        self.trade_log.append(entry)
        self._notify(self.on_log, entry)

    @staticmethod
    def _notify(callback, payload):
        # A storage failure must never stop the trading loop.
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:
            logger.exception("Persistence callback failed; trading continues")

    @_locked
    def on_crossover(self, event, decision: Decision, live_ltp: float,
                     ts: Optional[float] = None):
        # `ts` MUST be the triggering tick's timestamp. Entry and exit
        # timing (elapsed = tick.ts - entry_ts) have to share one clock;
        # with an accelerated simulated clock, mixing in time.time()
        # made every position look "held" for > max_hold_minutes and
        # close instantly at breakeven. Falls back to time.time() only
        # for callers that don't pass it (real-time feeds).
        symbol, signal = event.symbol, event.signal
        event_ts = ts if ts is not None else time.time()
        if self._after_close(event_ts):
            return   # the session is over: no new entries, and nothing is recorded
        self._notify(self.on_crossover_event, CrossoverRecord(
            symbol=symbol, signal=signal, ts=event_ts, price=live_ltp,
            decision=decision.decision, probability=decision.probability,
            features=_feature_dict(event.features),
        ))

        skipped = self._open(Book.BASIC, symbol, signal, live_ltp, event.features,
                             entry_prob=None, ts=ts)
        self._log(TradeLogEntry(
            book=Book.BASIC, symbol=symbol, signal=signal,
            decision=None, probability=None,
            reasons=[skipped] if skipped else ["Basic strategy: trade every crossover"],
            outcome="SKIPPED" if skipped else "OPEN",
        ))

        if decision.decision == "ACCEPT":
            skipped = self._open(Book.FILTERED, symbol, signal, live_ltp, event.features,
                                 entry_prob=decision.probability, ts=ts)
            self._log(TradeLogEntry(
                book=Book.FILTERED, symbol=symbol, signal=signal,
                decision=decision.decision, probability=decision.probability,
                reasons=[skipped, *decision.reasons] if skipped else decision.reasons,
                outcome="SKIPPED" if skipped else "OPEN",
            ))
        else:
            self.avoided_count += 1
            for r in decision.reasons:
                self.avoided_reasons[r] = self.avoided_reasons.get(r, 0) + 1
            self._log(TradeLogEntry(
                book=Book.FILTERED, symbol=symbol, signal=signal,
                decision=decision.decision, probability=decision.probability,
                reasons=decision.reasons, outcome="AVOIDED",
            ))

        # Re-score this setup on later ticks, so we can measure how stable
        # the model's verdict really is (see storage/analysis.py).
        if (self.on_sample is not None and symbol not in self._sampling
                and any((b, symbol) in self.open_positions for b in Book)):
            self._sampling[symbol] = _SamplingState(
                event_ts=event_ts, signal=signal, last_ts=event_ts)

    def _open(self, book: Book, symbol: str, signal: str, price: float,
              feat: FeatureSnapshot, entry_prob: Optional[float],
              ts: Optional[float] = None) -> Optional[str]:
        """Opens a position. Returns None on success, or why it was skipped."""
        key = (book, symbol)
        if key in self.open_positions:
            return "Position already open in this symbol"
        if sum(1 for (b, _) in self.open_positions if b == book) >= CONFIG.max_concurrent_positions:
            return "Max concurrent positions reached"
        qty = max(1, int(CONFIG.capital_per_trade // price))
        position = Position(
            book=book, symbol=symbol, signal=signal, entry_price=price,
            entry_ts=ts if ts is not None else time.time(),
            qty=qty, entry_feat=feat, entry_prob=entry_prob,
        )
        self.open_positions[key] = position
        self._notify(self.on_open, OpenPositionRecord(
            book=book.value, symbol=symbol, signal=signal, qty=qty, entry_price=price,
            entry_ts=position.entry_ts, entry_prob=entry_prob, features=_feature_dict(feat),
        ))
        return None

    @_locked
    def restore_position(self, row: dict) -> Optional[Position]:
        """
        Put back a position saved before a crash or restart. It is not saved
        again (it already is). Returns None if that book/symbol is open.
        """
        book = Book(row["book"])
        key = (book, row["symbol"])
        if key in self.open_positions:
            return None
        position = Position(
            book=book, symbol=row["symbol"], signal=row["signal"],
            entry_price=row["entry_price"], entry_ts=row["entry_ts"], qty=row["qty"],
            entry_feat=StoredFeatures(row.get("features")),
            entry_prob=row.get("entry_prob"),
        )
        self.open_positions[key] = position
        return position

    @_locked
    def close_position(self, book: Book, symbol: str, price: float, ts: float,
                       reason: str) -> bool:
        """Close one open position at a given price. False if it is not open."""
        pos = self.open_positions.get((book, symbol))
        if pos is None:
            return False
        self._close(pos, price, ts, reason)
        return True

    @_locked
    def on_tick(self, tick: Tick, live_feat: Optional[FeatureSnapshot]):
        if self._after_close(tick.ts):
            # The session is over. Post-close prices are not tradeable, so no
            # exit rule is evaluated on them. Every open position is marked at
            # the last price seen BEFORE the close: _last_tick is deliberately
            # not updated here.
            if self._post_close_ts is None or tick.ts > self._post_close_ts:
                self._post_close_ts = tick.ts
            if self.open_positions:
                self._sampling.clear()
                self.close_all()
            return

        self._last_tick[tick.symbol] = (tick.ltp, tick.ts)
        self._maybe_sample(tick, live_feat)
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
            elif (book == Book.FILTERED and live_feat is not None
                  and pos.entry_prob is not None
                  and elapsed >= self.monitor_grace_seconds):
                flag = check_deterioration(self.model, pos.entry_feat, pos.entry_prob, live_feat)
                if flag:
                    pos.monitor_flag = f"DETERIORATING: {flag}"
                    if pos.flag_since is None:
                        pos.flag_since = tick.ts
                    persisted = tick.ts - pos.flag_since >= self.monitor_confirm_seconds
                    if move_pct <= 0 and persisted:
                        reason = f"Deterioration exit ({flag})"
                else:
                    pos.monitor_flag = None
                    pos.flag_since = None

            if reason:
                self._close(pos, tick.ltp, tick.ts, reason)

    def _maybe_sample(self, tick: Tick, live_feat: Optional[FeatureSnapshot]):
        state = self._sampling.get(tick.symbol)
        if state is None:
            return
        still_open = any((b, tick.symbol) in self.open_positions for b in Book)
        if not still_open or tick.ts - state.event_ts > self.sample_window_seconds:
            del self._sampling[tick.symbol]
            return
        if (live_feat is None or self.model is None or self._sampling_disabled
                or tick.ts - state.last_ts < self.sample_interval_seconds):
            return
        state.last_ts = tick.ts
        try:
            score = float(self.model.score(live_feat))
        except Exception:
            logger.exception("Score sampling failed; switching it off for this run")
            self._sampling_disabled = True
            return
        self._notify(self.on_sample, ScoreSample(
            symbol=tick.symbol, signal=state.signal, event_ts=state.event_ts,
            ts=tick.ts, seconds_since_event=tick.ts - state.event_ts,
            score=score, price=tick.ltp,
        ))

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
        self._log(TradeLogEntry(
            book=pos.book, symbol=pos.symbol, signal=pos.signal,
            decision="ACCEPT" if pos.book == Book.FILTERED else None,
            probability=pos.entry_prob, reasons=[reason],
            outcome="WIN" if (pnl is not None and pnl > 0) else "LOSS", pnl=pnl,
        ))
        self._notify(self.on_close, pos)

    @_locked
    def latest_tick_ts(self) -> Optional[float]:
        """
        Timestamp of the newest tick seen on any symbol (None before the first).
        Ticks after the session close count too, so the caller can tell the
        trading day is over even though those ticks no longer move any price.
        """
        seen = [ts for _, ts in self._last_tick.values()]
        if self._post_close_ts is not None:
            seen.append(self._post_close_ts)
        return max(seen, default=None)

    @_locked
    def close_all(self, reason: str = SESSION_END_REASON) -> int:
        """
        Close every open position at the last price seen for its symbol
        (its entry price if no tick arrived since). Returns how many were
        closed; calling it again returns 0.
        """
        closed = 0
        for pos in list(self.open_positions.values()):
            price, ts = self._last_tick.get(pos.symbol, (pos.entry_price, pos.entry_ts))
            self._close(pos, price, ts, reason)
            closed += 1
        return closed

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