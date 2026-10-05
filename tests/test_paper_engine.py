"""
Regression tests for real bugs found and fixed in the paper-trading engine.

Run from the project root with:  pytest
"""
import time
from types import SimpleNamespace

from brokers.base import Tick
from config import CONFIG
from ml.predict import Decision
from trading.paper_engine import Book, PaperTradingEngine

PRICE = 100.0
# A simulated clock deliberately far ahead of wall-clock time. This is
# what the accelerated demo clock looks like, and it is what exposed the
# original bug (entry stamped with time.time(), exit compared against
# tick.ts).
SIM_START = time.time() + 10 * 24 * 3600
MAX_HOLD_SECONDS = CONFIG.max_hold_minutes * 60


def make_tick(symbol, ts, ltp):
    return Tick(
        symbol=symbol, ts=ts, ltp=ltp, ltq=100, total_traded_qty=10_000,
        bid_price=ltp - 0.05, bid_qty=900_000,
        ask_price=ltp + 0.05, ask_qty=900_000,
    )


def make_event(symbol="TEST", signal="BUY"):
    # The engine only reads .symbol, .signal and .features from an event.
    return SimpleNamespace(symbol=symbol, signal=signal, features=None)


def make_decision(kind, symbol="TEST", signal="BUY"):
    if kind == "ACCEPT":
        return Decision(symbol=symbol, signal=signal, probability=0.9,
                        decision="ACCEPT", reasons=["test accept"])
    return Decision(symbol=symbol, signal=signal, probability=None,
                    decision="AVOID", reasons=["No trained model available"])


def run_trade(engine, symbol, start_ts, exit_price, hold_seconds=5.0, signal="BUY"):
    """Open a Basic-book trade at PRICE, then send one exit tick."""
    engine.on_crossover(make_event(symbol, signal),
                        make_decision("AVOID", symbol, signal), PRICE, ts=start_ts)
    engine.on_tick(make_tick(symbol, start_ts + hold_seconds, exit_price), None)
    return start_ts + hold_seconds


# --- Bug: entry timestamp used wall-clock, exit used the simulated clock ---

def test_position_not_closed_instantly_on_simulated_clock():
    engine = PaperTradingEngine(model=None)
    engine.on_crossover(make_event(), make_decision("AVOID"), PRICE, ts=SIM_START)
    engine.on_tick(make_tick("TEST", SIM_START + 5, PRICE), None)

    assert (Book.BASIC, "TEST") in engine.open_positions
    assert engine.closed_positions == []


def test_max_hold_exit_is_measured_on_the_tick_clock():
    engine = PaperTradingEngine(model=None)
    engine.on_crossover(make_event(), make_decision("AVOID"), PRICE, ts=SIM_START)
    exit_ts = SIM_START + MAX_HOLD_SECONDS + 1
    engine.on_tick(make_tick("TEST", exit_ts, PRICE), None)

    assert len(engine.closed_positions) == 1
    pos = engine.closed_positions[0]
    assert pos.exit_reason == "Max hold time reached"
    assert pos.exit_ts == exit_ts


# --- Bug: exact-breakeven trades were dropped from win AND loss counts ---

def test_breakeven_trade_is_counted_as_breakeven():
    engine = PaperTradingEngine(model=None)
    run_trade(engine, "TEST", SIM_START, PRICE, hold_seconds=MAX_HOLD_SECONDS + 1)

    s = engine.summary(Book.BASIC)
    assert s["total_trades"] == 1
    assert s["breakeven"] == 1
    assert s["wins"] == 0
    assert s["losses"] == 0


def test_wins_losses_and_breakevens_add_up_to_total_trades():
    engine = PaperTradingEngine(model=None)
    win_price = PRICE * (1 + CONFIG.target_pct + 0.001)
    loss_price = PRICE * (1 - CONFIG.stop_loss_pct - 0.001)

    ts = SIM_START
    ts = run_trade(engine, "WIN", ts, win_price)
    ts = run_trade(engine, "LOSS", ts, loss_price)
    ts = run_trade(engine, "FLAT", ts, PRICE, hold_seconds=MAX_HOLD_SECONDS + 1)

    s = engine.summary(Book.BASIC)
    assert s["total_trades"] == 3
    assert (s["wins"], s["losses"], s["breakeven"]) == (1, 1, 1)
    assert s["wins"] + s["losses"] + s["breakeven"] == s["total_trades"]
    assert s["win_rate"] == 1 / 3


def test_sell_trade_profits_when_price_falls():
    engine = PaperTradingEngine(model=None)
    run_trade(engine, "TEST", SIM_START, PRICE * (1 - CONFIG.target_pct - 0.001),
              signal="SELL")

    pos = engine.closed_positions[0]
    assert pos.exit_reason == "Target hit"
    assert pos.pnl > 0


# --- Bug: Basic book never traded when no ML model was available ---
# (The demo wraps this by passing a placeholder AVOID decision; these tests
# cover the engine half: an AVOID decision must still open a Basic trade.)

def test_basic_book_trades_even_when_decision_is_avoid():
    engine = PaperTradingEngine(model=None)
    engine.on_crossover(make_event(), make_decision("AVOID"), PRICE, ts=SIM_START)

    assert (Book.BASIC, "TEST") in engine.open_positions
    assert (Book.FILTERED, "TEST") not in engine.open_positions
    assert engine.avoided_count == 1


def test_filtered_book_opens_only_on_accept():
    engine = PaperTradingEngine(model=None)
    engine.on_crossover(make_event(), make_decision("ACCEPT"), PRICE, ts=SIM_START)

    assert (Book.BASIC, "TEST") in engine.open_positions
    assert (Book.FILTERED, "TEST") in engine.open_positions
    assert engine.avoided_count == 0