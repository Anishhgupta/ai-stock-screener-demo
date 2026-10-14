"""
The deterioration monitor must not judge a position on one tick of noise.

Found on a live Fyers run: all three ACCEPTed trades were closed by the
monitor in the same second they opened (model confidence "fell" 19%, 58%
and 86% between consecutive ticks), so the Filtered book never really
held a position.
"""
import time
from types import SimpleNamespace

import pytest

import main
import trading.paper_engine as pe
from brokers.base import Tick
from config import CONFIG
from ml.predict import Decision
from trading.paper_engine import Book, PaperTradingEngine

PRICE = 100.0
START = 1_800_000_000.0
GRACE = 120.0
CONFIRM = 30.0


def make_tick(ts, ltp=PRICE, symbol="AAA"):
    return Tick(symbol=symbol, ts=ts, ltp=ltp, ltq=100, total_traded_qty=10_000,
                bid_price=ltp - 0.05, bid_qty=5000, ask_price=ltp + 0.05, ask_qty=5000)


def accepted_engine():
    engine = PaperTradingEngine(model=None, monitor_grace_seconds=GRACE,
                                monitor_confirm_seconds=CONFIRM)
    event = SimpleNamespace(symbol="AAA", signal="BUY", features=None)
    decision = Decision(symbol="AAA", signal="BUY", probability=0.9,
                        decision="ACCEPT", reasons=["strong bids"])
    engine.on_crossover(event, decision, PRICE, ts=START)
    return engine


@pytest.fixture
def warning(monkeypatch):
    """Controls what the monitor reports: warning['flag'] = 'text' or None."""
    state = {"flag": "model fell"}
    monkeypatch.setattr(pe, "check_deterioration", lambda *a, **k: state["flag"])
    return state


LIVE = object()  # any non-None live feature snapshot


def filtered_open(engine):
    return (Book.FILTERED, "AAA") in engine.open_positions


def test_defaults_are_a_two_minute_grace_and_thirty_second_confirmation():
    engine = PaperTradingEngine(model=None)
    assert engine.monitor_grace_seconds == 120.0
    assert engine.monitor_confirm_seconds == 30.0


def test_no_deterioration_exit_inside_the_grace_period(warning):
    engine = accepted_engine()
    for seconds in (0, 1, 5, 60, GRACE - 1):
        engine.on_tick(make_tick(START + seconds), LIVE)
    assert filtered_open(engine)
    assert engine.open_positions[(Book.FILTERED, "AAA")].monitor_flag is None


def test_warning_must_persist_before_the_position_is_closed(warning):
    engine = accepted_engine()
    engine.on_tick(make_tick(START + GRACE + 1), LIVE)           # warning first seen
    engine.on_tick(make_tick(START + GRACE + 1 + CONFIRM - 1), LIVE)
    assert filtered_open(engine)

    engine.on_tick(make_tick(START + GRACE + 1 + CONFIRM), LIVE)
    assert not filtered_open(engine)
    closed = [p for p in engine.closed_positions if p.book == Book.FILTERED][0]
    assert closed.exit_reason == "Deterioration exit (model fell)"


def test_a_cleared_warning_restarts_the_confirmation_timer(warning):
    engine = accepted_engine()
    t0 = START + GRACE + 1
    engine.on_tick(make_tick(t0), LIVE)                  # warning starts
    warning["flag"] = None
    engine.on_tick(make_tick(t0 + 10), LIVE)             # warning clears
    warning["flag"] = "model fell"
    engine.on_tick(make_tick(t0 + 20), LIVE)             # warning starts again
    engine.on_tick(make_tick(t0 + 20 + CONFIRM - 1), LIVE)
    assert filtered_open(engine)                         # timer restarted at t0+20
    engine.on_tick(make_tick(t0 + 20 + CONFIRM), LIVE)
    assert not filtered_open(engine)


def test_a_persistent_warning_does_not_close_a_position_that_is_in_profit(warning):
    engine = accepted_engine()
    in_profit = PRICE * (1 + CONFIG.target_pct / 2)
    t0 = START + GRACE + 1
    engine.on_tick(make_tick(t0, in_profit), LIVE)
    engine.on_tick(make_tick(t0 + CONFIRM + 10, in_profit), LIVE)
    assert filtered_open(engine)


def test_stop_loss_is_never_delayed_by_the_grace_period(warning):
    engine = accepted_engine()
    crash = PRICE * (1 - CONFIG.stop_loss_pct - 0.001)
    engine.on_tick(make_tick(START + 5, crash), LIVE)
    reasons = {p.exit_reason for p in engine.closed_positions}
    assert reasons == {"Stop-loss hit"}


def test_the_basic_book_ignores_the_monitor_entirely(warning):
    engine = accepted_engine()
    t0 = START + GRACE + 1
    engine.on_tick(make_tick(t0), LIVE)
    engine.on_tick(make_tick(t0 + CONFIRM), LIVE)
    assert (Book.BASIC, "AAA") in engine.open_positions


def test_main_stamps_entries_with_the_tick_clock(monkeypatch, tmp_path):
    class FakeModel:
        is_available = True

        def decide(self, event):
            return Decision(symbol=event.symbol, signal=event.signal, probability=0.9,
                            decision="ACCEPT", reasons=["fake"])

    class FakeEngine:
        def __init__(self):
            self.tick_store = SimpleNamespace(recent=lambda s, n: [])
            self.candles = SimpleNamespace(closed_candles=lambda s, n: [])
            self.smma = SimpleNamespace(get=lambda s: SimpleNamespace(
                fast=SimpleNamespace(value=1.0), slow=SimpleNamespace(value=2.0)))

        def on_tick(self, tick):
            return SimpleNamespace(symbol=tick.symbol, signal="BUY", features=None)

    monkeypatch.setattr(main, "STATE_PATH", tmp_path / "state.json")
    app = main.App.__new__(main.App)
    app.model, app.engine = FakeModel(), FakeEngine()
    app._latest_by_symbol, app._last_state_write = {}, 0.0
    app._last_tick_at = time.time()
    app.trader = app._make_trader(False)

    app.handle_tick(make_tick(1_234_567.0))

    assert app.trader.open_positions[(Book.BASIC, "AAA")].entry_ts == 1_234_567.0
    assert app.trader.open_positions[(Book.FILTERED, "AAA")].entry_ts == 1_234_567.0