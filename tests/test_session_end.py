"""
Positions still open when a session ends must be closed at the last price
and saved, not silently discarded. Found on live runs: the 14 and 15
October sessions each ended with positions that never reached the history.
"""
import threading
import time
from types import SimpleNamespace

import pytest

import main
from brokers.base import Tick
from config import CONFIG
from ml.predict import Decision
from storage.db import make_engine
from storage.repository import Repository
from trading.paper_engine import SESSION_END_REASON, Book, PaperTradingEngine

PRICE = 100.0
START = 1_800_000_000.0
AFTER_CLOSE = START + 2 * 3600    # 15:30 IST: the market has closed


def make_tick(ts, ltp, symbol="AAA"):
    return Tick(symbol=symbol, ts=ts, ltp=ltp, ltq=100, total_traded_qty=10_000,
                bid_price=ltp - 0.05, bid_qty=5000, ask_price=ltp + 0.05, ask_qty=5000)


def decision(kind="ACCEPT", symbol="AAA", signal="BUY"):
    return Decision(symbol=symbol, signal=signal, probability=0.9 if kind == "ACCEPT" else 0.2,
                    decision=kind, reasons=["test"])


def open_trade(engine, symbol="AAA", signal="BUY", kind="ACCEPT", ts=START):
    event = SimpleNamespace(symbol=symbol, signal=signal, features=None)
    engine.on_crossover(event, decision(kind, symbol, signal), PRICE, ts=ts)


def small_move(fraction_of_stop):
    return PRICE * (1 + CONFIG.stop_loss_pct * fraction_of_stop)


# ------------------------------------------------------------- the engine

def test_the_session_end_reason_keeps_a_stable_prefix_for_later_analysis():
    """Analyses filter on this prefix to separate marked closes from real exits."""
    assert SESSION_END_REASON.startswith("Session end")


def test_close_all_closes_every_position_at_the_last_price_seen():
    engine = PaperTradingEngine(model=None)
    open_trade(engine, "AAA", "BUY")
    open_trade(engine, "BBB", "SELL", kind="AVOID")
    engine.on_tick(make_tick(START + 10, small_move(0.5), "AAA"), None)   # BUY gains
    engine.on_tick(make_tick(START + 10, small_move(0.5), "BBB"), None)   # SELL loses

    assert engine.close_all() == 3   # AAA in both books + BBB in basic

    assert engine.open_positions == {}
    by_key = {(p.book, p.symbol): p for p in engine.closed_positions}
    assert {p.exit_reason for p in by_key.values()} == {SESSION_END_REASON}
    assert by_key[(Book.BASIC, "AAA")].exit_price == pytest.approx(small_move(0.5))
    assert by_key[(Book.BASIC, "AAA")].pnl > 0     # BUY, price rose
    assert by_key[(Book.BASIC, "BBB")].pnl < 0     # SELL, price rose


def test_the_most_recent_tick_price_is_the_one_used():
    engine = PaperTradingEngine(model=None)
    open_trade(engine, "AAA", "BUY", kind="AVOID")
    engine.on_tick(make_tick(START + 5, small_move(0.2)), None)
    engine.on_tick(make_tick(START + 9, small_move(-0.2)), None)

    engine.close_all()
    pos = engine.closed_positions[0]
    assert pos.exit_price == pytest.approx(small_move(-0.2))
    assert pos.exit_ts == START + 9
    assert pos.pnl < 0


def test_a_position_with_no_tick_since_entry_closes_flat_at_its_entry_price():
    engine = PaperTradingEngine(model=None)
    open_trade(engine, kind="AVOID")
    engine.close_all()
    pos = engine.closed_positions[0]
    assert pos.exit_price == PRICE
    assert pos.pnl == 0


def test_close_all_is_safe_to_call_twice_and_with_nothing_open():
    engine = PaperTradingEngine(model=None)
    assert engine.close_all() == 0

    open_trade(engine, kind="AVOID")
    assert engine.close_all() == 1
    assert engine.close_all() == 0
    assert len(engine.closed_positions) == 1
    assert len([e for e in engine.trade_log if e.outcome in ("WIN", "LOSS")]) == 1


def test_session_end_closes_are_saved_through_the_persistence_callbacks(tmp_path):
    repo = Repository(make_engine(f"sqlite:///{tmp_path / 'end.db'}"))
    uid = repo.ensure_user()
    engine = PaperTradingEngine(
        model=None,
        on_close=lambda pos: repo.record_closed_position(uid, pos),
        on_log=lambda entry: repo.record_trade_log(uid, entry))
    open_trade(engine)
    engine.on_tick(make_tick(START + 10, small_move(0.5)), None)
    engine.close_all()

    saved = repo.closed_positions(uid)
    assert len(saved) == 2
    assert {p.exit_reason for p in saved} == {SESSION_END_REASON}
    assert any(r["outcome"] == "WIN" for r in repo.trade_log(uid))


@pytest.mark.parametrize("call", ["close_all", "on_tick"])
def test_engine_calls_wait_for_the_engine_lock(call):
    """Ticks (broker thread) and close_all (main thread) must not overlap."""
    engine = PaperTradingEngine(model=None)
    open_trade(engine)
    actions = {
        "close_all": lambda: engine.close_all(),
        "on_tick": lambda: engine.on_tick(make_tick(START + 1, PRICE), None),
    }
    finished = threading.Event()

    def run():
        actions[call]()
        finished.set()

    with engine._lock:
        worker = threading.Thread(target=run)
        worker.start()
        assert not finished.wait(0.3)    # blocked while the lock is held
    assert finished.wait(5)              # runs once the lock is released
    worker.join()


# ---------------------------------------------------------------- main.py

class FakeModel:
    is_available = True

    def decide(self, event):
        return decision("ACCEPT", event.symbol, event.signal)


class FakeEngine:
    def __init__(self):
        self.tick_store = SimpleNamespace(recent=lambda s, n: [])
        self.candles = SimpleNamespace(closed_candles=lambda s, n: [])
        self.smma = SimpleNamespace(get=lambda s: SimpleNamespace(
            fast=SimpleNamespace(value=1.0), slow=SimpleNamespace(value=2.0)))
        self.fire = True

    def on_tick(self, tick):
        if self.fire:
            self.fire = False
            return SimpleNamespace(symbol=tick.symbol, signal="BUY", features=None)
        return None


@pytest.fixture
def app(monkeypatch, tmp_path):
    reports = []
    monkeypatch.setattr(main, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(main, "save_report",
                        lambda trader, path: reports.append(len(trader.closed_positions)) or {})
    monkeypatch.setattr(main, "print_report", lambda report: None)

    a = main.App.__new__(main.App)
    a.model, a.engine = FakeModel(), FakeEngine()
    a._latest_by_symbol, a._last_state_write = {}, 0.0
    a._last_tick_at = time.time()
    a.trader = a._make_trader(True, f"sqlite:///{tmp_path / 'main.db'}")
    a.reports = reports
    a.handle_tick(make_tick(AFTER_CLOSE, PRICE))
    a.handle_tick(make_tick(AFTER_CLOSE + 5, small_move(0.5)))
    return a


def test_finishing_a_session_closes_and_saves_open_positions(app, tmp_path):
    assert len(app.trader.open_positions) == 2

    app._finish()

    assert app.trader.open_positions == {}
    saved = app.repo.closed_positions(app.user_id)
    assert len(saved) == 2
    assert {p.exit_reason for p in saved} == {SESSION_END_REASON}


def test_the_report_is_written_after_the_positions_are_closed(app):
    app._finish()
    assert app.reports == [2]    # the report saw both session-end trades


def test_the_final_dashboard_state_shows_no_open_positions(app):
    app._finish()
    snap = app.repo.load_live_snapshot(app.user_id)
    assert snap["open_positions"] == []
    assert snap["basic_summary"]["total_trades"] == 1


def test_finishing_twice_is_harmless(app):
    app._finish()          # the Ctrl+C handler...
    app._finish()          # ...and the run() finally-block both call it
    assert len(app.repo.closed_positions(app.user_id)) == 2


def test_a_failing_final_state_write_does_not_block_the_report(app, monkeypatch):
    def explode():
        raise OSError("disk full")

    monkeypatch.setattr(app, "_write_state", explode)
    app._finish()
    assert app.reports == [2]