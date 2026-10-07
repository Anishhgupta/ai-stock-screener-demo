"""Tests for the live snapshot table, the dashboard data layer and main.py's snapshot saving."""
import json
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import main
from brokers.base import Tick
from ml.predict import Decision
from storage.dashboard_data import (
    all_time_summary, closed_trades_frame, decision_log_frame,
    load_live_state, watchlist_frame,
)
from storage.db import make_engine
from storage.repository import Repository
from trading.paper_engine import Book, Position, TradeLogEntry


@pytest.fixture
def repo(tmp_path):
    return Repository(make_engine(f"sqlite:///{tmp_path / 'live.db'}"))


def state(updated_at, **extra):
    base = {"updated_at": updated_at, "symbols": {}, "open_positions": [],
            "basic_summary": {}, "filtered_summary": {}}
    base.update(extra)
    return base


def closed_position(symbol, pnl, exit_ts, book=Book.BASIC):
    return Position(
        book=book, symbol=symbol, signal="BUY", entry_price=100.0, entry_ts=exit_ts - 60,
        qty=10, entry_feat=None, status="CLOSED", exit_price=101.0, exit_ts=exit_ts,
        exit_reason="Target hit", pnl=pnl, pnl_pct=pnl / 1000,
    )


# ------------------------------------------------------- snapshot table

def test_snapshot_round_trips_and_is_overwritten_not_appended(repo):
    uid = repo.ensure_user()
    assert repo.load_live_snapshot(uid) is None

    repo.save_live_snapshot(uid, state(100.0, marker="first"))
    repo.save_live_snapshot(uid, state(200.0, marker="second"))

    snap = repo.load_live_snapshot(uid)
    assert snap["marker"] == "second"
    assert snap["updated_at"] == 200.0


def test_snapshots_are_per_user(repo):
    alice, bob = repo.ensure_user("alice"), repo.ensure_user("bob")
    repo.save_live_snapshot(alice, state(1.0, marker="alice"))
    assert repo.load_live_snapshot(bob) is None


def test_ensure_user_survives_a_startup_race(repo):
    ids, errors = [], []

    def worker():
        try:
            ids.append(repo.ensure_user("local"))
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(set(ids)) == 1


def test_recent_closed_positions_are_newest_first_and_limited(repo):
    uid = repo.ensure_user()
    for i in range(5):
        repo.record_closed_position(uid, closed_position(f"S{i}", 10.0, 1000.0 + i))

    recent = repo.recent_closed_positions(uid, limit=3)
    assert [r["symbol"] for r in recent] == ["S4", "S3", "S2"]


# ----------------------------------------------------- load_live_state

def test_load_live_state_prefers_the_fresher_source(repo, tmp_path):
    uid = repo.ensure_user()
    path = tmp_path / "state.json"

    repo.save_live_snapshot(uid, state(100.0, marker="db"))
    path.write_text(json.dumps(state(200.0, marker="file")))
    assert load_live_state(repo, uid, path)["marker"] == "file"

    repo.save_live_snapshot(uid, state(300.0, marker="db-newer"))
    assert load_live_state(repo, uid, path)["marker"] == "db-newer"


def test_load_live_state_falls_back_when_the_database_fails(tmp_path):
    class BrokenRepo:
        def load_live_snapshot(self, user_id):
            raise RuntimeError("database is locked")

    path = tmp_path / "state.json"
    path.write_text(json.dumps(state(5.0, marker="file")))
    assert load_live_state(BrokenRepo(), 1, path)["marker"] == "file"


def test_load_live_state_ignores_a_corrupt_state_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ half-written")
    assert load_live_state(None, None, path) is None
    assert load_live_state(None, None, tmp_path / "missing.json") is None


# ----------------------------------------------------------- frames

def test_watchlist_frame_joins_reasons_and_renames_columns():
    snap = state(1.0, symbols={"AAA": {
        "symbol": "AAA", "ltp": 101.5, "decision": "ACCEPT", "probability": 0.7,
        "reasons": ["LTQ rising", "Bid support strong"],
    }})
    df = watchlist_frame(snap)

    assert list(df.columns)[:3] == ["Symbol", "LTP", "SMMA20"]
    assert df.loc[0, "Reasons"] == "LTQ rising, Bid support strong"
    assert df.loc[0, "Decision"] == "ACCEPT"
    assert "Bid/Ask Imbalance" in df.columns   # missing keys become empty columns


def test_watchlist_frame_handles_empty_state():
    assert len(watchlist_frame(None)) == 0
    assert "Decision" in watchlist_frame(state(1.0)).columns


def test_closed_trades_frame_shows_newest_first_in_indian_time(repo):
    uid = repo.ensure_user()
    nine_thirty_ist = datetime(2026, 10, 5, 4, 0, tzinfo=timezone.utc).timestamp()
    repo.record_closed_position(uid, closed_position("OLD", -5.0, nine_thirty_ist - 600))
    repo.record_closed_position(uid, closed_position("NEW", 12.346, nine_thirty_ist))

    df = closed_trades_frame(repo, uid)
    assert list(df["Symbol"]) == ["NEW", "OLD"]
    assert df.loc[0, "Closed at"] == "2026-10-05 09:30:00"
    assert df.loc[0, "P/L (Rs)"] == pytest.approx(12.35)


def test_decision_log_frame_joins_reasons(repo):
    uid = repo.ensure_user()
    repo.record_trade_log(uid, TradeLogEntry(
        book=Book.FILTERED, symbol="AAA", signal="BUY", decision="AVOID",
        probability=0.3, reasons=["Low model confidence", "Depth balanced"],
        outcome="AVOIDED", ts=1_700_000_000.0,
    ))
    df = decision_log_frame(repo, uid)
    assert df.loc[0, "Reasons"] == "Low model confidence, Depth balanced"
    assert df.loc[0, "Outcome"] == "AVOIDED"


def test_history_frames_are_empty_but_well_formed_without_data(repo):
    uid = repo.ensure_user()
    assert len(closed_trades_frame(repo, uid)) == 0
    assert "P/L (Rs)" in closed_trades_frame(repo, uid).columns
    assert len(decision_log_frame(None, None)) == 0
    assert all_time_summary(None, None) == {}


def test_all_time_summary_reads_saved_history(repo):
    uid = repo.ensure_user()
    repo.record_closed_position(uid, closed_position("A", 50.0, 1000.0))
    repo.record_closed_position(uid, closed_position("B", -20.0, 1001.0))
    summary = all_time_summary(repo, uid)
    assert summary["basic"]["total_trades"] == 2
    assert summary["basic"]["win_rate"] == pytest.approx(0.5)
    assert summary["filtered"]["total_trades"] == 0


# -------------------------------------------------- main.py snapshots

class FakeModel:
    is_available = True

    def decide(self, event):
        return Decision(symbol=event.symbol, signal=event.signal, probability=0.8,
                        decision="ACCEPT", reasons=["fake ACCEPT"])


class FakeEngine:
    def __init__(self, script):
        self._script = list(script)
        self.tick_store = SimpleNamespace(recent=lambda symbol, n: [])
        self.candles = SimpleNamespace(closed_candles=lambda symbol, n: [])
        self.smma = SimpleNamespace(get=lambda symbol: SimpleNamespace(
            fast=SimpleNamespace(value=1.0), slow=SimpleNamespace(value=2.0)))

    def on_tick(self, tick):
        return self._script.pop(0) if self._script else None


def make_app(monkeypatch, tmp_path, script, use_db=True):
    monkeypatch.setattr(main, "STATE_PATH", tmp_path / "state.json")
    app = main.App.__new__(main.App)
    app.model = FakeModel()
    app.engine = FakeEngine(script)
    app._latest_by_symbol = {}
    app._last_state_write = 0.0
    app._last_tick_at = time.time()
    app.trader = app._make_trader(use_db, f"sqlite:///{tmp_path / 'main.db'}")
    return app


def tick(symbol="AAA", ltp=100.0):
    return Tick(symbol=symbol, ts=time.time(), ltp=ltp, ltq=100, total_traded_qty=10_000,
                bid_price=ltp - 0.05, bid_qty=5000, ask_price=ltp + 0.05, ask_qty=5000)


def crossover():
    return SimpleNamespace(symbol="AAA", signal="BUY", features=None)


def test_write_state_saves_the_carried_decision_to_the_database(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, [crossover(), None])
    app.handle_tick(tick())
    app.handle_tick(tick())
    app._write_state()

    snap = app.repo.load_live_snapshot(app.user_id)
    row = snap["symbols"]["AAA"]
    assert row["decision"] == "ACCEPT"
    assert row["reasons"] == ["fake ACCEPT"]
    assert snap["basic_summary"]["open_trades"] == 1


def test_a_database_failure_does_not_break_state_writing(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, [crossover()])

    def explode(user_id, state):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(app.repo, "save_live_snapshot", explode)
    app.handle_tick(tick())
    app._write_state()   # must not raise

    assert (tmp_path / "state.json").exists()


def test_no_db_mode_skips_snapshots_cleanly(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, [crossover()], use_db=False)
    app.handle_tick(tick())
    app._write_state()
    assert app.repo is None
    assert (tmp_path / "state.json").exists()