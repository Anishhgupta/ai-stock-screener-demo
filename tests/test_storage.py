"""Tests for the database layer and its connection to the trading engine."""
import time
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from brokers.base import Tick
from config import CONFIG
from ml.predict import Decision
from storage.db import make_engine
from storage.repository import Repository
from trading.paper_engine import Book, PaperTradingEngine

PRICE = 100.0
SIM_START = time.time() + 10 * 24 * 3600
MAX_HOLD_SECONDS = CONFIG.max_hold_minutes * 60


def make_tick(symbol, ts, ltp):
    return Tick(
        symbol=symbol, ts=ts, ltp=ltp, ltq=100, total_traded_qty=10_000,
        bid_price=ltp - 0.05, bid_qty=900_000, ask_price=ltp + 0.05, ask_qty=900_000,
    )


def decision(kind):
    if kind == "ACCEPT":
        return Decision(symbol="X", signal="BUY", probability=0.9,
                        decision="ACCEPT", reasons=["strong bids", "LTQ rising"])
    return Decision(symbol="X", signal="BUY", probability=0.3,
                    decision="AVOID", reasons=["Low model confidence"])


def run_trade(engine, symbol, start_ts, exit_price, kind="ACCEPT", hold=5.0):
    event = SimpleNamespace(symbol=symbol, signal="BUY", features=None)
    engine.on_crossover(event, decision(kind), PRICE, ts=start_ts)
    engine.on_tick(make_tick(symbol, start_ts + hold, exit_price), None)
    return start_ts + hold


def wired_engine(repo, user_id):
    return PaperTradingEngine(
        model=None,
        on_close=lambda pos: repo.record_closed_position(user_id, pos),
        on_log=lambda entry: repo.record_trade_log(user_id, entry),
    )


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def repo(db_url):
    return Repository(make_engine(db_url))


def test_ensure_user_is_idempotent_and_distinct_per_name(repo):
    first = repo.ensure_user("local")
    assert repo.ensure_user("local") == first
    assert repo.ensure_user("someone-else") != first


def test_sqlite_uses_wal_mode(db_url):
    with make_engine(db_url).connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"


def test_closed_positions_round_trip_through_the_engine(repo):
    uid = repo.ensure_user()
    engine = wired_engine(repo, uid)
    win_price = PRICE * (1 + CONFIG.target_pct + 0.001)
    run_trade(engine, "WIN", SIM_START, win_price)

    rows = repo.closed_positions(uid, "basic")
    assert len(rows) == 1
    row = rows[0]
    assert (row.symbol, row.signal, row.book) == ("WIN", "BUY", "basic")
    assert row.exit_reason == "Target hit"
    assert row.pnl > 0
    assert row.exit_price == pytest.approx(win_price)
    # The ACCEPTed trade also opened (and closed) in the filtered book.
    assert len(repo.closed_positions(uid, "filtered")) == 1
    assert repo.closed_positions(uid, "filtered")[0].entry_prob == pytest.approx(0.9)


def test_database_summary_matches_engine_summary(repo):
    uid = repo.ensure_user()
    engine = wired_engine(repo, uid)
    win_price = PRICE * (1 + CONFIG.target_pct + 0.001)
    loss_price = PRICE * (1 - CONFIG.stop_loss_pct - 0.001)

    ts = SIM_START
    ts = run_trade(engine, "WIN", ts, win_price)
    ts = run_trade(engine, "LOSS", ts, loss_price)
    ts = run_trade(engine, "FLAT", ts, PRICE, hold=MAX_HOLD_SECONDS + 1)

    for book in (Book.BASIC, Book.FILTERED):
        expected = engine.summary(book)
        actual = repo.summary(uid, book.value)
        for key in ("total_trades", "wins", "losses", "breakeven"):
            assert actual[key] == expected[key], (book, key)
        assert actual["win_rate"] == pytest.approx(expected["win_rate"])
        assert actual["total_pnl"] == pytest.approx(expected["total_pnl"])
    assert repo.summary(uid, "basic")["breakeven"] == 1


def test_trade_log_persists_opened_avoided_and_closed_events(repo):
    uid = repo.ensure_user()
    engine = wired_engine(repo, uid)
    win_price = PRICE * (1 + CONFIG.target_pct + 0.001)
    ts = run_trade(engine, "A", SIM_START, win_price, kind="ACCEPT")
    run_trade(engine, "B", ts, win_price, kind="AVOID")

    stored = repo.trade_log(uid)
    assert len(stored) == len(engine.trade_log)
    avoided = [r for r in stored if r["outcome"] == "AVOIDED"]
    assert len(avoided) == 1
    assert avoided[0]["reasons"] == ["Low model confidence"]
    assert any(r["outcome"] == "WIN" and r["pnl"] > 0 for r in stored)


def test_users_cannot_see_each_others_trades(repo):
    alice, bob = repo.ensure_user("alice"), repo.ensure_user("bob")
    engine = wired_engine(repo, alice)
    run_trade(engine, "A", SIM_START, PRICE * (1 + CONFIG.target_pct + 0.001))

    assert repo.summary(alice, "basic")["total_trades"] == 1
    assert repo.summary(bob, "basic")["total_trades"] == 0
    assert repo.closed_positions(bob) == []
    assert repo.trade_log(bob) == []


def test_history_survives_a_restart(db_url):
    first = Repository(make_engine(db_url))
    uid = first.ensure_user()
    run_trade(wired_engine(first, uid), "A", SIM_START,
              PRICE * (1 + CONFIG.target_pct + 0.001))

    second = Repository(make_engine(db_url))  # simulates a fresh process
    assert second.ensure_user() == uid
    assert second.summary(uid, "basic")["total_trades"] == 1


def test_a_failing_database_never_stops_trading():
    def broken(_payload):
        raise RuntimeError("database is down")

    engine = PaperTradingEngine(model=None, on_close=broken, on_log=broken)
    run_trade(engine, "A", SIM_START, PRICE * (1 + CONFIG.target_pct + 0.001))

    assert len(engine.closed_positions) == 2   # basic + filtered both closed
    assert engine.open_positions == {}


def test_recording_an_unclosed_position_is_rejected(repo):
    uid = repo.ensure_user()
    open_pos = SimpleNamespace(pnl=None, exit_price=None, exit_ts=None)
    with pytest.raises(ValueError):
        repo.record_closed_position(uid, open_pos)