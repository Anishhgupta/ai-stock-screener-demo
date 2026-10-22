"""
A crash or restart must not lose open positions or the carried decisions.

Found on 15 October: a PC restart in the middle of a session discarded every
open position (they only lived in memory) and blanked all the dashboard's
signals until each symbol's next crossover.
"""
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import main
from brokers.base import Tick
from config import CONFIG
from features.engineering import FEATURE_ORDER
from ml.predict import Decision
from storage.db import make_engine
from storage.repository import Repository
from trading.paper_engine import (
    Book, OpenPositionRecord, PaperTradingEngine, StoredFeatures,
)

START = 1_800_000_000.0     # 13:30 IST on 15 Jan 2027: mid-session
AFTER_CLOSE = START + 2 * 3600   # 15:30 IST: the market has closed
YESTERDAY = START - 86_400
PRICE = 100.0
QTY = int(CONFIG.capital_per_trade // PRICE)


def make_tick(ts, ltp=PRICE, symbol="AAA"):
    return Tick(symbol=symbol, ts=ts, ltp=ltp, ltq=100, total_traded_qty=10_000,
                bid_price=ltp - 0.05, bid_qty=5000, ask_price=ltp + 0.05, ask_qty=5000)


class FakeFeat:
    bid_ask_imbalance = 0.1

    def to_feature_vector(self):
        return [float(i + 1) for i in range(len(FEATURE_ORDER))]


def decision(kind="ACCEPT", probability=0.9):
    return Decision(symbol="AAA", signal="BUY", probability=probability,
                    decision=kind, reasons=["test"])


def row(book="basic", symbol="AAA", entry_ts=START, entry_price=PRICE, signal="BUY",
        entry_prob=None, features=None):
    return {"book": book, "symbol": symbol, "signal": signal, "qty": QTY,
            "entry_price": entry_price, "entry_ts": entry_ts,
            "entry_prob": entry_prob, "features": features or {}}


def record(**kw):
    r = row(**kw)
    return OpenPositionRecord(book=r["book"], symbol=r["symbol"], signal=r["signal"],
                              qty=r["qty"], entry_price=r["entry_price"], entry_ts=r["entry_ts"],
                              entry_prob=r["entry_prob"], features=r["features"])


# ------------------------------------------------------------------ engine

def test_every_real_open_is_reported_with_everything_needed_to_restore_it():
    saved = []
    engine = PaperTradingEngine(model=None, on_open=saved.append)
    event = SimpleNamespace(symbol="AAA", signal="BUY", features=FakeFeat())
    engine.on_crossover(event, decision("ACCEPT", 0.9), PRICE, ts=START)

    basic, filtered = saved
    assert (basic.book, filtered.book) == ("basic", "filtered")
    assert (basic.qty, basic.entry_price, basic.entry_ts) == (QTY, PRICE, START)
    assert (basic.entry_prob, filtered.entry_prob) == (None, 0.9)
    assert list(basic.features) == FEATURE_ORDER and basic.features[FEATURE_ORDER[4]] == 5.0


def test_a_skipped_entry_is_not_reported_as_an_open_position():
    saved = []
    engine = PaperTradingEngine(model=None, on_open=saved.append)
    event = SimpleNamespace(symbol="AAA", signal="BUY", features=FakeFeat())
    engine.on_crossover(event, decision("AVOID", 0.2), PRICE, ts=START)
    engine.on_crossover(event, decision("AVOID", 0.2), PRICE, ts=START + 60)   # skipped
    assert len(saved) == 1


def test_restoring_does_not_save_again_and_blocks_a_duplicate_entry():
    saved = []
    engine = PaperTradingEngine(model=None, on_open=saved.append)

    assert engine.restore_position(row()) is not None
    assert engine.restore_position(row()) is None          # already open
    assert saved == []

    event = SimpleNamespace(symbol="AAA", signal="SELL", features=None)
    engine.on_crossover(event, decision("AVOID", 0.2), PRICE, ts=START + 5)
    assert [e.outcome for e in engine.trade_log if e.book == Book.BASIC] == ["SKIPPED"]


def test_a_restored_position_exits_like_any_other():
    closed = []
    engine = PaperTradingEngine(model=None, on_close=closed.append)
    engine.restore_position(row())
    engine.on_tick(make_tick(START + 30, PRICE * (1 + CONFIG.target_pct + 0.001)), None)

    assert engine.open_positions == {}
    assert closed[0].exit_reason == "Target hit" and closed[0].pnl > 0


def test_stored_features_keep_saved_values_and_default_the_rest_to_zero():
    feats = StoredFeatures({"bid_ask_imbalance": 0.4, "ltq_change_pct": -0.1})
    assert feats.bid_ask_imbalance == 0.4 and feats.ltq_change_pct == -0.1
    assert feats.spread_pct == 0.0
    vector = feats.to_feature_vector()
    assert len(vector) == len(FEATURE_ORDER)
    assert vector[FEATURE_ORDER.index("bid_ask_imbalance")] == 0.4


def test_the_real_monitor_can_watch_a_restored_filtered_position():
    class LowScoreModel:
        def score(self, feat):
            return 0.3

    engine = PaperTradingEngine(model=LowScoreModel(), monitor_grace_seconds=10,
                                monitor_confirm_seconds=5)
    engine.restore_position(row(book="filtered", entry_prob=0.9,
                                features={"bid_ask_imbalance": 0.5, "ltq_change_pct": 0.0}))
    live = SimpleNamespace(bid_ask_imbalance=0.5, ltq_change_pct=0.0)

    engine.on_tick(make_tick(START + 11), live)      # warning starts
    engine.on_tick(make_tick(START + 17), live)      # persisted, price flat: exit

    (closed,) = engine.closed_positions
    assert closed.exit_reason.startswith("Deterioration exit (Model confidence fell")


def test_the_newest_tick_time_is_tracked_across_symbols():
    engine = PaperTradingEngine(model=None)
    assert engine.latest_tick_ts() is None
    engine.on_tick(make_tick(START + 9, PRICE, "AAA"), None)
    engine.on_tick(make_tick(START + 3, PRICE, "BBB"), None)
    assert engine.latest_tick_ts() == START + 9


def test_close_position_closes_one_position_at_the_given_price():
    engine = PaperTradingEngine(model=None)
    engine.restore_position(row())
    assert engine.close_position(Book.BASIC, "AAA", 101.0, START + 9, "because") is True
    (pos,) = engine.closed_positions
    assert (pos.exit_price, pos.exit_ts, pos.exit_reason) == (101.0, START + 9, "because")
    assert engine.close_position(Book.BASIC, "AAA", 101.0, START + 9, "again") is False


# -------------------------------------------------------------- repository

@pytest.fixture
def repo(tmp_path):
    return Repository(make_engine(f"sqlite:///{tmp_path / 'restart.db'}"))


def test_open_positions_round_trip_with_their_features(repo):
    uid = repo.ensure_user()
    repo.save_open_position(uid, record(book="filtered", entry_prob=0.9,
                                        features={"bid_ask_imbalance": 0.5}))
    (saved,) = repo.load_open_positions(uid)
    assert (saved["book"], saved["symbol"], saved["qty"]) == ("filtered", "AAA", QTY)
    assert saved["entry_prob"] == 0.9 and saved["features"] == {"bid_ask_imbalance": 0.5}


def test_saving_the_same_book_and_symbol_again_replaces_the_stale_row(repo):
    uid = repo.ensure_user()
    repo.save_open_position(uid, record(entry_price=100.0))
    repo.save_open_position(uid, record(entry_price=105.0))
    repo.save_open_position(uid, record(book="filtered", entry_price=105.0))
    rows = repo.load_open_positions(uid)
    assert sorted((r["book"], r["entry_price"]) for r in rows) == [("basic", 105.0), ("filtered", 105.0)]


def test_open_positions_are_private_to_each_user(repo):
    alice, bob = repo.ensure_user("alice"), repo.ensure_user("bob")
    repo.save_open_position(alice, record())
    assert repo.load_open_positions(bob) == []


def test_closing_a_position_removes_its_saved_open_row(repo):
    uid = repo.ensure_user()
    engine = PaperTradingEngine(
        model=None,
        on_open=lambda rec: repo.save_open_position(uid, rec),
        on_close=lambda pos: repo.record_closed_position(uid, pos))
    event = SimpleNamespace(symbol="AAA", signal="BUY", features=FakeFeat())
    engine.on_crossover(event, decision("ACCEPT"), PRICE, ts=START)
    assert len(repo.load_open_positions(uid)) == 2

    engine.close_all()
    assert repo.load_open_positions(uid) == []
    assert len(repo.closed_positions(uid)) == 2


# ----------------------------------------------------------------- main.py

class FakeModel:
    is_available = True

    def decide(self, event):
        return decision("ACCEPT", 0.9)

    def score(self, feat):
        return 0.9


class FakeEngine:
    def __init__(self, crossovers=1):
        self.left = crossovers
        self.tick_store = SimpleNamespace(recent=lambda s, n: [])
        self.candles = SimpleNamespace(closed_candles=lambda s, n: [])
        self.smma = SimpleNamespace(get=lambda s: SimpleNamespace(
            fast=SimpleNamespace(value=1.0), slow=SimpleNamespace(value=2.0)))

    def on_tick(self, tick):
        if self.left:
            self.left -= 1
            return SimpleNamespace(symbol=tick.symbol, signal="BUY", features=FakeFeat())
        return None


def new_app(monkeypatch, tmp_path, db_url, crossovers=0, use_db=True):
    monkeypatch.setattr(main, "STATE_PATH", tmp_path / "state.json")
    app = main.App.__new__(main.App)
    app.model, app.engine = FakeModel(), FakeEngine(crossovers)
    app._latest_by_symbol, app._last_state_write = {}, 0.0
    app._last_tick_at = time.time()
    app.trader = app._make_trader(use_db, db_url)
    return app


def clock_at(monkeypatch, ts):
    """Make main.py's clock read `ts`, so snapshots carry the simulated session's time."""
    monkeypatch.setattr(main, "time", SimpleNamespace(time=lambda: ts, sleep=time.sleep))


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'app.db'}"


def test_a_crash_loses_nothing(monkeypatch, tmp_path, db_url):
    clock_at(monkeypatch, START + 5)
    first = new_app(monkeypatch, tmp_path, db_url, crossovers=1)
    first.handle_tick(make_tick(START))                 # opens Basic + Filtered
    first._write_state()                                # the dashboard snapshot
    del first                                           # crash: no _finish, nothing closed

    second = new_app(monkeypatch, tmp_path, db_url)     # restarted process, same database
    second._restore_after_restart(now=START + 600)

    assert sorted(k[0].value for k in second.trader.open_positions) == ["basic", "filtered"]
    restored = second.trader.open_positions[(Book.FILTERED, "AAA")]
    assert (restored.entry_price, restored.qty, restored.entry_ts) == (PRICE, QTY, START)
    assert restored.entry_prob == 0.9
    assert restored.entry_feat.bid_ask_imbalance == 5.0       # saved entry features came back
    carried = second._latest_by_symbol["AAA"]
    assert (carried["signal"], carried["decision"], carried["probability"]) == ("BUY", "ACCEPT", 0.9)

    second.handle_tick(make_tick(START + 660, PRICE * (1 + CONFIG.target_pct + 0.001)))
    assert second.trader.open_positions == {}
    assert len(second.repo.closed_positions(second.user_id)) == 2
    assert second.repo.load_open_positions(second.user_id) == []
    # the carried decision survives the first ordinary tick after the restart
    assert second._latest_by_symbol["AAA"]["decision"] == "ACCEPT"


def test_positions_left_open_on_an_earlier_day_are_closed_at_the_last_snapshot_price(
        monkeypatch, tmp_path, db_url):
    app = new_app(monkeypatch, tmp_path, db_url)
    app.repo.save_open_position(app.user_id, record(entry_ts=YESTERDAY))
    app.repo.save_live_snapshot(app.user_id, {
        "updated_at": YESTERDAY + 3600,
        "symbols": {"AAA": {"ltp": 101.0, "decision": "ACCEPT", "signal": "BUY",
                            "probability": 0.9, "reasons": ["old"]}}})

    app._restore_after_restart(now=START)

    (closed,) = app.repo.closed_positions(app.user_id)
    assert closed.exit_reason == main.RECOVERED_REASON
    assert closed.exit_price == 101.0 and closed.exit_ts == YESTERDAY + 3600
    assert closed.pnl == pytest.approx((101.0 - PRICE) * QTY)
    assert app.trader.open_positions == {} and app.repo.load_open_positions(app.user_id) == []
    assert app._latest_by_symbol == {}          # yesterday's decisions are not carried over


def test_an_old_position_with_no_snapshot_price_is_marked_flat(monkeypatch, tmp_path, db_url):
    app = new_app(monkeypatch, tmp_path, db_url)
    app.repo.save_open_position(app.user_id, record(entry_ts=YESTERDAY))

    app._restore_after_restart(now=START)

    (closed,) = app.repo.closed_positions(app.user_id)
    assert closed.exit_reason == main.RECOVERED_FLAT_REASON
    assert closed.pnl == 0


def test_only_todays_positions_are_resumed_when_both_kinds_exist(monkeypatch, tmp_path, db_url):
    app = new_app(monkeypatch, tmp_path, db_url)
    app.repo.save_open_position(app.user_id, record(symbol="OLD", entry_ts=YESTERDAY))
    app.repo.save_open_position(app.user_id, record(symbol="NEW", entry_ts=START))

    app._restore_after_restart(now=START + 60)

    assert [k[1] for k in app.trader.open_positions] == ["NEW"]
    assert [p.symbol for p in app.repo.closed_positions(app.user_id)] == ["OLD"]


def test_a_bad_saved_row_does_not_stop_the_good_ones(monkeypatch, tmp_path, db_url):
    app = new_app(monkeypatch, tmp_path, db_url)
    app.repo.save_open_position(app.user_id, record(book="weird", symbol="BAD"))
    app.repo.save_open_position(app.user_id, record(symbol="GOOD"))

    app._restore_after_restart(now=START + 60)

    assert [k[1] for k in app.trader.open_positions] == ["GOOD"]


def test_restoring_without_a_database_does_nothing(monkeypatch, tmp_path):
    app = new_app(monkeypatch, tmp_path, None, use_db=False)
    app._restore_after_restart(now=START)
    assert app.trader.open_positions == {} and app._latest_by_symbol == {}


def test_the_end_of_the_trading_day_leaves_nothing_to_resume(monkeypatch, tmp_path, db_url):
    monkeypatch.setattr(main, "save_report", lambda trader, path: {})
    monkeypatch.setattr(main, "print_report", lambda report: None)
    app = new_app(monkeypatch, tmp_path, db_url, crossovers=1)
    app.handle_tick(make_tick(AFTER_CLOSE))
    assert len(app.repo.load_open_positions(app.user_id)) == 2

    app._finish()

    assert app.repo.load_open_positions(app.user_id) == []
    assert len(app.repo.closed_positions(app.user_id)) == 2


def test_stopping_in_the_middle_of_the_session_keeps_positions_for_the_next_start(
        monkeypatch, tmp_path, db_url):
    """A restart, a rebuild or a shutdown mid-session must not close anything."""
    monkeypatch.setattr(main, "save_report", lambda trader, path: {})
    monkeypatch.setattr(main, "print_report", lambda report: None)
    clock_at(monkeypatch, START + 5)
    first = new_app(monkeypatch, tmp_path, db_url, crossovers=1)
    first.handle_tick(make_tick(START))
    first._write_state()

    first._finish()                                     # a graceful stop at 13:30 IST

    assert len(first.repo.load_open_positions(first.user_id)) == 2
    assert first.repo.closed_positions(first.user_id) == []

    second = new_app(monkeypatch, tmp_path, db_url)     # the next start
    second._restore_after_restart(now=START + 600)
    assert len(second.trader.open_positions) == 2


@pytest.mark.parametrize("offset, over", [(-1, False), (0, True), (1, True)])
def test_the_session_is_over_from_1530_indian_time(monkeypatch, tmp_path, db_url, offset, over):
    app = new_app(monkeypatch, tmp_path, db_url)
    assert app._session_is_over() is False              # no tick yet
    app.trader.on_tick(make_tick(AFTER_CLOSE + offset, PRICE), None)
    assert app._session_is_over() is over


def test_the_trading_day_is_the_indian_date():
    just_before = datetime(2027, 1, 15, 18, 29, 59, tzinfo=timezone.utc).timestamp()   # 23:59:59 IST
    assert str(main._ist_date(just_before)) == "2027-01-15"
    assert str(main._ist_date(just_before + 1)) == "2027-01-16"


def test_starting_the_app_resumes_todays_saved_positions(monkeypatch, tmp_path, db_url):
    """The wiring: App.__init__ itself must call the restore (not only the method)."""
    seed = Repository(make_engine(db_url))
    uid = seed.ensure_user("local")
    seed.save_open_position(uid, record(symbol="AAA", entry_ts=time.time() - 120))

    monkeypatch.setattr(main, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(main, "CrossoverEngine", lambda: FakeEngine(0))
    monkeypatch.setattr(main, "CrossoverModel", FakeModel)
    monkeypatch.setattr(main.App, "_build_broker", lambda self, name: object())

    app = main.App("mock", use_db=True, db_url=db_url)

    assert [k[1] for k in app.trader.open_positions] == ["AAA"]