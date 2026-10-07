"""
Tests for main.py's tick handling: decisions must stay visible on the
dashboard, and trading must not depend on a model being loaded.
"""
import json
import time
from types import SimpleNamespace

import pytest

import main
from brokers.base import Tick
from config import CONFIG
from ml.predict import Decision
from trading.paper_engine import Book

PRICE = 100.0


class FakeModel:
    def __init__(self, available=True, verdict="ACCEPT", probability=0.8):
        self._available = available
        self._verdict = verdict
        self._probability = probability

    @property
    def is_available(self):
        return self._available

    def decide(self, event):
        return Decision(symbol=event.symbol, signal=event.signal,
                        probability=self._probability, decision=self._verdict,
                        reasons=[f"fake {self._verdict}"])


class FakeEngine:
    """Stands in for CrossoverEngine: hands back a scripted event per tick."""

    def __init__(self, script):
        self._script = list(script)
        self.tick_store = SimpleNamespace(recent=lambda symbol, n: [])
        self.candles = SimpleNamespace(closed_candles=lambda symbol, n: [])
        self.smma = SimpleNamespace(get=lambda symbol: SimpleNamespace(
            fast=SimpleNamespace(value=1.0), slow=SimpleNamespace(value=2.0)))

    def on_tick(self, tick):
        return self._script.pop(0) if self._script else None


def crossover(symbol="AAA", signal="BUY"):
    return SimpleNamespace(symbol=symbol, signal=signal, features=None)


def tick(symbol="AAA", ltp=PRICE):
    return Tick(
        symbol=symbol, ts=time.time(), ltp=ltp, ltq=100, total_traded_qty=10_000,
        bid_price=ltp - 0.05, bid_qty=5000, ask_price=ltp + 0.05, ask_qty=5000,
    )


def make_app(monkeypatch, tmp_path, model, script, use_db=False, db_url=None):
    monkeypatch.setattr(main, "STATE_PATH", tmp_path / "state.json")
    app = main.App.__new__(main.App)   # skip __init__: no broker, no CSV writes
    app.model = model
    app.engine = FakeEngine(script)
    app._latest_by_symbol = {}
    app._last_state_write = 0.0
    app._last_tick_at = time.time()
    app.trader = app._make_trader(use_db, db_url)
    return app


def test_decision_carries_forward_on_ordinary_ticks(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeModel(), [crossover(), None, None])

    app.handle_tick(tick())
    first = dict(app._latest_by_symbol["AAA"])
    assert first["decision"] == "ACCEPT"
    assert first["signal"] == "BUY"
    assert first["probability"] == pytest.approx(0.8)

    app.handle_tick(tick())
    app.handle_tick(tick())
    later = app._latest_by_symbol["AAA"]
    for key in ("decision", "signal", "probability", "reasons"):
        assert later[key] == first[key], key


def test_a_new_crossover_replaces_the_carried_decision(monkeypatch, tmp_path):
    model = FakeModel(verdict="ACCEPT")
    app = make_app(monkeypatch, tmp_path, model, [crossover(), None, crossover(signal="SELL")])

    app.handle_tick(tick())
    app.handle_tick(tick())
    model._verdict = "AVOID"
    app.handle_tick(tick())

    row = app._latest_by_symbol["AAA"]
    assert row["decision"] == "AVOID"
    assert row["signal"] == "SELL"


def test_carried_decisions_do_not_leak_between_symbols(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeModel(), [crossover("AAA"), None])

    app.handle_tick(tick("AAA"))
    app.handle_tick(tick("BBB"))

    assert app._latest_by_symbol["AAA"]["decision"] == "ACCEPT"
    assert app._latest_by_symbol["BBB"]["decision"] is None
    assert app._latest_by_symbol["BBB"]["reasons"] == []


def test_state_file_keeps_showing_the_decision(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeModel(), [crossover(), None])

    app.handle_tick(tick())
    app.handle_tick(tick())
    app._write_state()

    row = json.loads((tmp_path / "state.json").read_text())["symbols"]["AAA"]
    assert row["decision"] == "ACCEPT"
    assert row["probability"] == pytest.approx(0.8)
    assert row["reasons"] == ["fake ACCEPT"]


def test_basic_book_still_trades_when_no_model_is_loaded(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeModel(available=False), [crossover()])

    app.handle_tick(tick())

    assert (Book.BASIC, "AAA") in app.trader.open_positions
    assert (Book.FILTERED, "AAA") not in app.trader.open_positions
    row = app._latest_by_symbol["AAA"]
    assert row["decision"] == "AVOID"
    assert row["reasons"] == ["No trained model available"]


def test_closed_trades_reach_the_database(monkeypatch, tmp_path):
    db_url = f"sqlite:///{tmp_path / 'live.db'}"
    app = make_app(monkeypatch, tmp_path, FakeModel(), [crossover(), None],
                   use_db=True, db_url=db_url)

    app.handle_tick(tick(ltp=PRICE))
    app.handle_tick(tick(ltp=PRICE * (1 + CONFIG.target_pct + 0.001)))

    assert app.repo.summary(app.user_id, "basic")["total_trades"] == 1
    assert app.repo.summary(app.user_id, "filtered")["wins"] == 1
    assert any(r["outcome"] == "WIN" for r in app.repo.trade_log(app.user_id))


def test_running_without_the_database_still_trades(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeModel(), [crossover(), None], use_db=False)

    app.handle_tick(tick(ltp=PRICE))
    app.handle_tick(tick(ltp=PRICE * (1 + CONFIG.target_pct + 0.001)))

    assert app.repo is None
    assert len(app.trader.closed_positions) == 2