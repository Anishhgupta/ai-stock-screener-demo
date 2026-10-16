"""
The engine records every crossover and re-scores the setup on later ticks,
so the stability of the model's verdict can be measured.
"""
import time
from types import SimpleNamespace

import pytest

import main
from brokers.base import Tick
from config import CONFIG
from features.engineering import FEATURE_ORDER
from ml.predict import Decision
from storage.db import make_engine
from storage.repository import Repository
from trading.paper_engine import Book, PaperTradingEngine

PRICE = 100.0
START = 1_800_000_000.0


def make_tick(ts, ltp=PRICE, symbol="AAA"):
    return Tick(symbol=symbol, ts=ts, ltp=ltp, ltq=100, total_traded_qty=10_000,
                bid_price=ltp - 0.05, bid_qty=5000, ask_price=ltp + 0.05, ask_qty=5000)


class FakeFeat:
    bid_ask_imbalance = 0.1   # main.py shows it on the dashboard row

    def to_feature_vector(self):
        return [float(i + 1) for i in range(len(FEATURE_ORDER))]


class FakeModel:
    def __init__(self, scores=None, fail=False):
        self.scores, self.fail, self.calls = list(scores or []), fail, 0

    def score(self, feat):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model exploded")
        return self.scores.pop(0) if self.scores else 0.5


def decision(kind="ACCEPT", probability=0.9):
    return Decision(symbol="AAA", signal="BUY", probability=probability,
                    decision=kind, reasons=["test"])


def event(features=FakeFeat()):
    return SimpleNamespace(symbol="AAA", signal="BUY", features=features)


class Recorder:
    def __init__(self):
        self.events, self.samples = [], []


def traced_engine(model=None, **kwargs):
    rec = Recorder()
    engine = PaperTradingEngine(
        model=model, on_crossover_event=rec.events.append, on_sample=rec.samples.append,
        monitor_grace_seconds=10**9, **kwargs)
    return engine, rec


# ---------------------------------------------------------- crossover events

def test_every_crossover_is_recorded_with_its_features_and_verdict():
    engine, rec = traced_engine(FakeModel())
    engine.on_crossover(event(), decision("ACCEPT", 0.91), PRICE, ts=START)

    (record,) = rec.events
    assert (record.symbol, record.signal, record.ts, record.price) == ("AAA", "BUY", START, PRICE)
    assert (record.decision, record.probability) == ("ACCEPT", 0.91)
    assert list(record.features) == FEATURE_ORDER
    assert record.features[FEATURE_ORDER[0]] == 1.0


def test_avoided_and_skipped_crossovers_are_recorded_too():
    engine, rec = traced_engine(FakeModel())
    engine.on_crossover(event(), decision("AVOID", 0.2), PRICE, ts=START)
    engine.on_crossover(event(), decision("AVOID", 0.3), PRICE, ts=START + 60)  # skipped: open
    assert [r.probability for r in rec.events] == [0.2, 0.3]


def test_missing_features_are_recorded_as_an_empty_dict():
    engine, rec = traced_engine(FakeModel())
    engine.on_crossover(event(features=None), decision(), PRICE, ts=START)
    assert rec.events[0].features == {}


# ------------------------------------------------------------- score samples

def test_the_setup_is_re_scored_every_interval_after_the_crossover():
    model = FakeModel(scores=[0.80, 0.60, 0.40])
    engine, rec = traced_engine(model, sample_interval_seconds=5.0)
    engine.on_crossover(event(), decision(), PRICE, ts=START)

    for second in range(1, 17):                      # one tick per second
        engine.on_tick(make_tick(START + second), FakeFeat())

    assert [s.seconds_since_event for s in rec.samples] == [5.0, 10.0, 15.0]
    assert [s.score for s in rec.samples] == [0.80, 0.60, 0.40]
    assert all(s.event_ts == START and s.symbol == "AAA" and s.price == PRICE
               for s in rec.samples)


def test_sampling_stops_after_the_window():
    engine, rec = traced_engine(FakeModel(), sample_interval_seconds=5.0,
                                sample_window_seconds=12.0)
    engine.on_crossover(event(), decision(), PRICE, ts=START)
    for second in range(1, 40):
        engine.on_tick(make_tick(START + second), FakeFeat())
    assert max(s.seconds_since_event for s in rec.samples) <= 12.0
    assert len(rec.samples) == 2


def test_sampling_stops_once_every_position_in_the_symbol_has_closed():
    engine, rec = traced_engine(FakeModel(), sample_interval_seconds=5.0)
    engine.on_crossover(event(), decision("AVOID", 0.2), PRICE, ts=START)
    engine.on_tick(make_tick(START + 6), FakeFeat())
    assert len(rec.samples) == 1

    engine.close_all()
    for second in range(7, 40):
        engine.on_tick(make_tick(START + second), FakeFeat())
    assert len(rec.samples) == 1


def test_a_skipped_second_crossover_does_not_restart_the_sampling():
    engine, rec = traced_engine(FakeModel(), sample_interval_seconds=5.0)
    engine.on_crossover(event(), decision("AVOID", 0.2), PRICE, ts=START)
    engine.on_crossover(event(), decision("AVOID", 0.3), PRICE, ts=START + 60)
    engine.on_tick(make_tick(START + 70), FakeFeat())
    assert rec.samples[0].event_ts == START


def test_nothing_is_sampled_without_a_callback_a_model_or_live_features():
    engine = PaperTradingEngine(model=FakeModel(), monitor_grace_seconds=10**9)   # no callbacks
    engine.on_crossover(event(), decision(), PRICE, ts=START)
    engine.on_tick(make_tick(START + 30), FakeFeat())
    assert engine.model.calls == 0

    engine, rec = traced_engine(model=None)
    engine.on_crossover(event(), decision(), PRICE, ts=START)
    engine.on_tick(make_tick(START + 30), FakeFeat())
    assert rec.samples == []

    model = FakeModel()
    engine, rec = traced_engine(model)
    engine.on_crossover(event(), decision(), PRICE, ts=START)
    engine.on_tick(make_tick(START + 30), None)
    assert rec.samples == [] and model.calls == 0


def test_a_failing_model_switches_sampling_off_but_trading_continues():
    engine, rec = traced_engine(FakeModel(fail=True), sample_interval_seconds=5.0)
    engine.on_crossover(event(), decision("AVOID", 0.2), PRICE, ts=START)
    for second in (6, 12, 18):
        engine.on_tick(make_tick(START + second), FakeFeat())
    assert engine.model.calls == 1          # tried once, then switched off
    assert rec.samples == []

    engine.on_tick(make_tick(START + 20, PRICE * (1 + CONFIG.target_pct + 0.001)), FakeFeat())
    assert engine.open_positions == {}      # target exit still worked


def test_a_failing_callback_never_stops_trading():
    def broken(_):
        raise RuntimeError("database is down")

    engine = PaperTradingEngine(model=FakeModel(), on_crossover_event=broken,
                                on_sample=broken, monitor_grace_seconds=10**9)
    engine.on_crossover(event(), decision(), PRICE, ts=START)
    engine.on_tick(make_tick(START + 6), FakeFeat())
    assert (Book.BASIC, "AAA") in engine.open_positions


# ------------------------------------------------------------------ storage

@pytest.fixture
def repo(tmp_path):
    return Repository(make_engine(f"sqlite:///{tmp_path / 'trace.db'}"))


def test_events_and_samples_round_trip_through_the_database(repo):
    uid = repo.ensure_user()
    engine = PaperTradingEngine(
        model=FakeModel(scores=[0.7]), monitor_grace_seconds=10**9,
        on_crossover_event=lambda r: repo.record_crossover(uid, r),
        on_sample=lambda s: repo.record_score_sample(uid, s))
    engine.on_crossover(event(), decision("ACCEPT", 0.95), PRICE, ts=START)
    engine.on_tick(make_tick(START + 6, PRICE + 1), FakeFeat())

    (stored_event,) = repo.crossover_events(uid)
    assert stored_event["ts"] == START and stored_event["probability"] == 0.95
    assert stored_event["features"][FEATURE_ORDER[2]] == 3.0
    (stored_sample,) = repo.score_samples(uid)
    assert stored_sample["event_ts"] == START
    assert stored_sample["score"] == 0.7 and stored_sample["price"] == PRICE + 1


def test_traces_are_private_to_each_user(repo):
    alice, bob = repo.ensure_user("alice"), repo.ensure_user("bob")
    engine = PaperTradingEngine(model=None, on_crossover_event=lambda r: repo.record_crossover(alice, r))
    engine.on_crossover(event(), decision(), PRICE, ts=START)
    assert len(repo.crossover_events(alice)) == 1
    assert repo.crossover_events(bob) == [] and repo.score_samples(bob) == []


# ------------------------------------------------------------------ main.py

def test_main_records_crossovers_and_score_samples(monkeypatch, tmp_path):
    class Model:
        is_available = True

        def decide(self, evt):
            return Decision(symbol=evt.symbol, signal=evt.signal, probability=0.9,
                            decision="ACCEPT", reasons=["fake"])

        def score(self, feat):
            return 0.55

    class Engine:
        def __init__(self):
            self.tick_store = SimpleNamespace(recent=lambda s, n: [])
            self.candles = SimpleNamespace(closed_candles=lambda s, n: [])
            self.smma = SimpleNamespace(get=lambda s: SimpleNamespace(
                fast=SimpleNamespace(value=1.0), slow=SimpleNamespace(value=2.0)))
            self.fire = True

        def on_tick(self, tick):
            if self.fire:
                self.fire = False
                return SimpleNamespace(symbol=tick.symbol, signal="BUY", features=FakeFeat())

    monkeypatch.setattr(main, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(main, "build_feature_snapshot", lambda **kw: FakeFeat())
    app = main.App.__new__(main.App)
    app.model, app.engine = Model(), Engine()
    app._latest_by_symbol, app._last_state_write = {}, 0.0
    app._last_tick_at = time.time()
    app.trader = app._make_trader(True, f"sqlite:///{tmp_path / 'main.db'}")

    for second in range(0, 8):
        app.handle_tick(make_tick(START + second))

    (stored_event,) = app.repo.crossover_events(app.user_id)
    assert stored_event["decision"] == "ACCEPT"
    samples = app.repo.score_samples(app.user_id)
    assert samples and samples[0]["score"] == 0.55