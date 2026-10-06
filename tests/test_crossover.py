"""Tests for the tick -> candle -> SMMA -> crossover pipeline."""
import pytest

import signals.crossover as crossover_module
from brokers.base import Tick
from config import CONFIG
from signals.crossover import CrossoverEngine

BUCKET = CONFIG.timeframe_minutes * 60
LIVE_START = 1_800_000_000 - (1_800_000_000 % BUCKET)
SYMBOL = "TEST"


def make_tick(ts, ltp):
    return Tick(
        symbol=SYMBOL, ts=ts, ltp=ltp, ltq=100, total_traded_qty=10_000,
        bid_price=ltp - 0.05, bid_qty=5000, ask_price=ltp + 0.05, ask_qty=5000,
    )


def warm_rows(prices):
    return [[i * BUCKET, p, p, p, p, 100] for i, p in enumerate(prices)]


def warm_count():
    return CONFIG.smma_slow + 30


def gentle_slope(start, end):
    n = warm_count()
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def run_live_close(engine, price):
    """Two live ticks: the first opens a candle at `price`, the second closes it."""
    first = engine.on_tick(make_tick(LIVE_START, price))
    assert first is None  # opening a candle never produces a signal
    return engine.on_tick(make_tick(LIVE_START + BUCKET, price))


@pytest.fixture
def screen_open(monkeypatch):
    monkeypatch.setattr(crossover_module, "passes_screen", lambda *a, **k: True)


def test_warm_start_primes_smma_and_candle_history():
    engine = CrossoverEngine(persist_ticks=False)
    applied = engine.warm_start(SYMBOL, warm_rows([100.0] * warm_count()))

    assert applied == warm_count()
    assert engine.smma.get(SYMBOL).fast.ready
    assert engine.smma.get(SYMBOL).slow.ready
    assert len(engine.candles.closed_candles(SYMBOL)) == warm_count()


def test_warm_start_skips_malformed_rows():
    engine = CrossoverEngine(persist_ticks=False)
    rows = warm_rows([100.0] * warm_count()) + [[1, 2], None]
    assert engine.warm_start(SYMBOL, rows) == warm_count()


def test_buy_crossover_emits_an_event_with_features(screen_open):
    engine = CrossoverEngine(persist_ticks=False)
    engine.warm_start(SYMBOL, warm_rows(gentle_slope(101.0, 100.0)))

    event = run_live_close(engine, 150.0)

    assert event is not None
    assert event.symbol == SYMBOL
    assert event.signal == "BUY"
    assert event.features.symbol == SYMBOL


def test_sell_crossover_emits_a_sell_event(screen_open):
    engine = CrossoverEngine(persist_ticks=False)
    engine.warm_start(SYMBOL, warm_rows(gentle_slope(100.0, 101.0)))

    event = run_live_close(engine, 50.0)

    assert event is not None
    assert event.signal == "SELL"


def test_no_event_without_a_crossover(screen_open):
    engine = CrossoverEngine(persist_ticks=False)
    engine.warm_start(SYMBOL, warm_rows([100.0] * warm_count()))

    assert run_live_close(engine, 100.0) is None


def test_crossover_is_dropped_when_the_screen_rejects_it(monkeypatch):
    monkeypatch.setattr(crossover_module, "passes_screen", lambda *a, **k: False)
    engine = CrossoverEngine(persist_ticks=False)
    engine.warm_start(SYMBOL, warm_rows(gentle_slope(101.0, 100.0)))

    assert run_live_close(engine, 150.0) is None


def test_ticks_inside_one_candle_never_signal(screen_open):
    engine = CrossoverEngine(persist_ticks=False)
    engine.warm_start(SYMBOL, warm_rows(gentle_slope(101.0, 100.0)))

    for i in range(5):
        assert engine.on_tick(make_tick(LIVE_START + i, 150.0)) is None