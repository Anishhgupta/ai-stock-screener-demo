"""Unit tests for tick -> 1-minute candle aggregation."""
from brokers.base import Tick
from data.resample import Candle, CandleBuilder

BASE = 1_700_000_040  # a multiple of 60
BASE_5M = 1_700_000_100  # a multiple of 300


def tick(symbol, ts, ltp, ltq=10):
    return Tick(
        symbol=symbol, ts=ts, ltp=ltp, ltq=ltq, total_traded_qty=1000,
        bid_price=ltp - 0.05, bid_qty=500, ask_price=ltp + 0.05, ask_qty=500,
    )


def test_first_tick_opens_a_candle_and_closes_nothing():
    cb = CandleBuilder(1)
    assert cb.on_tick(tick("A", BASE, 100.0)) is None
    assert cb.current_candle("A").open == 100.0
    assert cb.closed_candles("A") == []


def test_ticks_in_same_minute_update_ohlc_and_volume():
    cb = CandleBuilder(1)
    cb.on_tick(tick("A", BASE, 100.0, ltq=10))
    cb.on_tick(tick("A", BASE + 10, 105.0, ltq=5))
    cb.on_tick(tick("A", BASE + 20, 95.0, ltq=7))
    cb.on_tick(tick("A", BASE + 30, 101.0, ltq=3))

    c = cb.current_candle("A")
    assert (c.open, c.high, c.low, c.close) == (100.0, 105.0, 95.0, 101.0)
    assert c.volume == 25
    assert c.last_ltq == 3


def test_next_minute_closes_and_returns_previous_candle():
    cb = CandleBuilder(1)
    cb.on_tick(tick("A", BASE, 100.0, ltq=10))
    cb.on_tick(tick("A", BASE + 30, 103.0, ltq=5))

    closed = cb.on_tick(tick("A", BASE + 60, 110.0))
    assert closed is not None
    assert closed.minute_epoch == BASE
    assert closed.close == 103.0
    assert closed.volume == 15
    assert cb.closed_candles("A") == [closed]
    assert cb.current_candle("A").minute_epoch == BASE + 60
    assert cb.current_candle("A").open == 110.0


def test_minute_boundary_is_exact():
    cb = CandleBuilder(1)
    cb.on_tick(tick("A", BASE, 100.0))
    assert cb.on_tick(tick("A", BASE + 59.9, 101.0)) is None
    assert cb.on_tick(tick("A", BASE + 60.0, 102.0)) is not None


def test_symbols_do_not_interfere():
    cb = CandleBuilder(1)
    cb.on_tick(tick("A", BASE, 100.0))
    cb.on_tick(tick("B", BASE, 200.0))
    closed_a = cb.on_tick(tick("A", BASE + 60, 101.0))
    assert closed_a.symbol == "A"
    assert cb.closed_candles("B") == []
    assert cb.current_candle("B").open == 200.0


def test_closed_candles_returns_only_the_last_n():
    cb = CandleBuilder(1)
    for i in range(6):
        cb.on_tick(tick("A", BASE + 60 * i, 100.0 + i))
    assert len(cb.closed_candles("A")) == 5
    assert [c.minute_epoch for c in cb.closed_candles("A", 2)] == [BASE + 180, BASE + 240]


def test_five_minute_timeframe_buckets_correctly():
    cb = CandleBuilder(5)
    cb.on_tick(tick("A", BASE_5M, 100.0))
    assert cb.on_tick(tick("A", BASE_5M + 299, 101.0)) is None
    assert cb.on_tick(tick("A", BASE_5M + 300, 102.0)) is not None


def test_seed_closed_adds_history_without_touching_live_state():
    cb = CandleBuilder(1)
    for i in range(2):
        cb.seed_closed(Candle("A", BASE + 60 * i, 1, 1, 1, 1 + i, volume=1))

    assert [c.close for c in cb.closed_candles("A")] == [1, 2]
    assert cb.current_candle("A") is None
    # The first live tick must not "close" anything: no live candle existed.
    assert cb.on_tick(tick("A", BASE + 600, 100.0)) is None