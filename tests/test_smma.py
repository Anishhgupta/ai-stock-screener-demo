"""Unit tests for the incremental SMMA and crossover detection."""
import pytest

from indicators.smma import SMMA, SMMAPair, SMMATracker


def test_smma_is_none_until_seeded_then_starts_at_simple_average():
    s = SMMA(3)
    assert s.update(10) is None
    assert not s.ready
    assert s.update(20) is None
    assert s.update(30) == pytest.approx(20.0)  # simple average of first 3
    assert s.ready


def test_smma_follows_wilder_recursion_after_seed():
    s = SMMA(3)
    for p in (10, 20, 30):
        s.update(p)
    assert s.update(35) == pytest.approx(25.0)  # 20 + (35 - 20) / 3
    assert s.update(25) == pytest.approx(25.0)  # unchanged: price equals value
    assert s.update(31) == pytest.approx(27.0)  # 25 + (31 - 25) / 3


def test_smma_of_constant_prices_stays_constant():
    s = SMMA(5)
    for _ in range(20):
        s.update(42.0)
    assert s.value == pytest.approx(42.0)


def test_pair_gives_no_signal_while_seeding():
    pair = SMMAPair(2, 3)
    assert [pair.update(10) for _ in range(3)] == [None, None, None]


def test_pair_signals_buy_when_fast_crosses_above_slow():
    pair = SMMAPair(2, 3)
    for _ in range(3):
        pair.update(10)
    assert pair.update(20) == "BUY"  # fast 15.0 > slow 13.33


def test_pair_signals_sell_when_fast_crosses_below_slow():
    pair = SMMAPair(2, 3)
    for _ in range(3):
        pair.update(20)
    assert pair.update(10) == "SELL"  # fast 15.0 < slow 16.67


def test_pair_signals_only_once_per_crossover():
    pair = SMMAPair(2, 3)
    for _ in range(3):
        pair.update(10)
    assert pair.update(20) == "BUY"
    assert pair.update(20) is None  # still above, no new crossover


def test_pair_spread_is_none_until_both_ready_then_correct():
    pair = SMMAPair(2, 3)
    assert pair.spread is None and pair.spread_pct is None
    for p in (10, 10, 10, 20):
        pair.update(p)
    assert pair.spread == pytest.approx(15.0 - 40 / 3)
    assert pair.spread_pct == pytest.approx(0.125)


def test_tracker_keeps_independent_state_per_symbol():
    tracker = SMMATracker(2, 3)
    for _ in range(3):
        tracker.update("UP", 10)
        tracker.update("DOWN", 20)
    assert tracker.update("UP", 20) == "BUY"
    assert tracker.update("DOWN", 10) == "SELL"
    assert tracker.get("UP") is tracker.get("UP")
    assert tracker.get("UP") is not tracker.get("DOWN")