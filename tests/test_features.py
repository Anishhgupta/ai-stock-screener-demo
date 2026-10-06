"""Unit tests for feature engineering, the depth screen and signal reasons."""
import pytest

from brokers.base import Tick
from config import CONFIG
from data.resample import Candle
from features.engineering import (
    FEATURE_ORDER, FeatureSnapshot, build_feature_snapshot,
    explain_signal, passes_screen,
)
from indicators.smma import SMMAPair

BASE = 1_700_000_040


def tick(i=0, ltq=100, bid_qty=500, ask_qty=500, ltp=100.0,
         bid_price=99.95, ask_price=100.05):
    return Tick(
        symbol="TEST", ts=BASE + i, ltp=ltp, ltq=ltq, total_traded_qty=1000,
        bid_price=bid_price, bid_qty=bid_qty, ask_price=ask_price, ask_qty=ask_qty,
    )


def ready_pair():
    pair = SMMAPair(2, 3)
    for p in (100, 100, 100, 101):
        pair.update(p)
    return pair


def snapshot(ticks, candles=()):
    return build_feature_snapshot("TEST", list(ticks), list(candles), ready_pair())


def candle(close, volume=0):
    return Candle("TEST", 0, close, close, close, close, volume=volume)


def make_feat(**overrides):
    base = dict(
        symbol="TEST", ts=0.0, ltp=100.0, smma_fast=1.0, smma_slow=1.0,
        smma_spread_pct=0.0, smma_slope_fast=0.0, ltq_last=1, ltq_avg_recent=1.0,
        ltq_change_pct=0.0, ltq_acceleration=0.0, volume_last_candle=0,
        bid_price=99.95, bid_qty=500, ask_price=100.05, ask_qty=500,
        bid_ask_imbalance=0.0, imbalance_trend=0.0, spread_pct=0.0,
        price_change_pct_1m=0.0, price_change_pct_5m=0.0,
    )
    base.update(overrides)
    return FeatureSnapshot(**base)


# ---------------------------------------------------------------- snapshot

def test_snapshot_is_none_without_ticks_or_ready_smma():
    assert build_feature_snapshot("TEST", [], [], ready_pair()) is None
    assert build_feature_snapshot("TEST", [tick()], [], SMMAPair(20, 120)) is None


def test_bid_ask_imbalance_formula():
    assert snapshot([tick(bid_qty=750, ask_qty=250)]).bid_ask_imbalance == pytest.approx(0.5)
    assert snapshot([tick(bid_qty=250, ask_qty=750)]).bid_ask_imbalance == pytest.approx(-0.5)
    assert snapshot([tick(bid_qty=500, ask_qty=500)]).bid_ask_imbalance == 0.0


def test_imbalance_is_zero_when_there_is_no_depth():
    assert snapshot([tick(bid_qty=0, ask_qty=0)]).bid_ask_imbalance == 0.0


def test_flat_ltq_gives_no_change_and_no_acceleration():
    feat = snapshot([tick(i, ltq=100) for i in range(60)])
    assert feat.ltq_change_pct == pytest.approx(0.0)
    assert feat.ltq_acceleration == pytest.approx(0.0)


def test_ltq_spike_and_collapse_have_opposite_signs():
    base = [tick(i, ltq=100) for i in range(59)]
    assert snapshot(base + [tick(59, ltq=500)]).ltq_change_pct > 0
    assert snapshot(base + [tick(59, ltq=10)]).ltq_change_pct < 0


def test_rising_ltq_is_positive_acceleration():
    assert CONFIG.ltq_accel_lookback >= 3
    feat = snapshot([tick(i, ltq=100 + 10 * i) for i in range(60)])
    assert feat.ltq_acceleration > 0


def test_building_bids_give_positive_imbalance_trend():
    assert CONFIG.imbalance_lookback >= 2
    feat = snapshot([tick(i, bid_qty=300 + 15 * i, ask_qty=500) for i in range(40)])
    assert feat.imbalance_trend > 0


def test_price_change_features_come_from_candle_closes():
    candles = [candle(c) for c in (100, 100, 100, 100, 100, 120)]
    feat = snapshot([tick()], candles)
    assert feat.price_change_pct_1m == pytest.approx(0.2)   # 120 vs 100
    assert feat.price_change_pct_5m == pytest.approx(0.2)   # 120 vs 6th-from-last


def test_candle_features_degrade_to_zero_with_short_history():
    feat = snapshot([tick()], [candle(100)])
    assert feat.price_change_pct_1m == 0.0
    assert feat.price_change_pct_5m == 0.0
    assert feat.smma_slope_fast == 0.0


def test_volume_last_candle_is_taken_from_the_newest_candle():
    feat = snapshot([tick()], [candle(100, volume=5), candle(101, volume=777)])
    assert feat.volume_last_candle == 777


def test_spread_pct_is_positive_and_small_for_a_normal_book():
    assert 0 < snapshot([tick()]).spread_pct < 0.01


# ------------------------------------------------- model input contract

def test_feature_vector_matches_feature_order():
    """The model is trained on this exact ordering; a swap silently corrupts it."""
    values = {name: float(i + 1) * 1.5 for i, name in enumerate(FEATURE_ORDER)}
    feat = make_feat(**values)
    vector = feat.to_feature_vector()

    assert len(FEATURE_ORDER) == 11
    assert len(vector) == len(FEATURE_ORDER)
    for name, value in zip(FEATURE_ORDER, vector):
        assert value == pytest.approx(float(values[name])), name


# ------------------------------------------------------------ depth screen

MID_PRICE = (CONFIG.price_min + CONFIG.price_max) / 2


def test_screen_applies_price_band(monkeypatch):
    monkeypatch.setenv("SCREENER_MIN_DEPTH_QTY", "2000")
    assert passes_screen(MID_PRICE, 5000, 5000)
    assert not passes_screen(CONFIG.price_min - 1, 5000, 5000)
    assert not passes_screen(CONFIG.price_max + 1, 5000, 5000)


def test_screen_depth_override_requires_both_sides_strictly_above(monkeypatch):
    monkeypatch.setenv("SCREENER_MIN_DEPTH_QTY", "2000")
    assert passes_screen(MID_PRICE, 2001, 2001)
    assert not passes_screen(MID_PRICE, 2000, 5000)   # equal is not enough
    assert not passes_screen(MID_PRICE, 5000, 1999)


def test_screen_falls_back_to_spec_default_when_override_is_malformed(monkeypatch):
    monkeypatch.setenv("SCREENER_MIN_DEPTH_QTY", "not-a-number")
    spec = CONFIG.min_bid_qty
    assert passes_screen(MID_PRICE, spec + 1, spec + 1)
    assert not passes_screen(MID_PRICE, spec, spec)


def test_screen_uses_spec_default_when_override_is_unset(monkeypatch):
    monkeypatch.delenv("SCREENER_MIN_DEPTH_QTY", raising=False)
    spec = CONFIG.min_bid_qty
    assert passes_screen(MID_PRICE, spec + 1, spec + 1)
    assert not passes_screen(MID_PRICE, spec, spec)


# ---------------------------------------------------------- explain_signal

def test_buy_reasons_reflect_bid_support_and_rising_ltq():
    reasons = explain_signal("BUY", make_feat(bid_ask_imbalance=0.4, ltq_change_pct=0.5))
    assert "Bid support strong" in reasons
    assert "LTQ rising" in reasons


def test_buy_reasons_flag_ask_pressure():
    reasons = explain_signal("BUY", make_feat(bid_ask_imbalance=-0.4))
    assert "Ask pressure building" in reasons


def test_sell_reasons_reflect_ask_pressure():
    reasons = explain_signal("SELL", make_feat(bid_ask_imbalance=-0.4))
    assert "Ask pressure strong" in reasons


def test_acceleration_reasons_only_appear_beyond_thresholds():
    assert "LTQ accelerating" in explain_signal("BUY", make_feat(ltq_acceleration=0.5))
    assert "LTQ decelerating" in explain_signal("BUY", make_feat(ltq_acceleration=-0.5))
    quiet = explain_signal("BUY", make_feat(ltq_acceleration=0.0))
    assert "LTQ accelerating" not in quiet and "LTQ decelerating" not in quiet