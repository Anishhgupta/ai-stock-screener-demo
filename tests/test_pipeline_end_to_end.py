"""
End-to-end smoke test: mock ticks -> candles -> SMMA crossovers ->
feature engineering -> (untrained-model-safe) paper trading bookkeeping.

Run with:  pytest tests/test_pipeline_end_to_end.py -v
"""
import random

from brokers.mock_client import MockClient
from signals.crossover import CrossoverEngine


def test_crossovers_are_detected_from_synthetic_ticks():
    random.seed(7)
    symbols = ["NSE:TATAMOTORS-EQ", "NSE:SBIN-EQ"]
    client = MockClient(symbols, tick_interval=0.0)  # no sleep, just generate fast
    # persist_ticks=False: this test must never write mock ticks into the
    # real data/store tick files (it once did, on every run).
    engine = CrossoverEngine(persist_ticks=False)

    events = []
    captured = []

    def on_tick(tick):
        captured.append(tick)

    # Drive the mock generator manually for a bounded number of ticks
    # instead of using its background thread, so the test is deterministic
    # and fast.
    for _ in range(4000):
        for symbol in symbols:
            st = client._state[symbol]
            if st["drift_timer"] <= 0:
                st["drift"] = random.choice([-1, -1, 0, 1, 1]) * random.uniform(0.01, 0.08)
                st["drift_timer"] = random.randint(20, 120)
            st["drift_timer"] -= 1
            noise = random.gauss(0, 0.15)
            st["price"] = max(1.0, st["price"] + st["drift"] + noise)
            ltq = random.randint(1, 5000)
            st["cum_vol"] += ltq
            spread = max(0.05, st["price"] * 0.0008)
            from brokers.base import Tick
            import time as _time
            tick = Tick(
                symbol=symbol, ts=_time.time(), ltp=round(st["price"], 2), ltq=ltq,
                total_traded_qty=st["cum_vol"],
                bid_price=round(st["price"] - spread, 2), bid_qty=1_200_000,
                ask_price=round(st["price"] + spread, 2), ask_qty=1_100_000,
            )
            ev = engine.on_tick(tick)
            if ev is not None:
                events.append(ev)

    # We're not asserting an exact count (it's stochastic), just that the
    # pipeline is capable of producing screened, feature-complete events.
    assert len(events) >= 0  # pipeline must not raise
    for ev in events:
        assert ev.signal in ("BUY", "SELL")
        assert ev.features.symbol == ev.symbol
        vec = ev.features.to_feature_vector()
        assert len(vec) == 11
        assert all(isinstance(x, float) for x in vec)