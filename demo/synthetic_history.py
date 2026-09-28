"""
Generates synthetic historical 1-minute candles AND full tick-level
data for demo purposes -- no real broker account or credentials needed.

Two generators, two different jobs:
  generate_synthetic_history()  -> coarse OHLCV candles, used to
      instantly warm-start SMMA20/120 for a LIVE demo session (see
      demo_app.py), matching the [epoch,o,h,l,c,v] format a real
      broker's historical-candle endpoint returns.
  generate_synthetic_ticks()    -> full tick-level data (bid/ask/LTQ
      granularity), used OFFLINE to train a small demo-only ML model
      (see demo/train_demo_model.py) -- CrossoverEngine needs real tick
      detail, not just OHLC bars, to compute LTQ acceleration and
      Bid/Ask imbalance features.

Both share the same mean-reverting-random-walk-with-drift price model
as brokers/mock_client.py, so the demo's history/training data and its
live feed look statistically consistent with each other.
"""
from __future__ import annotations

import random
import time


def generate_synthetic_history(symbol_seed: int, num_candles: int = 150,
                                start_price: float | None = None) -> tuple[list, float]:
    """
    Returns (candles, ending_price). `candles` is oldest-first, ending
    at "now - 1 minute" so it hands off cleanly to live ticking that
    starts at "now". `ending_price` should be passed to
    MockClient.seed_prices() so live ticking continues from the same
    level instead of jumping to an unrelated fresh random price.
    """
    rng = random.Random(symbol_seed)
    price = start_price if start_price is not None else rng.uniform(40, 480)
    now = int(time.time())
    candles = []
    drift = 0.0
    drift_timer = 0

    for i in range(num_candles):
        epoch = now - (num_candles - i) * 60
        if drift_timer <= 0:
            drift = rng.choice([-1, -1, 0, 1, 1]) * rng.uniform(0.02, 0.12)
            drift_timer = rng.randint(15, 40)
        drift_timer -= 1

        open_price = price
        path = [price]
        for _ in range(10):
            price = max(1.0, price + drift / 10 + rng.gauss(0, 0.15))
            path.append(price)
        close_price = price

        candles.append([
            epoch,
            round(open_price, 2),
            round(max(path), 2),
            round(min(path), 2),
            round(close_price, 2),
            rng.randint(1000, 8000),
        ])

    return candles, price


def generate_synthetic_ticks(symbols: list[str], sim_minutes: int = 400,
                              ticks_per_minute: int = 6) -> list:
    """
    Generates a full synthetic "trading day" of tick-level data (not
    just OHLC candles) entirely in memory, for training a demo-only ML
    model with zero real broker data or files involved.

    Returns a flat list of brokers.base.Tick objects spanning
    `sim_minutes` simulated minutes, `ticks_per_minute` ticks per symbol
    per minute.
    """
    from brokers.base import Tick

    now = time.time()
    all_ticks = []

    for symbol in symbols:
        rng = random.Random(abs(hash(symbol)) % (2**31))
        price = rng.uniform(40, 480)
        drift = 0.0
        drift_timer = 0
        cum_vol = rng.randint(50_000, 500_000)
        start_ts = now - sim_minutes * 60

        for minute in range(sim_minutes):
            if drift_timer <= 0:
                drift = rng.choice([-1, -1, 0, 1, 1]) * rng.uniform(0.01, 0.08)
                drift_timer = rng.randint(20, 120)
            drift_timer -= 1

            for sub in range(ticks_per_minute):
                ts = start_ts + minute * 60 + sub * (60 / ticks_per_minute)
                price = max(1.0, price + drift / ticks_per_minute + rng.gauss(0, 0.15))
                ltq = rng.randint(1, 5000)
                cum_vol += ltq
                spread = max(0.05, price * 0.0008)
                skew = 1.0 + max(-0.6, min(0.6, drift * 4))
                bid_qty = int(max(500, rng.gauss(3000, 1200) * skew))
                ask_qty = int(max(500, rng.gauss(3000, 1200) * (2 - skew)))
                all_ticks.append(Tick(
                    symbol=symbol, ts=ts, ltp=round(price, 2), ltq=ltq,
                    total_traded_qty=cum_vol,
                    bid_price=round(price - spread, 2), bid_qty=bid_qty,
                    ask_price=round(price + spread, 2), ask_qty=ask_qty,
                ))

    return all_ticks


def generate_all_symbol_histories(symbols: list[str], num_candles: int = 150) -> dict[str, tuple[list, float]]:
    """
    One deterministic synthetic history per symbol (seeded from the
    symbol's own name, so re-running the demo with the same watchlist
    produces the same-shaped price paths each time -- reproducible,
    not different random luck on every restart).
    """
    result = {}
    for symbol in symbols:
        seed = abs(hash(symbol)) % (2**31)
        result[symbol] = generate_synthetic_history(seed, num_candles)
    return result
