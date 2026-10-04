from __future__ import annotations
import random, threading, time
from typing import Callable
from brokers.base import BrokerClient, Tick

class MockClient(BrokerClient):
    name = "mock"
    def __init__(self, symbols, tick_interval: float = 0.25, speed: float = 1.0,
                 sim_seconds_per_loop: float = 0.0):
        """
        sim_seconds_per_loop: if > 0, ticks are timestamped using an
        ACCELERATED simulated clock instead of real wall-clock time,
        advancing by this many simulated seconds per full pass over all
        symbols. This is what lets a demo show many candles closing (and
        therefore real crossovers) within real seconds, since
        CandleBuilder buckets candles by whatever `ts` a tick carries --
        it doesn't know or care whether that's real or simulated time.
        Leave at 0 (default) for a realistic real-time feed.
        """
        super().__init__(symbols)
        self.tick_interval = tick_interval
        self.speed = speed
        self.sim_seconds_per_loop = sim_seconds_per_loop
        self._sim_clock = time.time()
        self._running = False
        self._state = {}
        for s in self.symbols:
            base = random.uniform(40, 480)
            self._state[s] = {
                "price": base, "drift": 0.0, "drift_timer": 0,
                "cum_vol": random.randint(50_000, 500_000),
                # Persistent depth multipliers (mean-reverting around 1.0).
                # Real order books are autocorrelated: depth at tick t is
                # close to depth at tick t-1. Redrawing it independently
                # every tick made bid/ask imbalance flip randomly, which
                # tripped the position monitor's "imbalance flipped"
                # deterioration flag on pure noise.
                "bid_f": 1.0, "ask_f": 1.0,
            }

    def connect(self) -> None:
        self._running = True

    def seed_prices(self, price_map: dict) -> None:
        """
        Overrides each symbol's starting price. Used by demo mode to
        continue seamlessly from wherever a synthetic historical
        warm-up left off, instead of jumping to a fresh independent
        random starting price the moment live ticking begins.
        """
        for symbol, price in price_map.items():
            if symbol in self._state:
                self._state[symbol]["price"] = price

    def disconnect(self) -> None:
        self._running = False

    def subscribe(self, on_tick: Callable[[Tick], None]) -> None:
        self._running = True
        thread = threading.Thread(target=self._run, args=(on_tick,), daemon=True)
        thread.start()
        self._thread = thread

    @staticmethod
    def _step_depth_factor(f: float) -> float:
        # AR(1) around 1.0: pulls back toward 1.0 by 10% per tick, plus
        # small noise. Stationary std is roughly 0.09, so depth stays
        # within about +/-10% of its mean instead of +/-33% iid.
        f += 0.1 * (1.0 - f) + random.gauss(0, 0.04)
        return max(0.2, min(2.0, f))

    def _run(self, on_tick):
        while self._running:
            for symbol in self.symbols:
                st = self._state[symbol]
                if st["drift_timer"] <= 0:
                    st["drift"] = random.choice([-1, -1, 0, 1, 1]) * random.uniform(0.01, 0.08)
                    st["drift_timer"] = random.randint(20, 120)
                st["drift_timer"] -= 1
                noise = random.gauss(0, 0.15)
                st["price"] = max(1.0, st["price"] + st["drift"] + noise)
                ltq = random.randint(1, 5000)
                st["cum_vol"] += ltq
                spread = max(0.05, st["price"] * 0.0008)
                bid_price = round(st["price"] - spread, 2)
                ask_price = round(st["price"] + spread, 2)
                skew = 1.0 + max(-0.6, min(0.6, st["drift"] * 4))
                st["bid_f"] = self._step_depth_factor(st["bid_f"])
                st["ask_f"] = self._step_depth_factor(st["ask_f"])
                bid_qty = int(max(50_000, 9_00_000 * st["bid_f"] * skew))
                ask_qty = int(max(50_000, 9_00_000 * st["ask_f"] * (2 - skew)))
                tick_ts = self._sim_clock if self.sim_seconds_per_loop > 0 else time.time()
                tick = Tick(symbol=symbol, ts=tick_ts, ltp=round(st["price"], 2), ltq=ltq,
                            total_traded_qty=st["cum_vol"], bid_price=bid_price, bid_qty=bid_qty,
                            ask_price=ask_price, ask_qty=ask_qty)
                on_tick(tick)
            if self.sim_seconds_per_loop > 0:
                self._sim_clock += self.sim_seconds_per_loop
            time.sleep(self.tick_interval / self.speed)