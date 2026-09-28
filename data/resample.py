"""
Builds rolling 1-minute OHLCV candles per symbol from incoming ticks.

SMMA and crossover detection operate on these candles, not raw ticks --
the assignment specifies a 1-minute timeframe.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from brokers.base import Tick
from config import CONFIG


@dataclass
class Candle:
    symbol: str
    minute_epoch: int  # bucket start, unix seconds truncated to the minute
    open: float
    high: float
    low: float
    close: float
    volume: int = 0     # sum of ltq within the minute
    last_ltq: int = 0   # most recent single trade's ltq (for acceleration features)
    ticks: List[Tick] = field(default_factory=list)


class CandleBuilder:
    def __init__(self, timeframe_minutes: int = CONFIG.timeframe_minutes):
        self.bucket_seconds = timeframe_minutes * 60
        self._current: Dict[str, Candle] = {}
        self._closed: Dict[str, List[Candle]] = defaultdict(list)

    def _bucket(self, ts: float) -> int:
        return int(ts // self.bucket_seconds * self.bucket_seconds)

    def on_tick(self, tick: Tick) -> Candle | None:
        """
        Feed one tick in. Returns the just-closed Candle if this tick
        rolled the bucket over to a new minute, else None.
        """
        bucket = self._bucket(tick.ts)
        cur = self._current.get(tick.symbol)

        if cur is None or cur.minute_epoch != bucket:
            closed = None
            if cur is not None:
                self._closed[tick.symbol].append(cur)
                closed = cur
            self._current[tick.symbol] = Candle(
                symbol=tick.symbol, minute_epoch=bucket,
                open=tick.ltp, high=tick.ltp, low=tick.ltp, close=tick.ltp,
                volume=tick.ltq, last_ltq=tick.ltq, ticks=[tick],
            )
            return closed

        cur.high = max(cur.high, tick.ltp)
        cur.low = min(cur.low, tick.ltp)
        cur.close = tick.ltp
        cur.volume += tick.ltq
        cur.last_ltq = tick.ltq
        cur.ticks.append(tick)
        return None

    def closed_candles(self, symbol: str, n: int | None = None) -> List[Candle]:
        hist = self._closed[symbol]
        return hist if n is None else hist[-n:]

    def current_candle(self, symbol: str) -> Candle | None:
        return self._current.get(symbol)

    def seed_closed(self, candle: Candle) -> None:
        """
        Directly append a historical candle to closed history, bypassing
        live tick aggregation. Used to warm-start SMMA from a broker's
        REST historical-candle endpoint at startup, so SMMA20/120 don't
        need ~2 hours of live 1-minute candles to become ready after
        every process start or reconnect. Caller is responsible for
        appending in chronological (oldest-first) order.
        """
        self._closed[candle.symbol].append(candle)
