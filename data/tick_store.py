from __future__ import annotations
import csv
from collections import defaultdict, deque
from datetime import date
from typing import Dict, Deque
from brokers.base import Tick
from config import DATA_DIR

TICK_FIELDS = ["symbol", "ts", "ltp", "ltq", "total_traded_qty", "bid_price", "bid_qty", "ask_price", "ask_qty"]

class TickStore:
    def __init__(self, buffer_len=500, out_dir=DATA_DIR, persist: bool = True):
        self.buffer_len = buffer_len
        self.buffers: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=buffer_len))
        self.out_dir = out_dir
        self.persist = persist
        self._file_handles = {}
        self._writers = {}
    def _csv_path_for_today(self):
        return self.out_dir / f"ticks_{date.today().isoformat()}.csv"
    def _writer_for_today(self):
        path = self._csv_path_for_today()
        key = str(path)
        if key not in self._writers:
            is_new = not path.exists()
            fh = open(path, "a", newline="")
            writer = csv.DictWriter(fh, fieldnames=TICK_FIELDS)
            if is_new: writer.writeheader()
            self._file_handles[key] = fh
            self._writers[key] = writer
        return self._writers[key], self._file_handles[key]
    def add(self, tick: Tick):
        self.buffers[tick.symbol].append(tick)
        if not self.persist:
            return
        writer, fh = self._writer_for_today()
        writer.writerow({"symbol": tick.symbol, "ts": tick.ts, "ltp": tick.ltp, "ltq": tick.ltq,
            "total_traded_qty": tick.total_traded_qty, "bid_price": tick.bid_price,
            "bid_qty": tick.bid_qty, "ask_price": tick.ask_price, "ask_qty": tick.ask_qty})
        fh.flush()
    def recent(self, symbol, n=None):
        buf = self.buffers[symbol]
        return list(buf) if n is None else list(buf)[-n:]
    def close(self):
        for fh in self._file_handles.values():
            fh.close()