"""
Builds a labeled training set from a completed trading day's raw tick
CSV (as written by data/tick_store.py).

Each crossover is replayed forward the way a Basic-book trade would end: a stop, a
target, or the maximum hold time. Pass session_close=(15, 15) to stop the replay at the
end of continuous trading, as the live engine does. Since 3 August 2026 F&O stocks stop
trading continuously at 15:15 and a closing auction follows, so later prices are frozen
and then jump to an auction price nobody could have traded at. Without the cutoff, a
crossover after about 14:30 can be labelled by that jump. With it, a crossover at or
after 15:15 is skipped (the engine opens nothing then), and one still open at 15:15 is
labelled by the last price before it, which is how the live engine marks it.

    python -m ml.labeling data/store/ticks_2026-08-18.csv data/store/ticks_2026-08-31.csv

prints, for each file, how many labels the close affects. Nothing is written.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from brokers.base import Tick
from config import CONFIG
from signals.crossover import CrossoverEngine, CrossoverEvent

LOOKAHEAD_MINUTES = CONFIG.max_hold_minutes
IST = timezone(timedelta(hours=5, minutes=30))
MARKET_CLOSE_IST = (15, 15)   # keep equal to main.MARKET_CLOSE_IST (a test checks this)


@dataclass
class LabeledExample:
    symbol: str
    signal: str
    ts: float
    features: list
    outcome_pct: float
    label: int


def new_stats() -> dict:
    """Counters that show what the close of trading did to a set of labels."""
    return {"crossovers": 0, "skipped_after_close": 0, "window_crossed_close": 0,
            "label_changed": 0, "changed_to_win": 0, "changed_to_loss": 0}


def _close_ts(ts: float, session_close) -> float:
    """The close of continuous trading on the Indian trading day that `ts` falls in."""
    day = datetime.fromtimestamp(ts, IST)
    return day.replace(hour=session_close[0], minute=session_close[1], second=0, microsecond=0).timestamp()


def load_ticks_csv(path: Path) -> List[Tick]:
    ticks = []
    skipped = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ticks.append(Tick(
                    symbol=row["symbol"], ts=float(row["ts"]), ltp=float(row["ltp"]),
                    ltq=int(row["ltq"]), total_traded_qty=int(row["total_traded_qty"]),
                    bid_price=float(row["bid_price"]), bid_qty=int(row["bid_qty"]),
                    ask_price=float(row["ask_price"]), ask_qty=int(row["ask_qty"]),
                ))
            except (TypeError, ValueError, KeyError):
                skipped += 1
    if skipped:
        print(f"  (skipped {skipped} malformed row(s) in {path})")
    return ticks


def _forward_outcome(ticks_by_symbol, symbol, entry_idx, entry_price, signal, session_close=None):
    series = ticks_by_symbol[symbol]
    entry_ts = series[entry_idx].ts
    stop = CONFIG.stop_loss_pct
    target = CONFIG.target_pct
    direction = 1 if signal == "BUY" else -1
    close_ts = _close_ts(entry_ts, session_close) if session_close is not None else None

    def inside(t):
        """A tick the trade could still have seen: within the hold time and before the close."""
        return t.ts - entry_ts <= LOOKAHEAD_MINUTES * 60 and (close_ts is None or t.ts < close_ts)

    for t in series[entry_idx + 1:]:
        if not inside(t):
            break
        move = direction * (t.ltp - entry_price) / entry_price
        if move >= target:
            return target
        if move <= -stop:
            return -stop
    last = series[entry_idx]
    for t in series[entry_idx + 1:]:
        if not inside(t):
            break
        last = t
    return direction * (last.ltp - entry_price) / entry_price


def build_training_set_from_ticks(ticks: List[Tick], engine: Optional[CrossoverEngine] = None,
                                  session_close=None, stats: Optional[dict] = None) -> List[LabeledExample]:
    """
    Same labeling logic as build_training_set(), but operating on an
    already-loaded list of Tick objects instead of reading a CSV file.
    Used by both build_training_set() (after loading from disk) and
    demo/train_demo_model.py (which generates synthetic ticks entirely
    in memory, with no file ever written).

    session_close=(hour, minute) in Indian time stops each replay at that time and skips
    crossovers at or after it. It is off by default, so synthetic ticks with an arbitrary
    clock are untouched. If `stats` (see new_stats) is given, it is filled in.
    """
    ticks = sorted(ticks, key=lambda t: t.ts)

    ticks_by_symbol = {}
    for t in ticks:
        ticks_by_symbol.setdefault(t.symbol, []).append(t)

    if engine is None:
        # persist_ticks=False: reading ticks to build a training set must never
        # write them back into data/store (it once re-appended every tick to
        # today's tick file on each training run).
        engine = CrossoverEngine(persist_ticks=False)
    events = []

    for t in ticks:
        ev = engine.on_tick(t)
        if ev is not None:
            events.append((ev, t.symbol, t.ts))

    examples = []
    for ev, symbol, entry_ts in events:
        if session_close is not None and entry_ts >= _close_ts(entry_ts, session_close):
            if stats is not None:
                stats["skipped_after_close"] += 1
            continue
        series = ticks_by_symbol[symbol]
        entry_idx = next((i for i, t in enumerate(series) if t.ts >= entry_ts), None)
        if entry_idx is None:
            continue
        entry_price = series[entry_idx].ltp
        outcome_pct = _forward_outcome(ticks_by_symbol, symbol, entry_idx, entry_price, ev.signal, session_close)
        if stats is not None:
            stats["crossovers"] += 1
            if session_close is not None and entry_ts + LOOKAHEAD_MINUTES * 60 > _close_ts(entry_ts, session_close):
                stats["window_crossed_close"] += 1
                without_close = _forward_outcome(ticks_by_symbol, symbol, entry_idx, entry_price, ev.signal)
                if (outcome_pct > 0) != (without_close > 0):
                    stats["label_changed"] += 1
                    stats["changed_to_win" if outcome_pct > 0 else "changed_to_loss"] += 1
        examples.append(LabeledExample(
            symbol=symbol, signal=ev.signal, ts=entry_ts,
            features=ev.features.to_feature_vector(),
            outcome_pct=outcome_pct,
            label=1 if outcome_pct > 0 else 0,
        ))

    return examples


def build_training_set(tick_csv_path: Path, engine: Optional[CrossoverEngine] = None,
                       session_close=None, stats: Optional[dict] = None) -> List[LabeledExample]:
    ticks = load_ticks_csv(tick_csv_path)
    return build_training_set_from_ticks(ticks, engine=engine, session_close=session_close, stats=stats)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Show how much the 15:15 close changes the training labels.")
    parser.add_argument("tick_csv", nargs="+", type=Path, help="tick files, for example the ones a model was trained on")
    args = parser.parse_args(argv)
    total = new_stats()
    print(f"Labels with the replay stopped at {MARKET_CLOSE_IST[0]:02d}:{MARKET_CLOSE_IST[1]:02d} IST, "
          "compared with the old replay that ran on into the closing auction:\n")
    print(f"{'file':<46}{'kept':>6}{'after':>7}{'crossed':>9}{'changed':>9}")
    for path in args.tick_csv:
        stats = new_stats()
        build_training_set(path, session_close=MARKET_CLOSE_IST, stats=stats)
        print(f"{str(path):<46}{stats['crossovers']:>6}{stats['skipped_after_close']:>7}"
              f"{stats['window_crossed_close']:>9}{stats['label_changed']:>9}")
        for key, value in stats.items():
            total[key] += value
    print(f"\nkept = crossovers used; after = crossovers at or after the close (skipped);"
          f"\ncrossed = labelled by a window that would have run past the close; changed = labels that flip.")
    seen = total["crossovers"] + total["skipped_after_close"]
    if seen:
        print(f"\nOf {seen} crossovers, {total['skipped_after_close']} came after the close and "
              f"{total['window_crossed_close']} of the rest had a window running past it; "
              f"{total['label_changed']} labels change ({total['changed_to_win']} to profitable, "
              f"{total['changed_to_loss']} to not profitable).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())