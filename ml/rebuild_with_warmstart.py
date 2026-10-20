"""
Rebuilds a properly warm-started training set for a given day's tick
CSV, instead of the cold-start replay ml.labeling.build_training_set()
uses by default.

WHY THIS EXISTS (found 2026-09-02): main.py warm-starts SMMA from the
broker's historical-candle endpoint at every live startup, so it can
detect crossovers almost immediately. build_training_set() previously
always replayed cold -- no warm-start -- so it had to spend the first
~2 hours of a day's OWN tick data just re-warming SMMA20/120 before
detecting any crossover, silently losing every real crossover that
happened during that window. Confirmed directly: a live session on
2026-09-02 logged 18 real ML-scored crossovers (1 accepted + 17
avoided), but a cold-start replay of that same day's tick CSV
reconstructed only 1.

This script fetches the SAME KIND of historical candles main.py uses,
but ending exactly at the tick CSV's first recorded timestamp (not
"now") so there's no time overlap between the historical warm-start
window and the tick CSV's own data.

Requires a live broker connection (valid .env credentials) since it
calls the broker's historical-candle REST endpoint -- this works any
time of day, not just during market hours, as long as the requested
date range is a real trading period.

Usage:
    python -m ml.rebuild_with_warmstart --tick-csv data/store/ticks_2026-09-02.csv --broker fyers
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from config import CONFIG
from signals.crossover import CrossoverEngine
from ml.labeling import build_training_set, load_ticks_csv


def build_warmstarted_engine(broker_name: str, symbols: list[str], tick_csv_path: Path,
                              minutes_back: int = 150) -> CrossoverEngine:
    ticks = load_ticks_csv(tick_csv_path)
    if not ticks:
        raise RuntimeError(f"{tick_csv_path} has no ticks -- nothing to warm-start against.")
    first_tick_ts = min(t.ts for t in ticks)

    if broker_name == "fyers":
        from brokers.fyers_client import FyersClient
        client = FyersClient(symbols)
    else:
        raise ValueError(f"Warm-start rebuild currently only supports --broker fyers, got '{broker_name}'")

    print(f"Connecting to {broker_name} to fetch historical candles...")
    client.connect()

    engine = CrossoverEngine(persist_ticks=False)
    warmed = 0
    for symbol in symbols:
        try:
            candles = client.historical_candles(symbol, minutes_back=minutes_back, end_ts=first_tick_ts)
        except Exception as e:
            print(f"  Warm-start fetch failed for {symbol}: {e}")
            continue
        n = engine.warm_start(symbol, candles)
        if n > 0:
            warmed += 1
        time.sleep(0.25)  # be polite to rate limits across many symbols
    print(f"Warm-started {warmed}/{len(symbols)} symbols, ending at "
          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(first_tick_ts))} "
          f"(the tick CSV's first recorded timestamp).")

    # IMPORTANT: "warmed" above only means SOME candles were applied per
    # symbol -- it does NOT confirm SMMA120 actually reached readiness
    # (needs 120+ candles). If the broker's historical endpoint returned
    # fewer bars than requested for that window, SMMA can silently still
    # be unready, which would just reproduce the original cold-start
    # problem under a different name. Verify explicitly:
    not_ready = []
    for symbol in symbols:
        pair = engine.smma.get(symbol)
        if not pair.slow.ready:
            not_ready.append(symbol)
    if not_ready:
        print(
            f"\nWARNING: SMMA120 is NOT yet ready for {len(not_ready)}/{len(symbols)} "
            f"symbols after warm-start -- the historical fetch likely returned "
            f"fewer than 120 candles for at least one symbol in this window. "
            f"Try a larger --minutes-back (e.g. 300+) to fetch more history. "
            f"Not-ready symbols: {not_ready[:10]}{'...' if len(not_ready) > 10 else ''}"
        )
    else:
        print(f"Confirmed: SMMA120 is fully ready for all {len(symbols)} symbols before replay begins.")

    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tick-csv", required=True, type=Path)
    parser.add_argument("--broker", default="fyers", choices=["fyers"])
    parser.add_argument("--minutes-back", type=int, default=150,
                         help="How far back to fetch historical candles, ending at the tick CSV's first timestamp.")
    args = parser.parse_args()

    engine = build_warmstarted_engine(args.broker, CONFIG.watchlist, args.tick_csv, args.minutes_back)
    examples = build_training_set(args.tick_csv, engine=engine)

    print(f"\nReconstructed {len(examples)} crossover examples from {args.tick_csv} (warm-started replay).")
    print("Compare this to a cold-start count via: python -c \"from ml.labeling import build_training_set; "
          f"from pathlib import Path; print(len(build_training_set(Path('{args.tick_csv}'))))\"")