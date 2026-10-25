r"""
Is a saved tick file real market data? Read-only.

    python check_ticks.py data\store\ticks_2026-10-16.csv

Mock ticks (written by the mock broker or by test runs) differ from a real
session in several ways this script looks for:
  * they appear outside market hours, or on a weekend;
  * their prices wander far more than a real stock does in a day;
  * the mock order book never shows less than 50,000 shares on the bid,
    while real depth for many NSE stocks is often a few hundred or thousand.
Compare the output with a file you KNOW is real (for example a recent live
session) to see what normal looks like.
"""
import sys
from datetime import timedelta, timezone

import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
OPEN_MIN, CLOSE_MIN = 9 * 60, 15 * 60 + 45     # 09:00 to 15:45 IST, a little generous
SUSPICIOUS_RATIO = 1.5                          # day high / day low
MOCK_MIN_DEPTH = 50_000                         # the mock never quotes less than this


def check(path: str) -> None:
    rows = outside = fractional = thin_book = 0
    first = last = None
    per_symbol = {}
    cols = ["symbol", "ts", "ltp", "bid_qty"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000):
        rows += len(chunk)
        stamps = pd.to_datetime(chunk["ts"], unit="s", utc=True).dt.tz_convert(IST)
        minutes = stamps.dt.hour * 60 + stamps.dt.minute
        outside += int(((minutes < OPEN_MIN) | (minutes > CLOSE_MIN)).sum())
        fractional += int((chunk["ts"] % 1 != 0).sum())
        thin_book += int(((chunk["bid_qty"] > 0) & (chunk["bid_qty"] < MOCK_MIN_DEPTH)).sum())
        lo, hi = stamps.min(), stamps.max()
        first = lo if first is None or lo < first else first
        last = hi if last is None or hi > last else last
        for symbol, grp in chunk.groupby("symbol")["ltp"]:
            cur = per_symbol.get(symbol, [float("inf"), 0.0])
            per_symbol[symbol] = [min(cur[0], grp.min()), max(cur[1], grp.max())]

    ratios = sorted(((hi / lo if lo > 0 else float("inf"), s, lo, hi)
                     for s, (lo, hi) in per_symbol.items()), reverse=True)
    suspicious = [r for r in ratios if r[0] > SUSPICIOUS_RATIO]

    print(path)
    print(f"  rows: {rows:,}")
    print(f"  first tick: {first:%Y-%m-%d %H:%M:%S} IST ({first:%A})   "
          f"last tick: {last:%Y-%m-%d %H:%M:%S} IST")
    print(f"  rows outside 09:00-15:45 IST: {outside:,}")
    print(f"  rows with a fractional-second timestamp: {fractional / rows:.0%}")
    print(f"  rows with a bid quantity under {MOCK_MIN_DEPTH:,}: {thin_book:,} ({thin_book / rows:.0%})")
    print(f"  symbols whose day high is more than {SUSPICIOUS_RATIO}x the day low: {len(suspicious)}")
    for ratio, symbol, lo, hi in ratios[:3]:
        print(f"    {symbol}: low {lo:.2f}, high {hi:.2f}  (x{ratio:.2f})")

    problems = []
    if first.weekday() >= 5:
        problems.append("first tick is on a weekend")
    if rows > 1000 and thin_book == 0:
        problems.append("the order book never goes below the mock's minimum depth")
    if outside:
        problems.append("ticks outside market hours")
    if suspicious:
        problems.append("price swings too large for one real day")
    if problems:
        print("  VERDICT: NOT CLEAN -- " + "; ".join(problems))
    else:
        print("  VERDICT: looks like clean real data")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["(pass one or more tick csv paths)"]:
        check(p)