"""
What did the deterioration monitor add or cost?

Both books enter every ACCEPTed crossover at the same price and time, so
each Filtered trade has a Basic "twin" that ran the same entry without the
monitor. Comparing the two isolates the monitor's effect:

    effect = Filtered P/L - Basic P/L      (positive: the monitor helped)

Run it against the saved database:
    python -m storage.monitor_effect
    python -m storage.monitor_effect --since 2026-10-15   # new monitor only
"""
from __future__ import annotations

import argparse
import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

IST = timezone(timedelta(hours=5, minutes=30))
MONITOR_PREFIX = "Deterioration exit"
SESSION_END_PREFIX = "Session end"
MATCH_SECONDS = 5.0
FLAT_RUPEES = 0.005
MIN_PAIRS_FOR_CONCLUSIONS = 30


def load_rows(repo, user_id) -> list[dict]:
    return [
        {"book": p.book, "symbol": p.symbol, "signal": p.signal,
         "entry_ts": p.entry_ts, "entry_price": p.entry_price,
         "exit_ts": p.exit_ts, "exit_reason": p.exit_reason, "pnl": p.pnl}
        for p in repo.closed_positions(user_id)
    ]


def pair_trades(rows: list[dict], since_ts: Optional[float] = None):
    """
    Match each Filtered trade with the Basic trade opened on the same
    crossover: same symbol, signal and entry price, entered within
    MATCH_SECONDS. Returns (pairs, unmatched_filtered_trades).
    """
    basic = [r for r in rows if r["book"] == "basic"]
    filtered = [r for r in rows if r["book"] == "filtered"
                and (since_ts is None or r["entry_ts"] >= since_ts)]
    used: set[int] = set()
    pairs, unmatched = [], []
    for f in filtered:
        best = None
        for i, b in enumerate(basic):
            if i in used or b["symbol"] != f["symbol"] or b["signal"] != f["signal"]:
                continue
            if not math.isclose(b["entry_price"], f["entry_price"], rel_tol=1e-9):
                continue
            gap = abs(b["entry_ts"] - f["entry_ts"])
            if gap <= MATCH_SECONDS and (best is None or gap < best[0]):
                best = (gap, i)
        if best is None:
            unmatched.append(f)
        else:
            used.add(best[1])
            pairs.append((f, basic[best[1]]))
    return pairs, unmatched


def analyze(rows: list[dict], since_ts: Optional[float] = None) -> dict:
    pairs, unmatched = pair_trades(rows, since_ts)
    cut = [(f, b) for f, b in pairs if f["exit_reason"].startswith(MONITOR_PREFIX)]
    other = [(f, b) for f, b in pairs if not f["exit_reason"].startswith(MONITOR_PREFIX)]

    diffs = [f["pnl"] - b["pnl"] for f, b in cut]
    basic_pnls = [b["pnl"] for _, b in cut]
    mean = statistics.fmean(diffs) if diffs else None
    std_error = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) >= 2 else None

    return {
        "filtered_trades": len(pairs) + len(unmatched),
        "paired": len(pairs),
        "unmatched": len(unmatched),
        "monitor_cuts": {
            "n": len(cut),
            "filtered_pnl": sum(f["pnl"] for f, _ in cut),
            "basic_pnl": sum(basic_pnls),
            "effect": sum(diffs),
            "mean_effect": mean,
            "std_error": std_error,
            "helped": sum(1 for d in diffs if d > FLAT_RUPEES),
            "hurt": sum(1 for d in diffs if d < -FLAT_RUPEES),
            "same": sum(1 for d in diffs if abs(d) <= FLAT_RUPEES),
            "basic_wins": sum(1 for p in basic_pnls if p > 0),
            "basic_losses": sum(1 for p in basic_pnls if p < 0),
            "basic_flat": sum(1 for p in basic_pnls if p == 0),
            "basic_session_end": sum(1 for _, b in cut
                                     if b["exit_reason"].startswith(SESSION_END_PREFIX)),
        },
        "other_exits": {
            "n": len(other),
            "mismatches": sum(1 for f, b in other if abs(f["pnl"] - b["pnl"]) > 0.01),
        },
    }


def format_report(summary: dict, since: Optional[str] = None) -> str:
    if summary["filtered_trades"] == 0:
        if since:
            return f"No closed Filtered trades entered on or after {since}."
        return "No closed Filtered trades saved yet."
    if summary["paired"] == 0:
        return (f"{summary['filtered_trades']} Filtered trades, but none has a matching "
                "Basic trade to compare with.")

    cuts, other = summary["monitor_cuts"], summary["other_exits"]
    lines = [
        f"Filtered trades: {summary['filtered_trades']}  "
        f"(paired with a Basic twin: {summary['paired']}, no twin found: {summary['unmatched']})",
        "",
        f"Cut early by the monitor: {cuts['n']} trades",
    ]
    if cuts["n"]:
        lines += [
            f"  Filtered book P/L on these: Rs {cuts['filtered_pnl']:.1f}",
            f"  Basic book P/L on the same entries (no monitor): Rs {cuts['basic_pnl']:.1f}",
            f"  Monitor effect (Filtered - Basic): Rs {cuts['effect']:.1f}  "
            f"[positive = the monitor helped]",
        ]
        if cuts["std_error"] is not None:
            lines.append(f"  Per trade: Rs {cuts['mean_effect']:.1f} +/- {cuts['std_error']:.1f} (standard error)")
        lines += [
            f"  Monitor helped on {cuts['helped']}, hurt on {cuts['hurt']}, no difference on {cuts['same']}",
            f"  Left alone, those entries ended: {cuts['basic_wins']} wins, "
            f"{cuts['basic_losses']} losses, {cuts['basic_flat']} flat",
        ]
        if cuts["basic_session_end"]:
            lines.append(f"  ({cuts['basic_session_end']} of the Basic twins were closed by a session-end "
                         "mark, not a real exit)")
    lines += ["", f"Other exits (target, stop, max hold, session end): {other['n']} paired trades; "
                  f"{other['mismatches']} where the twins differ (expected 0)."]
    if summary["unmatched"]:
        lines.append(f"{summary['unmatched']} Filtered trades had no recorded Basic twin "
                     "(for example, Basic was still open from an earlier crossover, or was "
                     "never closed before the session-end fix). They are left out.")
    if cuts["n"] < MIN_PAIRS_FOR_CONCLUSIONS:
        lines += ["", f"Only {cuts['n']} monitor exits so far: too few to draw conclusions "
                      f"(want at least {MIN_PAIRS_FOR_CONCLUSIONS})."]
    return "\n".join(lines)


def since_to_epoch(day: str) -> float:
    """A YYYY-MM-DD date (Indian time, midnight) as epoch seconds."""
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=IST).timestamp()


def main(argv=None) -> None:
    from storage.db import make_engine
    from storage.repository import Repository

    parser = argparse.ArgumentParser(description="What did the deterioration monitor add or cost?")
    parser.add_argument("--since", help="only Filtered trades entered on/after this date (YYYY-MM-DD, IST)")
    args = parser.parse_args(argv)

    repo = Repository(make_engine())
    user_id = repo.ensure_user("local")
    since_ts = since_to_epoch(args.since) if args.since else None
    print(format_report(analyze(load_rows(repo, user_id), since_ts), args.since))


if __name__ == "__main__":
    main()