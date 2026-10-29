"""
Entry-filter report: did the model's ACCEPT/AVOID verdict at entry say
anything about how the trade ended?

Every crossover is saved with the model's decision and score. The Basic book
enters every crossover, so its trade is the outcome of that crossover whatever
the model said. This report links each saved crossover to its Basic trade and
compares ACCEPT with AVOID.

    python -m storage.entry_filter [--since YYYY-MM-DD]
    docker compose --profile live run --rm dashboard python -m storage.entry_filter

Only crossovers whose Basic trade closed on a real exit count. Session-end and
recovery closes are marks, not outcomes, and are left out. Crossovers that did
not open a Basic trade (already open, or book full) have no outcome. Trades
overlap in time and share market moves, so they are not independent and the
intervals are probably too narrow.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import text

from storage.db import make_engine

IST = timezone(timedelta(hours=5, minutes=30))
MARK_PREFIXES = ("Session end", "Recovered after restart")
MATCH_TOLERANCE_SECONDS = 0.5
MIN_PER_GROUP = 30
MIN_VALID_RESAMPLES = 100
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 12345


def is_mark(reason) -> bool:
    """True for exits that are marks to a price, not real exits."""
    return str(reason or "").startswith(MARK_PREFIXES)


def load_data(engine, since_ts: float | None = None):
    """Saved crossovers and Basic-book trades, optionally from a timestamp on."""
    since = -1.0 if since_ts is None else float(since_ts)
    with engine.connect() as conn:
        events = [dict(row._mapping) for row in conn.execute(text(
            "SELECT id, symbol, signal, ts, decision, probability "
            "FROM crossover_events WHERE ts >= :since ORDER BY ts"), {"since": since})]
        trades = [dict(row._mapping) for row in conn.execute(text(
            "SELECT symbol, signal, entry_ts, pnl, exit_reason "
            "FROM paper_positions WHERE book = 'basic' AND entry_ts >= :since"),
            {"since": since - 1.0})]
    return events, trades


def match_outcomes(events, trades):
    """
    Link each crossover to the Basic trade it opened (same symbol and signal,
    entry time equal to the crossover time). Returns the matched outcomes, how
    many were left out because the trade closed on a mark, and how many
    crossovers have no Basic trade.
    """
    by_key: dict[tuple, list] = {}
    for t in trades:
        by_key.setdefault((t["symbol"], t["signal"]), []).append(t)

    matched, marks, unmatched = [], 0, 0
    for e in events:
        found = None
        for t in by_key.get((e["symbol"], e["signal"]), []):
            if abs(t["entry_ts"] - e["ts"]) < MATCH_TOLERANCE_SECONDS:
                found = t
                break
        if found is None:
            unmatched += 1
        elif is_mark(found["exit_reason"]):
            marks += 1
        else:
            matched.append({
                "decision": e["decision"],
                "probability": e["probability"],
                "pnl": float(found["pnl"]),
            })
    return matched, marks, unmatched


def auc(win_scores, loss_scores) -> float:
    """Chance that a random profitable trade had a higher score than a random
    unprofitable one (ties count half). 0.5 is chance."""
    w = np.asarray(win_scores, dtype=float)[:, None]
    l = np.asarray(loss_scores, dtype=float)[None, :]
    return float(((w > l).sum() + 0.5 * (w == l).sum()) / (w.size * l.size))


def _group_stats(pnls) -> dict:
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "n": n,
        "wins": wins,
        "win_rate": wins / n if n else None,
        "mean_pnl": float(np.mean(pnls)) if n else None,
        "total_pnl": float(np.sum(pnls)) if n else 0.0,
    }


def _interval(values):
    if len(values) < MIN_VALID_RESAMPLES:
        return None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _diff_interval(matched, resamples, rng):
    pnl = np.array([m["pnl"] for m in matched], dtype=float)
    is_accept = np.array([m["decision"] == "ACCEPT" for m in matched])
    n = len(matched)
    diffs = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        a = is_accept[idx]
        if a.any() and not a.all():
            p = pnl[idx]
            diffs.append(p[a].mean() - p[~a].mean())
    return _interval(diffs)


def _auc_interval(scored, resamples, rng):
    if not scored:
        return None
    prob = np.array([s[0] for s in scored], dtype=float)
    win = np.array([s[1] for s in scored], dtype=bool)
    n = len(scored)
    values = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        w = win[idx]
        if w.any() and not w.all():
            p = prob[idx]
            values.append(auc(p[w], p[~w]))
    return _interval(values)


def analyse(matched, marks=0, unmatched=0,
            resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED) -> dict:
    accept_pnl = [m["pnl"] for m in matched if m["decision"] == "ACCEPT"]
    avoid_pnl = [m["pnl"] for m in matched if m["decision"] != "ACCEPT"]
    result = {
        "used": len(matched), "marks": marks, "unmatched": unmatched,
        "accept": _group_stats(accept_pnl), "avoid": _group_stats(avoid_pnl),
        "diff": None, "diff_ci": None, "auc": None, "auc_ci": None, "n_scored": 0,
    }
    if accept_pnl and avoid_pnl:
        result["diff"] = float(np.mean(accept_pnl) - np.mean(avoid_pnl))

    scored = [(m["probability"], m["pnl"] > 0) for m in matched
              if m["probability"] is not None]
    result["n_scored"] = len(scored)
    wins = [p for p, w in scored if w]
    losses = [p for p, w in scored if not w]
    if wins and losses:
        result["auc"] = auc(wins, losses)

    if matched:
        rng = np.random.default_rng(seed)
        result["diff_ci"] = _diff_interval(matched, resamples, rng)
        result["auc_ci"] = _auc_interval(scored, resamples, rng)
    return result


def _group_line(name: str, g: dict) -> str:
    if g["n"] == 0:
        return f"{name}: no trades"
    return (f"{name}: {g['n']} trades, {g['wins']} profitable ({g['win_rate']:.0%}), "
            f"mean P/L Rs {g['mean_pnl']:.1f}, total Rs {g['total_pnl']:.0f}")


def format_report(r: dict) -> str:
    total = r["used"] + r["marks"] + r["unmatched"]
    lines = [
        "Entry filter: did the ACCEPT/AVOID verdict say anything about how the trade ended?",
        "",
        f"Crossovers saved: {total}",
        f"  with a closed Basic trade (real exit): {r['used']}",
        f"  left out, Basic trade closed by a session-end or recovery mark: {r['marks']}",
        f"  no Basic trade (skipped, or none recorded): {r['unmatched']}",
    ]
    if r["used"] == 0:
        lines += ["", "No outcomes to report yet."]
        return "\n".join(lines)

    lines += ["", _group_line("ACCEPT", r["accept"]), _group_line("AVOID ", r["avoid"]), ""]

    if r["diff"] is None:
        lines.append("Mean P/L difference: needs trades in both groups.")
    else:
        span = (f"  [95% bootstrap interval Rs {r['diff_ci'][0]:.1f} to Rs {r['diff_ci'][1]:.1f}]"
                if r["diff_ci"] else "  [interval not available]")
        lines.append(f"Mean P/L difference (ACCEPT - AVOID): Rs {r['diff']:.1f}{span}")
        lines.append("  positive = ACCEPT trades did better")

    if r["auc"] is None:
        lines.append("Score AUC: needs scored trades that were both profitable and not.")
    else:
        span = (f"  [95% bootstrap interval {r['auc_ci'][0]:.2f} to {r['auc_ci'][1]:.2f}]"
                if r["auc_ci"] else "  [interval not available]")
        lines.append(f"Score AUC (profitable vs not): {r['auc']:.2f}{span}, "
                     f"on {r['n_scored']} scored trades (0.50 = chance)")

    small = [name for name, g in (("ACCEPT", r["accept"]), ("AVOID", r["avoid"]))
             if g["n"] < MIN_PER_GROUP]
    if small:
        lines += ["", f"Too few trades in {' and '.join(small)}: want at least "
                      f"{MIN_PER_GROUP} per group before reading anything into this."]
    lines += [
        "",
        "Trades overlap in time and share market moves, so they are not independent;",
        "the intervals are probably too narrow.",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare ACCEPT and AVOID crossovers by the outcome of their Basic trade.")
    parser.add_argument("--since", help="only crossovers on or after this date, YYYY-MM-DD (Indian time)")
    args = parser.parse_args(argv)

    since_ts = None
    if args.since:
        since_ts = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=IST).timestamp()

    events, trades = load_data(make_engine(), since_ts)
    matched, marks, unmatched = match_outcomes(events, trades)
    print(format_report(analyse(matched, marks, unmatched)))


if __name__ == "__main__":
    main()