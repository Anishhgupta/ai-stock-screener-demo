"""
How stable is the model's verdict? Offline analysis of the crossovers and
re-scores that the trading engine records.

Run it against the saved database:
    python -m storage.analysis

For every crossover it compares the entry probability (scored on the
crossover tick) with the model's scores on the ticks that followed, and
counts how often a later score would have flipped ACCEPT <-> AVOID.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

EVENT_COLUMNS = ["symbol", "signal", "decision", "entry_prob",
                 "n_early", "early_mean", "n_late", "late_mean"]
MIN_EVENTS_FOR_CONCLUSIONS = 30


def score_stability(repo, user_id, early=(10.0, 60.0), late=(120.0, 600.0)) -> pd.DataFrame:
    """One row per crossover: entry score vs the mean of later re-scores."""
    events = pd.DataFrame(repo.crossover_events(user_id))
    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    samples = pd.DataFrame(repo.score_samples(user_id))

    rows = []
    for ev in events.itertuples():
        if samples.empty:
            mine = samples
        else:
            mine = samples[(samples["symbol"] == ev.symbol) & (samples["event_ts"] == ev.ts)]

        def window(lo, hi):
            if mine.empty:
                return pd.Series(dtype=float)
            sel = mine[(mine["seconds_since_event"] >= lo) & (mine["seconds_since_event"] <= hi)]
            return sel["score"]

        early_scores, late_scores = window(*early), window(*late)
        rows.append({
            "symbol": ev.symbol, "signal": ev.signal, "decision": ev.decision,
            "entry_prob": ev.probability,
            "n_early": len(early_scores),
            "early_mean": early_scores.mean() if len(early_scores) else None,
            "n_late": len(late_scores),
            "late_mean": late_scores.mean() if len(late_scores) else None,
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def summarize(df: pd.DataFrame, threshold: float) -> dict:
    """Per decision group: how far and how often later scores disagree with entry."""
    out = {}
    scored = df.dropna(subset=["entry_prob"])
    for decision, group in scored.groupby("decision"):
        measured = group.dropna(subset=["early_mean"])
        if decision == "ACCEPT":
            flipped = (measured["early_mean"] < threshold).sum()
        else:
            flipped = (measured["early_mean"] >= threshold).sum()
        out[decision] = {
            "events": len(group),
            "measured": len(measured),
            "mean_entry": float(group["entry_prob"].mean()),
            "mean_early": float(measured["early_mean"].mean()) if len(measured) else None,
            "mean_late": (float(group.dropna(subset=["late_mean"])["late_mean"].mean())
                          if group["late_mean"].notna().any() else None),
            "flip_rate": float(flipped / len(measured)) if len(measured) else None,
        }
    return out


def format_report(df: pd.DataFrame, threshold: float) -> str:
    summary = summarize(df, threshold)
    if not summary:
        return "No scored crossovers saved yet."

    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"

    def num(x):
        return "n/a" if x is None else f"{x:.2f}"

    lines = [f"Model score stability (ACCEPT threshold {threshold:.2f})", ""]
    for decision in ("ACCEPT", "AVOID"):
        if decision not in summary:
            continue
        g = summary[decision]
        lines.append(
            f"{decision}: {g['events']} crossovers ({g['measured']} re-scored 10-60s later)\n"
            f"  mean entry score {num(g['mean_entry'])} -> "
            f"mean re-score at 10-60s {num(g['mean_early'])}, at 2-10 min {num(g['mean_late'])}\n"
            f"  verdict flipped by the later score: {pct(g['flip_rate'])}"
        )
    total = sum(g["events"] for g in summary.values())
    if total < MIN_EVENTS_FOR_CONCLUSIONS:
        lines += ["", f"Only {total} crossovers so far: too few to draw conclusions "
                      f"(want at least {MIN_EVENTS_FOR_CONCLUSIONS})."]
    return "\n".join(lines)


def main(threshold: Optional[float] = None) -> None:
    from config import CONFIG
    from storage.db import make_engine
    from storage.repository import Repository

    repo = Repository(make_engine())
    user_id = repo.ensure_user("local")
    df = score_stability(repo, user_id)
    print(format_report(df, threshold if threshold is not None else CONFIG.accept_probability))


if __name__ == "__main__":
    main()