"""
Is the deployed ACCEPT/AVOID model better than chance, judged on EVERY day with a Basic-book outcome?

    python -m ml.deployed_check corrected_nov.db
    python -m ml.deployed_check corrected_nov.db --include-marks

The walk-forward test (ml/walk_forward.py) has to hold back early days to train its
candidates, so it can score the deployed model only on the later days. The deployed
model needs no training, so this check uses every day. Run it on a COPY of the database
(for past data, the corrected copy that remark_trades.py writes).

The interval comes from resampling whole DAYS, because trades on the same day share
market moves and are not independent. The row-by-row interval is printed to show how
much narrower it is. A real effect should also point the same way week after week; one
stretch that drives the whole result is a warning sign. Nothing here changes the model.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from ml.walk_forward import CURRENT, auc, bootstrap, load_rows

MIN_WEEK_ROWS = 15


def by_week(rows) -> list:
    """Per calendar week (Monday to Sunday): crossovers, profitable ones, and the AUC if the week is big enough."""
    groups = defaultdict(list)
    for r in rows:
        iso = date.fromisoformat(r.day).isocalendar()
        groups[(iso[0], iso[1])].append(r)
    weeks = []
    for (year, week), members in sorted(groups.items()):
        y = np.array([m.profitable for m in members], dtype=bool)
        s = np.array([m.probability for m in members], dtype=float)
        weeks.append({
            "week_of": date.fromisocalendar(year, week, 1).isoformat(), "n": len(members),
            "profitable": int(y.sum()),
            "auc": auc(y, s) if len(members) >= MIN_WEEK_ROWS else float("nan"),
        })
    return weeks


def evaluate(rows, resamples: int = 3000, seed: int = 12345):
    """The deployed model's AUC over all rows that have a saved probability, or None if there are none."""
    scored = [r for r in rows if r.probability is not None]
    if not scored:
        return None
    y = np.array([r.profitable for r in scored], dtype=bool)
    s = np.array([r.probability for r in scored], dtype=float)
    days = np.array([r.day for r in scored])
    day_ci, _ = bootstrap(y, days, {CURRENT: s}, "day", resamples, seed)
    row_ci, _ = bootstrap(y, days, {CURRENT: s}, "row", resamples, seed)
    return {
        "n": len(scored), "n_days": len(set(days.tolist())), "positives": int(y.sum()),
        "auc": auc(y, s), "day_ci": day_ci[CURRENT], "row_ci": row_ci[CURRENT], "weeks": by_week(scored),
    }


def verdict(result: dict) -> str:
    ci, row_ci, a = result["day_ci"], result["row_ci"], result["auc"]
    if ci is None:
        return "Too few days to judge."
    if ci[1] < 0.5:
        return (f"Resampling whole days, the deployed model ranks trades WORSE than chance "
                f"(AUC {a:.2f}, interval [{ci[0]:.2f}, {ci[1]:.2f}]).")
    if ci[0] > 0.5:
        return (f"Resampling whole days, the deployed model ranks trades better than chance "
                f"(AUC {a:.2f}, interval [{ci[0]:.2f}, {ci[1]:.2f}]).")
    text = (f"Resampling whole days, the deployed model cannot be told apart from chance "
            f"(AUC {a:.2f}, interval [{ci[0]:.2f}, {ci[1]:.2f}]).")
    if row_ci is not None and (row_ci[1] < 0.5 or row_ci[0] > 0.5):
        text += (" The row-level interval does exclude 0.5, but it ignores that trades on the same "
                 "day move together, so it overstates the evidence.")
    return text


def format_report(result: dict, counts: dict, include_marks: bool) -> str:
    out = ["Deployed ACCEPT/AVOID model, judged on every day (it needs no training)", ""]
    out.append(f"Crossovers saved: {counts['crossovers']}; no Basic trade recorded: {counts['no_basic_trade']}; "
               f"left out as marks: {counts['left_out_marks']}"
               + (" (use --include-marks to count them)" if not include_marks else " (none: marks are included)"))
    if result is None:
        out += ["", "None of the usable crossovers has a saved probability, so there is nothing to score."]
        return "\n".join(out)
    out.append(f"Scored: {result['n']} crossovers on {result['n_days']} days; "
               f"{result['positives']} profitable ({result['positives'] / result['n']:.0%}).")
    fmt = lambda ci: f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "(too few resamples)"
    out += ["", f"AUC {result['auc']:.2f}   (0.5 = guessing; below 0.5 = higher scores went with worse trades)",
            f"  95% interval, resampling whole days:   {fmt(result['day_ci'])}   <- the honest one",
            f"  95% interval, resampling single trades: {fmt(result['row_ci'])}   (too narrow)", "",
            f"By week (weeks with fewer than {MIN_WEEK_ROWS} crossovers are not scored):"]
    for week in result["weeks"]:
        shown = "too few" if np.isnan(week["auc"]) else f"{week['auc']:.2f}"
        out.append(f"  week of {week['week_of']}   {week['n']:>3} crossovers   "
                   f"{week['profitable'] / week['n']:>4.0%} profitable   AUC {shown}")
    out += ["", verdict(result),
            "Weeks that point the same way are more convincing than one stretch that drives the result, "
            "but each week is small. Nothing here changes the deployed model."]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Judge the deployed ACCEPT/AVOID model on every saved day.")
    parser.add_argument("db", help="a COPY of the database (for past data, the corrected copy)")
    parser.add_argument("--include-marks", action="store_true",
                        help="also use crossovers whose Basic trade closed by a session-end or recovery mark")
    parser.add_argument("--resamples", type=int, default=3000)
    args = parser.parse_args(argv)
    if not Path(args.db).exists():
        sys.exit(f"No such file: {args.db}")
    rows, counts = load_rows(args.db, args.include_marks)
    result = evaluate(rows, args.resamples) if rows else None
    print(format_report(result, counts, args.include_marks))
    return 0


if __name__ == "__main__":
    sys.exit(main())