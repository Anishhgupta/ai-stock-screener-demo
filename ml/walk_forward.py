"""
Walk-forward test: does a model trained on EARLIER days rank LATER days' trades better than chance?

    python -m ml.walk_forward corrected_nov.db
    python -m ml.walk_forward corrected_nov.db --include-marks --min-train 30

Run it on a COPY of the database (for past data, the corrected copy that
remark_trades.py writes, so every outcome follows the 15:15 rule). It never changes
the database and never changes the model that is deployed.

Each crossover is linked to the Basic book's trade (the Basic book enters every
crossover, whatever the model said), and the label is whether that trade made money
(an exact breakeven counts as not profitable). Then, day by day, each candidate model
is trained only on days BEFORE the day it predicts, so nothing it scores was seen in
training. The model that is deployed is scored on the same days from the probability
saved at the time. The AUC is the chance that a profitable trade got a higher score
than an unprofitable one: 0.5 is guessing.

Trades on the same day share market moves, so they are not independent. Intervals
therefore come from resampling whole DAYS (the honest one). The row-by-row interval is
printed for comparison; it is too narrow and is shown only to make that point.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features.engineering import FEATURE_ORDER

from storage.entry_filter import MATCH_TOLERANCE_SECONDS, is_mark

IST = timezone(timedelta(hours=5, minutes=30))
CURRENT = "current model (as deployed)"
MIN_VALID_RESAMPLES = 100


@dataclass
class Row:
    day: str
    ts: float
    symbol: str
    signal: str
    probability: float | None
    features: dict
    profitable: bool


@dataclass
class Fold:
    test_day: str
    n_train: int
    last_train_day: str
    n_test: int


def candidates() -> dict:
    """The models that retraining would produce. The first uses the production settings."""
    return {
        "gradient boosting (production settings)":
            lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0),
        "logistic regression":
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)),
    }


def auc(y, score) -> float:
    """Chance that a profitable trade scored above an unprofitable one (ties count half)."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def parse_features(text):
    """The saved feature dictionary, or None if it is missing, malformed or has a non-finite value."""
    try:
        values = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(values, dict) or not values:
        return None
    for value in values.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
    return {k: float(v) for k, v in values.items()}


def load_rows(db, include_marks: bool = False):
    """(rows, counts): every crossover that has a Basic-book outcome and usable features."""
    con = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    events = [dict(r) for r in con.execute(
        "select symbol, signal, ts, probability, features from crossover_events order by ts")]
    trades = [dict(r) for r in con.execute(
        "select symbol, signal, entry_ts, pnl, exit_reason from paper_positions where book = 'basic'")]
    con.close()

    by_key = defaultdict(list)
    for trade in trades:
        by_key[(trade["symbol"], trade["signal"])].append(trade)
    counts = {"crossovers": len(events), "no_basic_trade": 0, "left_out_marks": 0, "bad_features": 0}
    rows = []
    for event in events:
        trade = next((t for t in by_key.get((event["symbol"], event["signal"]), [])
                      if abs(t["entry_ts"] - event["ts"]) < MATCH_TOLERANCE_SECONDS), None)
        if trade is None:
            counts["no_basic_trade"] += 1
            continue
        if is_mark(trade["exit_reason"]) and not include_marks:
            counts["left_out_marks"] += 1
            continue
        features = parse_features(event["features"])
        if features is None:
            counts["bad_features"] += 1
            continue
        rows.append(Row(
            day=datetime.fromtimestamp(event["ts"], IST).strftime("%Y-%m-%d"), ts=event["ts"],
            symbol=event["symbol"], signal=event["signal"],
            probability=None if event["probability"] is None else float(event["probability"]),
            features=features, profitable=float(trade["pnl"]) > 0))
    return rows, counts


def feature_names(rows) -> list:
    """The model's own features (FEATURE_ORDER), in its own order, that every row has. Extra saved fields are ignored."""
    return [k for k in FEATURE_ORDER if all(k in r.features for r in rows)]


def walk_forward(rows, names, min_train: int = 40, min_class: int = 8):
    """
    Train on days before each day and score that day. Returns (folds, scores, y, days), where
    scores[name] holds a score for every row that was predicted out of time and NaN elsewhere.
    """
    X = np.array([[r.features[k] for k in names] for r in rows], dtype=float)
    y = np.array([r.profitable for r in rows], dtype=bool)
    days = np.array([r.day for r in rows])
    ordinal = {d: datetime.strptime(d, "%Y-%m-%d").toordinal() for d in set(days.tolist())}
    day_number = np.array([ordinal[d] for d in days.tolist()])       # plain integers compare safely in any NumPy
    prob = np.array([np.nan if r.probability is None else r.probability for r in rows], dtype=float)
    scores = {name: np.full(len(rows), np.nan) for name in candidates()}
    scores[CURRENT] = np.full(len(rows), np.nan)
    folds = []
    for day in sorted(ordinal, key=ordinal.get):
        train = np.where(day_number < ordinal[day])[0]               # strictly earlier days only
        test = np.where(day_number == ordinal[day])[0]
        if len(train) < min_train or y[train].sum() < min_class or (~y[train]).sum() < min_class:
            continue
        for name, make in candidates().items():
            model = make()
            model.fit(X[train], y[train])
            scores[name][test] = model.predict_proba(X[test])[:, 1]
        scores[CURRENT][test] = prob[test]
        folds.append(Fold(day, len(train), max(days[train].tolist()), len(test)))
    return folds, scores, y, days


def bootstrap(y, days, scores, kind: str, resamples: int, seed: int):
    """
    95% intervals for each AUC, and for each candidate's AUC minus the deployed model's, by
    resampling whole days (kind="day") or single rows (kind="row"). None if too few resamples
    contained both outcomes.
    """
    rng = np.random.default_rng(seed)
    if kind == "day":
        units = [np.where(days == d)[0] for d in sorted(set(days))]
    else:
        units = [np.array([i]) for i in range(len(y))]
    aucs = {name: [] for name in scores}
    diffs = {name: [] for name in scores if name != CURRENT and CURRENT in scores}
    for _ in range(resamples):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([units[i] for i in pick])
        ys = y[idx]
        if ys.all() or (~ys).all():
            continue
        got = {name: auc(ys, s[idx]) for name, s in scores.items()}
        for name, value in got.items():
            aucs[name].append(value)
        for name in diffs:
            diffs[name].append(got[name] - got[CURRENT])

    def interval(values):
        if len(values) < MIN_VALID_RESAMPLES:
            return None
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)

    return ({n: interval(v) for n, v in aucs.items()}, {n: interval(v) for n, v in diffs.items()})


def analyse(rows, min_train: int = 40, resamples: int = 3000, seed: int = 12345) -> dict:
    names = feature_names(rows)
    folds, scores, y, days = walk_forward(rows, names, min_train) if names else ([], {}, None, None)
    if not folds:
        return {"folds": [], "features": names}
    tested = ~np.isnan(scores[next(iter(candidates()))])
    y, days = y[tested], days[tested]
    scores = {name: s[tested] for name, s in scores.items()}
    if np.isnan(scores[CURRENT]).any():
        del scores[CURRENT]                              # the deployed model's scores were not saved
    day_ci, day_diff = bootstrap(y, days, scores, "day", resamples, seed)
    row_ci, _ = bootstrap(y, days, scores, "row", resamples, seed)
    return {
        "features": names, "folds": folds, "n_tested": int(tested.sum()),
        "n_days": len(set(days)), "positives": int(y.sum()),
        "auc": {name: auc(y, s) for name, s in scores.items()},
        "day_ci": day_ci, "row_ci": row_ci, "day_diff": day_diff,
    }


def verdict(result: dict) -> list:
    lines = []
    ci = result["day_ci"].get(CURRENT)
    if CURRENT in result["auc"]:
        a = result["auc"][CURRENT]
        if ci is None:
            lines.append("The deployed model: too few days to judge.")
        elif ci[1] < 0.5:
            lines.append(f"The deployed model ranked these later trades WORSE than chance "
                         f"(AUC {a:.2f}; the whole day-level interval is below 0.5).")
        elif ci[0] > 0.5:
            lines.append(f"The deployed model ranked these later trades better than chance (AUC {a:.2f}).")
        else:
            lines.append(f"The deployed model cannot be told apart from chance on these days (AUC {a:.2f}).")
    cleared = []
    for name in candidates():
        interval = result["day_ci"].get(name)
        diff = result["day_diff"].get(name)
        beats_chance = interval is not None and interval[0] > 0.5
        beats_current = CURRENT not in result["auc"] or (diff is not None and diff[0] > 0)
        if beats_chance and beats_current:
            cleared.append(name)
    if cleared:
        lines.append("Cleared the bar (lower end of the day-level interval above 0.5"
                     + (" and ahead of the deployed model): " if CURRENT in result["auc"] else "): ")
                     + ", ".join(cleared) + ". Two candidates were tried, so confirm it on days not used here "
                     "before relying on it.")
    else:
        lines.append("No retrained model is distinguishable from chance out of time. Retraining on this "
                     "data would not give a model worth deploying yet.")
    lines.append("Nothing here changes the deployed model.")
    return lines


def format_report(result: dict, counts: dict, n_rows: int, include_marks: bool) -> str:
    out = ["Walk-forward test: does a model trained on earlier days rank later days' trades better than chance?", ""]
    out.append(f"Crossovers saved: {counts['crossovers']}")
    out.append(f"  with a Basic-book outcome and usable features: {n_rows}")
    out.append(f"  no Basic trade recorded: {counts['no_basic_trade']}")
    out.append(f"  left out, Basic trade closed by a mark: {counts['left_out_marks']}"
               + ("  (use --include-marks to count them)" if not include_marks else "  (none: marks are included)"))
    out.append(f"  unusable features: {counts['bad_features']}")
    out.append(f"  features used: {len(result['features'])} of {len(FEATURE_ORDER)}: " + ", ".join(result["features"]))
    if not result["folds"]:
        out += ["", "Not enough history yet: no day has enough earlier days (and both outcomes) to train on."]
        return "\n".join(out)
    first, last = result["folds"][0].test_day, result["folds"][-1].test_day
    out += ["", f"Out-of-time test: {result['n_tested']} crossovers on {result['n_days']} days ({first} to {last}),",
            f"each predicted by a model trained only on earlier days. {result['positives']} were profitable "
            f"({result['positives'] / result['n_tested']:.0%}).", ""]
    header = f"{'':44}{'AUC':>6}   {'95% interval, by day':<22}{'95% interval, by row (too narrow)'}"
    out.append(header)
    order = ([CURRENT] if CURRENT in result["auc"] else []) + list(candidates())
    for name in order:
        def fmt(interval):
            return f"[{interval[0]:.2f}, {interval[1]:.2f}]" if interval else "(too few resamples)"
        out.append(f"{name:44}{result['auc'][name]:>6.2f}   {fmt(result['day_ci'][name]):<22}{fmt(result['row_ci'][name])}")
    if CURRENT in result["auc"]:
        out += ["", "Candidate minus the deployed model, same days (positive = the candidate ranks better):"]
        for name in candidates():
            diff = result["day_diff"].get(name)
            shown = f"[{diff[0]:+.2f}, {diff[1]:+.2f}]" if diff else "(too few resamples)"
            out.append(f"  {name:42}{result['auc'][name] - result['auc'][CURRENT]:>+6.2f}   {shown}")
    out += [""] + verdict(result)
    out += ["", f"Only {result['n_days']} test days: intervals are wide whatever the models do. Trades on the same "
                "day share market moves, which is why the by-day interval is the honest one."]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward test of the ACCEPT/AVOID model on saved live data.")
    parser.add_argument("db", help="a COPY of the database (for past data, the corrected copy)")
    parser.add_argument("--include-marks", action="store_true",
                        help="also use crossovers whose Basic trade closed by a session-end or recovery mark")
    parser.add_argument("--min-train", type=int, default=40, help="smallest number of earlier crossovers to train on")
    parser.add_argument("--resamples", type=int, default=3000)
    args = parser.parse_args(argv)
    if not Path(args.db).exists():
        sys.exit(f"No such file: {args.db}")
    rows, counts = load_rows(args.db, args.include_marks)
    if len(rows) < 2 * args.min_train:
        print(f"Only {len(rows)} usable crossovers; need at least {2 * args.min_train} for a train and a test part.")
        return 1
    result = analyse(rows, args.min_train, args.resamples)
    print(format_report(result, counts, len(rows), args.include_marks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
