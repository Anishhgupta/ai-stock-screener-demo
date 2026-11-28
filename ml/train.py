"""
Trains the crossover-quality classifier and saves it to ml/model_store/.

Usage (single day):
    python -m ml.train --tick-csv data/store/ticks_2026-08-18.csv

Usage (combine multiple real trading days):
    python -m ml.train --tick-csv data/store/ticks_2026-08-18.csv data/store/ticks_2026-08-31.csv

Usage (emergency small-sample retrain, e.g. recovering from a broken
model with no way to reach 30 examples today):
    python -m ml.train --tick-csv data/store/ticks_2026-08-31.csv --min-examples 10

The --min-examples override exists so a below-threshold retrain is a
deliberate, visible choice recorded in train_metrics.json (via
"min_examples_required" / "below_recommended_30_example_minimum") -- not
a silent change to the safety check itself. Any model trained this way
should be treated as provisional and replaced once enough real data
accumulates to clear the real 30-example default.

By default the labels stop each replay at 15:15 IST, as the live engine does (see
ml/labeling.py): since 3 Aug 2026 F&O stocks stop trading continuously then, and the
closing auction that follows would otherwise decide the label of a late crossover.
Pass --no-session-cutoff to reproduce the older labels. train_metrics.json records which
was used and how many labels the close changed. NOTE: train_metrics.json is written next
to the model file, so never save an experimental model into the deployed model's folder.

Combining multiple days' data is legitimate here because every example
is reduced to the same broker-agnostic numeric feature vector (SMMA
spread, LTQ acceleration, Bid/Ask imbalance, etc. -- see
features/engineering.py) before training ever sees it. Which broker or
which specific day a crossover came from has no bearing on the model;
only the engineered features and the realized outcome do.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from config import MODEL_DIR
from features.engineering import FEATURE_ORDER
from ml.depth_setting import DEPTH_ENV, announce_depth
from ml.labeling import MARKET_CLOSE_IST, build_training_set, new_stats

DEFAULT_MIN_EXAMPLES = 30


def _clock(session_close):
    return None if session_close is None else f"{session_close[0]:02d}:{session_close[1]:02d}"


def train(tick_csvs: list[Path], model_out: Path = MODEL_DIR / "crossover_model.joblib",
          min_examples: int = DEFAULT_MIN_EXAMPLES, session_close=None):
    examples = []
    per_file_counts = {}
    close_stats = new_stats()
    for csv_path in tick_csvs:
        if session_close is None:
            file_examples = build_training_set(csv_path)
        else:
            file_stats = new_stats()
            file_examples = build_training_set(csv_path, session_close=session_close, stats=file_stats)
            for key, value in file_stats.items():
                close_stats[key] += value
        examples.extend(file_examples)
        per_file_counts[str(csv_path)] = len(file_examples)

    if len(examples) < min_examples:
        raise RuntimeError(
            f"Only {len(examples)} crossover examples found across "
            f"{len(tick_csvs)} file(s) ({per_file_counts}) -- below the "
            f"required minimum of {min_examples}. Need a fuller trading "
            f"day / wider watchlist / more combined days, or pass "
            f"--min-examples to override deliberately (not recommended "
            f"except as a documented, temporary measure)."
        )

    below_recommended = len(examples) < DEFAULT_MIN_EXAMPLES

    X = np.array([e.features for e in examples])
    y = np.array([e.label for e in examples])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else float("nan")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_order": FEATURE_ORDER}, model_out)

    metrics = {
        "source_files": per_file_counts,
        "n_examples": len(examples),
        "n_train": len(X_train), "n_val": len(X_val),
        "val_auc": auc,
        "val_report": report,
        "base_rate_profitable": float(y.mean()),
        "min_examples_required": min_examples,
        "below_recommended_30_example_minimum": below_recommended,
        "session_close": _clock(session_close),
        "close_effect": close_stats if session_close is not None else None,
        "min_depth_qty": os.environ.get(DEPTH_ENV),
    }
    (model_out.parent / "train_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"Trained on {len(examples)} crossovers across {len(tick_csvs)} file(s):")
    for path, count in per_file_counts.items():
        print(f"  {path}: {count} crossovers")
    if session_close is not None:
        print(f"Replay stopped at {_clock(session_close)} IST: {close_stats['skipped_after_close']} crossover(s) "
              f"after the close skipped, {close_stats['window_crossed_close']} replay window(s) ran past it, "
              f"{close_stats['label_changed']} label(s) changed.")
    if below_recommended:
        print(
            f"\nWARNING: {len(examples)} examples is below the recommended "
            f"minimum of {DEFAULT_MIN_EXAMPLES}. This model is PROVISIONAL "
            f"-- treat predictions with reduced confidence and retrain "
            f"properly once more real data accumulates."
        )
    print(f"Base rate profitable: {y.mean():.1%}")
    print(f"Validation AUC: {auc:.3f}")
    print(f"Model saved to {model_out}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tick-csv", required=True, type=Path, nargs="+",
                         help="One or more tick CSV files. Multiple files are combined into one training set.")
    parser.add_argument("--model-out", type=Path, default=MODEL_DIR / "crossover_model.joblib")
    parser.add_argument("--min-examples", type=int, default=DEFAULT_MIN_EXAMPLES,
                         help=f"Minimum crossover examples required to train (default {DEFAULT_MIN_EXAMPLES}). "
                              f"Only lower this deliberately, as a documented emergency measure.")
    parser.add_argument("--no-session-cutoff", action="store_true",
                         help="Reproduce the older labels, whose replay ran on past 15:15 into the closing "
                              "auction. Not recommended.")
    args = parser.parse_args(argv)
    announce_depth()
    return train(args.tick_csv, args.model_out, args.min_examples,
                 None if args.no_session_cutoff else MARKET_CLOSE_IST)


if __name__ == "__main__":
    main()
