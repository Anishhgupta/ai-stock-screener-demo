"""
Trains a small demo-only ML model from purely synthetic data, so the
Filtered/AI book in demo_app.py shows real ACCEPT/AVOID activity for
any visitor -- no real broker account, real market data, or waiting
required.

IMPORTANT: this model is trained on FAKE, randomly generated price
data. It has no relationship to real market behavior and must never be
confused with, or substituted for, a real trading model trained on
genuine live data (see the main project's ml/train.py for that). Its
only purpose is to make the demo's ML layer visibly functional.

Usage:
    python -m demo.train_demo_model
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("SCREENER_MIN_DEPTH_QTY", "2000")

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from config import CONFIG
from demo.synthetic_history import generate_synthetic_ticks
from features.engineering import FEATURE_ORDER
from ml.labeling import build_training_set_from_ticks
from signals.crossover import CrossoverEngine

DEMO_MODEL_DIR = Path(__file__).resolve().parent / "model_store"
DEMO_MODEL_PATH = DEMO_MODEL_DIR / "demo_model.joblib"


def main():
    print("Generating a full synthetic trading day (in memory, no files)...")
    ticks = generate_synthetic_ticks(CONFIG.watchlist, sim_minutes=1000, ticks_per_minute=6)
    print(f"Generated {len(ticks)} synthetic ticks across {len(CONFIG.watchlist)} symbols.")

    print("Replaying through the real crossover-detection pipeline...")
    engine = CrossoverEngine(persist_ticks=False)
    examples = build_training_set_from_ticks(ticks, engine=engine)
    print(f"Reconstructed {len(examples)} synthetic crossover examples.")

    if len(examples) < 20:
        raise RuntimeError(
            f"Only {len(examples)} synthetic crossovers generated -- too few to "
            f"train even a demo model. Try increasing sim_minutes in this script."
        )

    X = np.array([e.features for e in examples])
    y = np.array([e.label for e in examples])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else float("nan")

    DEMO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_order": FEATURE_ORDER}, DEMO_MODEL_PATH)

    metrics = {
        "n_examples": len(examples),
        "n_train": len(X_train), "n_val": len(X_val),
        "val_auc": auc,
        "base_rate_profitable": float(y.mean()),
        "note": "Trained on SYNTHETIC data for demo purposes only -- not a real trading model.",
    }
    (DEMO_MODEL_DIR / "demo_train_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"\nBase rate profitable (synthetic): {y.mean():.1%}")
    print(f"Validation AUC (synthetic): {auc:.3f}")
    print(f"Demo model saved to {DEMO_MODEL_PATH}")
    print("\nThis file can be safely committed to git and shipped with the repo --")
    print("it contains no real market data or credentials, only a small trained")
    print("scikit-learn model.")


if __name__ == "__main__":
    main()
