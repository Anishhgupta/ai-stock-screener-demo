"""
Everything the dashboard needs, as plain functions that return data
(dicts and DataFrames). Kept separate from dashboard/app.py so it can be
unit-tested without running Streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

DISPLAY_TZ = "Asia/Kolkata"

WATCHLIST_COLUMNS = [
    ("symbol", "Symbol"), ("ltp", "LTP"), ("smma_fast", "SMMA20"),
    ("smma_slow", "SMMA120"), ("signal", "Signal"), ("ltq", "LTQ"),
    ("etq", "ETQ"), ("bid_price", "Bid Px"), ("bid_qty", "Bid Qty"),
    ("ask_price", "Ask Px"), ("ask_qty", "Ask Qty"),
    ("imbalance", "Bid/Ask Imbalance"), ("probability", "AI Probability"),
    ("decision", "Decision"), ("reasons", "Reasons"),
]
CLOSED_TRADE_COLUMNS = ["Closed at", "Book", "Symbol", "Signal", "Qty",
                        "Entry", "Exit", "Reason", "P/L (Rs)"]
DECISION_LOG_COLUMNS = ["Time", "Book", "Symbol", "Signal", "Decision",
                        "Probability", "Outcome", "P/L (Rs)", "Reasons"]


def _join_reasons(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _local_time(epoch_seconds: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(epoch_seconds, unit="s", utc=True).dt.tz_convert(DISPLAY_TZ)
    return stamps.dt.strftime("%Y-%m-%d %H:%M:%S")


def _read_state_file(path: Optional[Path]) -> Optional[dict]:
    if path is None or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_live_state(repo, user_id, state_path: Optional[Path] = None) -> Optional[dict]:
    """
    Newest live state available. Reads the database snapshot and the legacy
    state.json (used by --no-db runs) and returns whichever was updated last.
    """
    candidates = []
    if repo is not None and user_id is not None:
        try:
            candidates.append(repo.load_live_snapshot(user_id))
        except Exception:
            pass  # a briefly locked database must not take the dashboard down
    candidates.append(_read_state_file(state_path))
    valid = [c for c in candidates if isinstance(c, dict) and "updated_at" in c]
    return max(valid, key=lambda s: s["updated_at"], default=None)


def watchlist_frame(state: Optional[dict]) -> pd.DataFrame:
    labels = [label for _, label in WATCHLIST_COLUMNS]
    rows = list((state or {}).get("symbols", {}).values())
    if not rows:
        return pd.DataFrame(columns=labels)
    df = pd.DataFrame(rows).reindex(columns=[key for key, _ in WATCHLIST_COLUMNS])
    df["reasons"] = df["reasons"].apply(_join_reasons)
    df.columns = labels
    return df


def closed_trades_frame(repo, user_id, limit: int = 50) -> pd.DataFrame:
    rows = repo.recent_closed_positions(user_id, limit) if repo is not None else []
    if not rows:
        return pd.DataFrame(columns=CLOSED_TRADE_COLUMNS)
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "Closed at": _local_time(df["exit_ts"]),
        "Book": df["book"], "Symbol": df["symbol"], "Signal": df["signal"],
        "Qty": df["qty"], "Entry": df["entry_price"], "Exit": df["exit_price"],
        "Reason": df["exit_reason"], "P/L (Rs)": pd.to_numeric(df["pnl"]).round(2),
    })


def decision_log_frame(repo, user_id, limit: int = 50) -> pd.DataFrame:
    rows = repo.trade_log(user_id, limit) if repo is not None else []
    if not rows:
        return pd.DataFrame(columns=DECISION_LOG_COLUMNS)
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "Time": _local_time(df["ts"]),
        "Book": df["book"], "Symbol": df["symbol"], "Signal": df["signal"],
        "Decision": df["decision"], "Probability": df["probability"],
        "Outcome": df["outcome"], "P/L (Rs)": pd.to_numeric(df["pnl"]).round(2),
        "Reasons": df["reasons"].apply(_join_reasons),
    })


def all_time_summary(repo, user_id) -> dict:
    if repo is None:
        return {}
    return {book: repo.summary(user_id, book) for book in ("basic", "filtered")}