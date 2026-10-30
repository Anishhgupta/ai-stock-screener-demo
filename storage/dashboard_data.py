"""
Everything the dashboard needs, as plain functions that return data
(dicts and DataFrames). Kept separate from dashboard/app.py so it can be
unit-tested without running Streamlit.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

DISPLAY_TZ = "Asia/Kolkata"
IST = timezone(timedelta(hours=5, minutes=30))
TODAY_ROW_LIMIT = 5000   # most recent closed trades read to find today's

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

# How each table's numbers are shown. These change the display only, never the data.
WATCHLIST_FORMATS = {
    "LTP": "{:.2f}", "SMMA20": "{:.2f}", "SMMA120": "{:.2f}",
    "Bid Px": "{:.2f}", "Ask Px": "{:.2f}",
    "LTQ": "{:,.0f}", "ETQ": "{:,.0f}", "Bid Qty": "{:,.0f}", "Ask Qty": "{:,.0f}",
    "Bid/Ask Imbalance": "{:.3f}", "AI Probability": "{:.3f}",
}
CLOSED_TRADE_FORMATS = {"Entry": "{:.2f}", "Exit": "{:.2f}", "P/L (Rs)": "{:.2f}"}
DECISION_LOG_FORMATS = {"Probability": "{:.3f}", "P/L (Rs)": "{:.2f}"}
POSITION_FORMATS = {"entry_price": "{:.2f}", "entry_prob": "{:.3f}"}

ACCEPT_ROW_CSS = "background-color: #d4f7d4; color: #1a1a1a"
AVOID_ROW_CSS = "background-color: #f7d4d4; color: #1a1a1a"


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


def _ist_date(epoch_seconds: float):
    """The Indian calendar date of a timestamp: the trading day it belongs to."""
    return datetime.fromtimestamp(epoch_seconds, IST).date()


def today_summary(repo, user_id, as_of_ts: float, limit: int = TODAY_ROW_LIMIT) -> dict:
    """
    Per book, the trades closed on the Indian trading day of `as_of_ts`, read
    from the database so the numbers survive a restart of the screener.
    Returns {} when there is no usable database; the caller then falls back
    to the live process's own counts.
    """
    if repo is None:
        return {}
    try:
        rows = repo.recent_closed_positions(user_id, limit)
    except Exception:
        return {}   # a briefly locked database must not take the dashboard down
    day = _ist_date(as_of_ts)
    summary = {}
    for book in ("basic", "filtered"):
        pnls = [float(r["pnl"]) for r in rows
                if r["book"] == book and r["pnl"] is not None
                and _ist_date(r["exit_ts"]) == day]
        wins = sum(1 for p in pnls if p > 0)
        summary[book] = {
            "total_trades": len(pnls),
            "wins": wins,
            "losses": sum(1 for p in pnls if p < 0),
            "breakeven": sum(1 for p in pnls if p == 0),
            "win_rate": wins / len(pnls) if pnls else 0.0,
            "total_pnl": sum(pnls),
        }
    return summary


def tile_numbers(state_summary: Optional[dict], saved_today: Optional[dict]) -> Optional[dict]:
    """
    Numbers for one book's tiles. Trade counts and P/L come from the saved
    history for today when it is available; the open count is always live.
    Without saved history the live process's own summary is used as is.
    """
    if state_summary is None:
        return None
    if saved_today is None:
        return state_summary
    return {
        "total_trades": saved_today["total_trades"],
        "win_rate": saved_today["win_rate"],
        "open_trades": state_summary["open_trades"],
        "total_pnl": saved_today["total_pnl"],
    }


def _highlight_decision(row):
    if row.get("Decision") == "ACCEPT":
        return [ACCEPT_ROW_CSS] * len(row)
    if row.get("Decision") == "AVOID":
        return [AVOID_ROW_CSS] * len(row)
    return [""] * len(row)


def style_table(df: pd.DataFrame, formats: Optional[dict] = None,
                highlight_decisions: bool = False):
    """
    A display version of a table: numbers formatted as given, missing values
    left blank instead of shown as "None", and optionally ACCEPT/AVOID rows
    coloured (with dark text, so it stays readable in dark mode).
    """
    styler = df.style
    if highlight_decisions:
        styler = styler.apply(_highlight_decision, axis=1)
    return styler.format(formats or {}, na_rep="")