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

from storage.entry_filter import is_mark

DISPLAY_TZ = "Asia/Kolkata"
IST = timezone(timedelta(hours=5, minutes=30))
TODAY_ROW_LIMIT = 5000      # most recent closed trades read to find today's
EXPORT_ROW_LIMIT = 100000   # rows read for the CSV downloads
BOOKS = ("basic", "filtered")
MARKET_OPEN_IST = (9, 15)

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
POSITION_COLUMNS = ["Book", "Symbol", "Signal", "Entry", "Now", "Qty",
                    "P/L now (Rs)", "Entry score", "Monitor"]

# How each table's numbers are shown. These change the display only, never the data.
WATCHLIST_FORMATS = {
    "LTP": "{:.2f}", "SMMA20": "{:.2f}", "SMMA120": "{:.2f}",
    "Bid Px": "{:.2f}", "Ask Px": "{:.2f}",
    "LTQ": "{:,.0f}", "ETQ": "{:,.0f}", "Bid Qty": "{:,.0f}", "Ask Qty": "{:,.0f}",
    "Bid/Ask Imbalance": "{:.3f}", "AI Probability": "{:.3f}",
}
CLOSED_TRADE_FORMATS = {"Entry": "{:.2f}", "Exit": "{:.2f}", "P/L (Rs)": "{:,.2f}"}
DECISION_LOG_FORMATS = {"Probability": "{:.3f}", "P/L (Rs)": "{:,.2f}"}
POSITION_FORMATS = {"Entry": "{:.2f}", "Now": "{:.2f}",
                    "P/L now (Rs)": "{:,.2f}", "Entry score": "{:.3f}"}

# Row colours: dark fills with light text, so they read the same in light and dark mode.
ACCEPT_BG, ACCEPT_FG = "#0f5132", "#75b798"
AVOID_BG, AVOID_FG = "#842029", "#ea868f"
ACCEPT_ROW_CSS = f"background-color: {ACCEPT_BG}; color: {ACCEPT_FG}; font-weight: bold"
AVOID_ROW_CSS = f"background-color: {AVOID_BG}; color: {AVOID_FG}; font-weight: bold"


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


def sort_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    """ACCEPT rows first, then AVOID, then symbols with no verdict yet. Otherwise the order is kept."""
    if df.empty:
        return df
    rank = df["Decision"].map({"ACCEPT": 0, "AVOID": 1}).fillna(2)
    return (df.assign(_rank=rank).sort_values("_rank", kind="stable")
              .drop(columns="_rank").reset_index(drop=True))


def filter_watchlist(df: pd.DataFrame, choice: str) -> pd.DataFrame:
    """Keep only ACCEPT or only AVOID rows; any other choice shows everything."""
    if choice in ("ACCEPT", "AVOID") and not df.empty:
        return df[df["Decision"] == choice].reset_index(drop=True)
    return df


def positions_frame(state: Optional[dict]) -> pd.DataFrame:
    """Open paper positions with the latest price and the profit or loss on paper right now."""
    symbols = (state or {}).get("symbols", {})
    rows = []
    for p in (state or {}).get("open_positions", []):
        ltp = (symbols.get(p["symbol"]) or {}).get("ltp")
        now_pnl = None
        if ltp is not None and p.get("entry_price"):
            direction = 1 if p["signal"] == "BUY" else -1
            now_pnl = direction * (ltp - p["entry_price"]) * p["qty"]
        rows.append({
            "Book": p["book"], "Symbol": p["symbol"], "Signal": p["signal"],
            "Entry": p["entry_price"], "Now": ltp, "Qty": p["qty"],
            "P/L now (Rs)": now_pnl, "Entry score": p.get("entry_prob"),
            "Monitor": p.get("monitor_flag"),
        })
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


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
    return {book: repo.summary(user_id, book) for book in BOOKS}


def _ist_date(epoch_seconds: float):
    """The Indian calendar date of a timestamp: the trading day it belongs to."""
    return datetime.fromtimestamp(epoch_seconds, IST).date()


def recent_rows(repo, user_id, limit: int = TODAY_ROW_LIMIT) -> Optional[list]:
    """The most recent closed trades as dicts, or None if there is no usable database."""
    if repo is None:
        return None
    try:
        return list(repo.recent_closed_positions(user_id, limit))
    except Exception:
        return None   # a briefly locked database must not take the dashboard down


def summarise_today(rows: list, as_of_ts: float) -> dict:
    """Per book, the trades closed on the Indian trading day of `as_of_ts`."""
    day = _ist_date(as_of_ts)
    summary = {}
    for book in BOOKS:
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


def today_summary(repo, user_id, as_of_ts: float, limit: int = TODAY_ROW_LIMIT) -> dict:
    """
    Per book, the trades closed on the Indian trading day of `as_of_ts`, read
    from the database so the numbers survive a restart of the screener.
    Returns {} when there is no usable database; the caller then falls back
    to the live process's own counts.
    """
    rows = recent_rows(repo, user_id, limit)
    return {} if rows is None else summarise_today(rows, as_of_ts)


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


def pnl_curve(rows: list, as_of_ts: float) -> pd.DataFrame:
    """
    Realised profit and loss through the trading day, one step per closed trade,
    for each book (cumulative, Indian time). Empty if nothing closed that day.
    """
    columns = ["Time", "Basic", "Filtered"]
    day = _ist_date(as_of_ts)
    today = [r for r in rows if r["pnl"] is not None and r["book"] in BOOKS
             and _ist_date(r["exit_ts"]) == day]
    if not today:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(today).sort_values("exit_ts", kind="stable")
    times = (pd.to_datetime(df["exit_ts"], unit="s", utc=True)
               .dt.tz_convert(DISPLAY_TZ).dt.tz_localize(None))
    pnl = df["pnl"].astype(float)
    curve = pd.DataFrame({
        "Time": times.to_numpy(),
        "Basic": pnl.where(df["book"] == "basic", 0.0).cumsum().to_numpy(),
        "Filtered": pnl.where(df["book"] == "filtered", 0.0).cumsum().to_numpy(),
    })
    start = pd.Timestamp(day.year, day.month, day.day, *MARKET_OPEN_IST)
    if curve["Time"].min() > start:
        curve = pd.concat([pd.DataFrame({"Time": [start], "Basic": [0.0], "Filtered": [0.0]}),
                           curve], ignore_index=True)
    return curve


def verdict_counts(state: Optional[dict]) -> dict:
    """How many watchlist symbols currently carry an ACCEPT and how many an AVOID verdict."""
    counts = {"ACCEPT": 0, "AVOID": 0}
    for row in (state or {}).get("symbols", {}).values():
        if row.get("decision") in counts:
            counts[row["decision"]] += 1
    return counts


def ranked_trades(frame: pd.DataFrame, n: int = 5, book: Optional[str] = None) -> dict:
    """
    The n best and n worst closed trades by profit or loss. Trades closed by a
    session-end or recovery mark are left out (they are marks, not fills) and
    counted separately. `frame` is closed_trades_frame(); `book` limits it to one book.
    """
    empty = frame.iloc[0:0]
    if frame.empty:
        return {"best": empty, "worst": empty, "used": 0, "marks": 0}
    df = frame if book is None else frame[frame["Book"] == book]
    real = ~df["Reason"].map(is_mark).astype(bool)
    used = df[real]
    return {
        "best": used.nlargest(n, "P/L (Rs)").reset_index(drop=True),
        "worst": used.nsmallest(n, "P/L (Rs)").reset_index(drop=True),
        "used": int(real.sum()),
        "marks": int((~real).sum()),
    }


def _csv_safe(value):
    """Stop a spreadsheet from running a text cell as a formula."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """A table as CSV bytes that Excel opens cleanly (UTF-8 with a byte-order mark)."""
    safe = df.copy()
    for column in safe.columns:
        if not pd.api.types.is_numeric_dtype(safe[column]):
            safe[column] = safe[column].map(_csv_safe)
    return safe.to_csv(index=False).encode("utf-8-sig")


def format_rupees(value) -> str:
    """₹1,234 or -₹1,234; blank for a missing value."""
    if value is None or pd.isna(value):
        return ""
    rounded = round(float(value))
    return f"{'-' if rounded < 0 else ''}₹{abs(rounded):,}"


def status_line(age_seconds: float) -> str:
    """One line telling the person whether the numbers on screen are current."""
    if age_seconds <= 10:
        return f"🟢 Live: last update {age_seconds:.0f}s ago"
    if age_seconds <= 120:
        return f"🟡 Delayed: last update {age_seconds:.0f}s ago"
    minutes = age_seconds / 60
    when = f"{minutes:.0f} min ago" if minutes < 120 else f"{minutes / 60:.1f} h ago"
    return f"🔴 Not updating: last update {when}. The screener has stopped or the market is closed."


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
    coloured (dark fill with light text, so it stays readable in any theme).
    """
    styler = df.style
    if highlight_decisions:
        styler = styler.apply(_highlight_decision, axis=1)
    return styler.format(formats or {}, na_rep="")