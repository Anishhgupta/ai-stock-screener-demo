"""
Wording and counts for the dashboard cards: wins, losses and breakevens shown
as plain counts (a win rate on its own hides that an exact breakeven counts as
not a win), the all-time line, and a note when the saved history includes
trades from before the engine stopped trading at 15:15 IST.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from storage.dashboard_data import BOOKS, format_rupees

IST = timezone(timedelta(hours=5, minutes=30))
# The trading day the engine started stopping at 15:15 IST. Before it, positions stayed open
# until 15:30 and some were stopped out or marked at frozen or auction-period prices.
RULE_CHANGE_DAY = "2026-11-04"
PRE_RULE_NOTE = (
    "The all-time lines include trades from before 4 Nov 2026. Until then positions stayed open until "
    "15:30, and some were stopped out or marked at frozen or auction-period prices after 15:15, so "
    "those totals are not reliable.")


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def outcome_text(summary: Optional[dict]) -> str:
    """'3 wins, 4 losses, 2 breakeven' from a summary that carries those counts, else ''."""
    if not summary or any(k not in summary for k in ("wins", "losses", "breakeven")):
        return ""
    if summary["wins"] + summary["losses"] + summary["breakeven"] == 0:
        return "No closed trades yet."
    return (f"{_plural(summary['wins'], 'win', 'wins')}, "
            f"{_plural(summary['losses'], 'loss', 'losses')}, {summary['breakeven']} breakeven")


def all_time_from_rows(rows: Optional[list], limit: int) -> dict:
    """
    Per book counts over every saved trade, or {} when they cannot be complete: no
    database, or as many rows as the read limit (there may be older ones).
    """
    if rows is None or len(rows) >= limit:
        return {}
    summary = {}
    for book in BOOKS:
        pnls = [float(r["pnl"]) for r in rows if r["book"] == book and r["pnl"] is not None]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        summary[book] = {
            "total_trades": len(pnls), "wins": wins, "losses": losses,
            "breakeven": len(pnls) - wins - losses,
            "win_rate": wins / len(pnls) if pnls else 0.0, "total_pnl": sum(pnls),
        }
    return summary


def all_time_line(saved: Optional[dict]) -> str:
    """The all-time caption for one book, from counts or from a plain repository summary."""
    if not saved:
        return ""
    if all(k in saved for k in ("wins", "losses", "breakeven")):
        return (f"All time (saved): {_plural(saved['total_trades'], 'trade', 'trades')}: "
                f"{outcome_text(saved)}. P/L {format_rupees(saved['total_pnl'])}")
    return (f"All time (saved): {saved['total_trades']} trades, {saved['win_rate']:.0%} win rate, "
            f"P/L {format_rupees(saved['total_pnl'])}")


def has_pre_rule_trades(rows: Optional[list]) -> bool:
    """True if any saved trade was entered before the 15:15 rule started (entries with no time are ignored)."""
    for r in rows or []:
        entry = r.get("entry_ts")
        if entry is not None and datetime.fromtimestamp(entry, IST).strftime("%Y-%m-%d") < RULE_CHANGE_DAY:
            return True
    return False