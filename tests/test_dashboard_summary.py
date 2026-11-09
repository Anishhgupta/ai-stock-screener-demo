"""Wins, losses and breakevens as plain counts, the all-time line, and the note about older history."""
from datetime import datetime, timedelta, timezone

import pytest

from storage import dashboard_summary as ds

IST = timezone(timedelta(hours=5, minutes=30))


def at(day, hour=0, minute=0, second=0, month=11):
    return datetime(2026, month, day, hour, minute, second, tzinfo=IST).timestamp()


def row(book, pnl, entry_ts=None):
    return {"book": book, "pnl": pnl, "entry_ts": entry_ts}


def test_outcome_text_counts_each_kind_and_gets_the_plurals_right():
    assert ds.outcome_text({"wins": 3, "losses": 4, "breakeven": 2}) == "3 wins, 4 losses, 2 breakeven"
    assert ds.outcome_text({"wins": 1, "losses": 1, "breakeven": 0}) == "1 win, 1 loss, 0 breakeven"


def test_outcome_text_with_no_trades_or_no_counts():
    assert ds.outcome_text({"wins": 0, "losses": 0, "breakeven": 0}) == "No closed trades yet."
    assert ds.outcome_text({"total_trades": 5}) == ""
    assert ds.outcome_text(None) == ""


def test_all_time_counts_treat_an_exact_zero_as_a_breakeven_not_a_loss():
    rows = [row("basic", 10.0), row("basic", -5.0), row("basic", 0.0), row("basic", None),
            row("filtered", 0.0), row("filtered", 0.0), row("filtered", 7.0)]
    s = ds.all_time_from_rows(rows, limit=5000)
    assert s["basic"] == {"total_trades": 3, "wins": 1, "losses": 1, "breakeven": 1,
                          "win_rate": pytest.approx(1 / 3), "total_pnl": 5.0}
    assert (s["filtered"]["wins"], s["filtered"]["losses"], s["filtered"]["breakeven"]) == (1, 0, 2)


def test_all_time_counts_are_withheld_when_they_might_be_incomplete():
    assert ds.all_time_from_rows(None, limit=10) == {}
    assert ds.all_time_from_rows([row("basic", 1.0)] * 10, limit=10) == {}      # as many rows as the limit
    assert ds.all_time_from_rows([row("basic", 1.0)] * 9, limit=10)["basic"]["total_trades"] == 9


def test_all_time_line_from_counts_and_from_a_plain_summary():
    full = {"total_trades": 130, "wins": 59, "losses": 69, "breakeven": 2, "win_rate": 0.45, "total_pnl": -3542.0}
    assert ds.all_time_line(full) == "All time (saved): 130 trades: 59 wins, 69 losses, 2 breakeven. P/L -₹3,542"
    plain = {"total_trades": 87, "win_rate": 0.46, "total_pnl": -2126.0}
    assert ds.all_time_line(plain) == "All time (saved): 87 trades, 46% win rate, P/L -₹2,126"
    assert ds.all_time_line({}) == "" and ds.all_time_line(None) == ""
    one = {"total_trades": 1, "wins": 1, "losses": 0, "breakeven": 0, "win_rate": 1.0, "total_pnl": 5.0}
    assert ds.all_time_line(one).startswith("All time (saved): 1 trade: 1 win, 0 losses")


def test_the_note_about_older_history_depends_on_the_day_the_rule_started():
    day = ds.RULE_CHANGE_DAY
    assert day == "2026-11-04"
    assert ds.has_pre_rule_trades([row("basic", 1.0, at(3, 23, 59, 59))]) is True
    assert ds.has_pre_rule_trades([row("basic", 1.0, at(4, 0, 0, 0))]) is False
    assert ds.has_pre_rule_trades([row("basic", 1.0, at(5, 10, 0))]) is False


def test_the_note_is_not_shown_without_entry_times_or_trades():
    assert ds.has_pre_rule_trades(None) is False
    assert ds.has_pre_rule_trades([]) is False
    assert ds.has_pre_rule_trades([row("basic", 1.0, None), {"book": "basic", "pnl": 1.0}]) is False


def test_the_note_explains_why_the_old_totals_cannot_be_trusted():
    assert "4 Nov 2026" in ds.PRE_RULE_NOTE and "15:15" in ds.PRE_RULE_NOTE and "not reliable" in ds.PRE_RULE_NOTE