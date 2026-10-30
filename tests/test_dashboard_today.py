"""Dashboard tiles read today's trades from the database, and tables show
tidy numbers and blanks instead of six decimals and "None"."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from storage import dashboard_data as dd

IST = timezone(timedelta(hours=5, minutes=30))


def ts(day, hour=0, minute=0, second=0, month=10):
    return datetime(2026, month, day, hour, minute, second, tzinfo=IST).timestamp()


def row(book, pnl, exit_ts):
    return {"book": book, "pnl": pnl, "exit_ts": exit_ts}


class FakeRepo:
    def __init__(self, rows):
        self.rows = rows
        self.limits = []

    def recent_closed_positions(self, user_id, limit):
        self.limits.append(limit)
        return self.rows[:limit]


class BrokenRepo:
    def recent_closed_positions(self, user_id, limit):
        raise RuntimeError("database is locked")


def test_only_trades_closed_on_the_indian_day_of_the_snapshot_count():
    rows = [
        row("basic", 100.0, ts(30, 14, 0)),
        row("basic", -40.0, ts(30, 9, 30)),
        row("basic", 0.0, ts(30, 11, 0)),
        row("basic", 500.0, ts(29, 14, 0)),            # yesterday: left out
        row("filtered", 25.0, ts(30, 10, 0)),
        row("filtered", -10.0, ts(29, 15, 29)),         # yesterday: left out
    ]
    s = dd.today_summary(FakeRepo(rows), 1, as_of_ts=ts(30, 15, 0))
    assert s["basic"] == {"total_trades": 3, "wins": 1, "losses": 1, "breakeven": 1,
                          "win_rate": 1 / 3, "total_pnl": 60.0}
    assert s["filtered"]["total_trades"] == 1 and s["filtered"]["wins"] == 1


def test_the_day_changes_at_indian_midnight():
    rows = [row("basic", 10.0, ts(29, 23, 59, 59)), row("basic", 20.0, ts(30, 0, 0, 0))]
    s = dd.today_summary(FakeRepo(rows), 1, as_of_ts=ts(30, 12, 0))
    assert s["basic"]["total_trades"] == 1 and s["basic"]["total_pnl"] == 20.0


def test_no_trades_today_gives_zeros_for_both_books():
    s = dd.today_summary(FakeRepo([row("basic", 5.0, ts(29, 12, 0))]), 1, as_of_ts=ts(30, 12, 0))
    for book in ("basic", "filtered"):
        assert s[book]["total_trades"] == 0
        assert s[book]["win_rate"] == 0.0 and s[book]["total_pnl"] == 0


def test_no_database_or_a_failing_one_gives_an_empty_summary():
    assert dd.today_summary(None, None, as_of_ts=ts(30, 12, 0)) == {}
    assert dd.today_summary(BrokenRepo(), 1, as_of_ts=ts(30, 12, 0)) == {}


def test_tiles_use_saved_numbers_but_the_live_open_count():
    live = {"total_trades": 0, "win_rate": 0.0, "open_trades": 4, "total_pnl": 0.0}
    saved = {"total_trades": 9, "wins": 3, "win_rate": 1 / 3, "total_pnl": 120.0}
    assert dd.tile_numbers(live, saved) == {
        "total_trades": 9, "win_rate": 1 / 3, "open_trades": 4, "total_pnl": 120.0}


def test_tiles_fall_back_to_the_live_numbers_without_saved_history():
    live = {"total_trades": 2, "win_rate": 0.5, "open_trades": 1, "total_pnl": 7.0}
    assert dd.tile_numbers(live, None) is live
    assert dd.tile_numbers(None, {"total_trades": 1}) is None


def _state(**overrides):
    symbol = {"symbol": "NSE:AAA-EQ", "ltp": 68.35, "smma_fast": 68.349091,
              "smma_slow": 68.501889, "signal": None, "ltq": 371, "etq": 3549087,
              "bid_price": 68.3, "bid_qty": 10577, "ask_price": 68.35, "ask_qty": 7751,
              "imbalance": 0.15419, "probability": None, "decision": None, "reasons": []}
    symbol.update(overrides)
    return {"symbols": {"AAA": symbol}}


def test_watchlist_shows_two_decimal_prices_and_blanks():
    pytest.importorskip("jinja2")
    html = dd.style_table(dd.watchlist_frame(_state()), dd.WATCHLIST_FORMATS).to_html()
    assert "68.35" in html and "68.350000" not in html
    assert "68.50" in html
    assert "3,549,087" in html
    assert "0.154" in html
    assert "None" not in html and "nan" not in html.lower()


def test_accept_and_avoid_rows_are_coloured_with_dark_text():
    pytest.importorskip("jinja2")
    accept = dd.style_table(
        dd.watchlist_frame(_state(decision="ACCEPT", probability=0.93, signal="BUY")),
        dd.WATCHLIST_FORMATS, highlight_decisions=True).to_html()
    avoid = dd.style_table(
        dd.watchlist_frame(_state(decision="AVOID", probability=0.07, signal="SELL")),
        dd.WATCHLIST_FORMATS, highlight_decisions=True).to_html()
    assert "#d4f7d4" in accept and "#1a1a1a" in accept and "#f7d4d4" not in accept
    assert "#f7d4d4" in avoid and "#d4f7d4" not in avoid
    assert "0.930" in accept


def test_styling_does_not_change_the_underlying_numbers():
    df = dd.watchlist_frame(_state())
    before = df.copy()
    dd.style_table(df, dd.WATCHLIST_FORMATS, highlight_decisions=True).to_html()
    pd.testing.assert_frame_equal(df, before)
    assert df.loc[0, "SMMA20"] == 68.349091


def test_decision_log_blanks_the_missing_basic_book_fields():
    pytest.importorskip("jinja2")

    class LogRepo:
        def trade_log(self, user_id, limit):
            return [{"ts": ts(30, 10, 0), "book": "basic", "symbol": "NSE:AAA-EQ",
                     "signal": "BUY", "decision": None, "probability": None,
                     "outcome": "OPEN", "pnl": None, "reasons": ["Basic strategy: trade every crossover"]}]

    html = dd.style_table(dd.decision_log_frame(LogRepo(), 1), dd.DECISION_LOG_FORMATS).to_html()
    assert "None" not in html and "nan" not in html.lower()
    assert "Basic strategy" in html