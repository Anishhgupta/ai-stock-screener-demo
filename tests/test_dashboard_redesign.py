"""The redesigned dashboard: day curve, verdict counts, ranked trades, CSV export and charts."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from storage import dashboard_data as dd

IST = timezone(timedelta(hours=5, minutes=30))


def ts(day, hour=0, minute=0, second=0, month=10):
    return datetime(2026, month, day, hour, minute, second, tzinfo=IST).timestamp()


def row(book, pnl, exit_ts):
    return {"book": book, "pnl": pnl, "exit_ts": exit_ts}


def test_pnl_curve_steps_through_the_day_for_each_book():
    rows = [row("filtered", 5.0, ts(30, 12, 0)), row("basic", 10.0, ts(30, 10, 0)),
            row("basic", -4.0, ts(30, 11, 0)), row("basic", 99.0, ts(29, 14, 0))]
    curve = dd.pnl_curve(rows, as_of_ts=ts(30, 15, 0))
    assert list(curve.columns) == ["Time", "Basic", "Filtered"]
    assert list(curve["Basic"]) == [0.0, 10.0, 6.0, 6.0]
    assert list(curve["Filtered"]) == [0.0, 0.0, 0.0, 5.0]
    assert curve["Time"].iloc[0] == pd.Timestamp(2026, 10, 30, 9, 15)
    assert curve["Time"].iloc[1] == pd.Timestamp(2026, 10, 30, 10, 0)


def test_pnl_curve_is_empty_when_nothing_closed_today():
    assert dd.pnl_curve([], ts(30, 12)).empty
    assert dd.pnl_curve([row("basic", 5.0, ts(29, 12))], ts(30, 12)).empty


def test_verdict_counts_use_the_latest_decision_per_symbol():
    state = {"symbols": {"a": {"decision": "ACCEPT"}, "b": {"decision": "AVOID"},
                         "c": {"decision": "ACCEPT"}, "d": {"decision": None}, "e": {}}}
    assert dd.verdict_counts(state) == {"ACCEPT": 2, "AVOID": 1}
    assert dd.verdict_counts(None) == {"ACCEPT": 0, "AVOID": 0}


def _watch(decisions):
    return pd.DataFrame({"Symbol": list("ABCDE")[:len(decisions)], "Decision": decisions})


def test_watchlist_puts_verdicts_first_and_keeps_the_rest_in_order():
    df = _watch([None, "AVOID", "ACCEPT", None, "ACCEPT"])
    assert list(dd.sort_watchlist(df)["Symbol"]) == ["C", "E", "B", "A", "D"]
    assert dd.sort_watchlist(df.iloc[0:0]).empty


def test_watchlist_filter():
    df = _watch([None, "AVOID", "ACCEPT"])
    assert list(dd.filter_watchlist(df, "ACCEPT")["Symbol"]) == ["C"]
    assert list(dd.filter_watchlist(df, "AVOID")["Symbol"]) == ["B"]
    assert len(dd.filter_watchlist(df, "All")) == 3


def test_open_positions_show_the_paper_profit_right_now():
    state = {
        "symbols": {"NSE:A-EQ": {"ltp": 102.0}, "NSE:B-EQ": {"ltp": 48.0}},
        "open_positions": [
            {"book": "basic", "symbol": "NSE:A-EQ", "signal": "BUY", "entry_price": 100.0,
             "qty": 10, "monitor_flag": None, "entry_prob": None},
            {"book": "filtered", "symbol": "NSE:B-EQ", "signal": "SELL", "entry_price": 50.0,
             "qty": 100, "monitor_flag": "DETERIORATING: x", "entry_prob": 0.9},
            {"book": "basic", "symbol": "NSE:GONE-EQ", "signal": "BUY", "entry_price": 10.0,
             "qty": 1, "monitor_flag": None, "entry_prob": None},
        ],
    }
    df = dd.positions_frame(state)
    assert list(df.columns) == dd.POSITION_COLUMNS
    assert df.loc[0, "P/L now (Rs)"] == pytest.approx(20.0)      # long: (102 - 100) * 10
    assert df.loc[1, "P/L now (Rs)"] == pytest.approx(200.0)     # short: (50 - 48) * 100
    assert pd.isna(df.loc[2, "P/L now (Rs)"])                    # no price yet for that symbol
    assert dd.positions_frame({}).empty and list(dd.positions_frame({}).columns) == dd.POSITION_COLUMNS


def _closed_frame():
    return pd.DataFrame({
        "Closed at": ["t"] * 6, "Book": ["basic", "basic", "basic", "basic", "filtered", "filtered"],
        "Symbol": list("ABCDEF"), "Signal": ["BUY"] * 6, "Qty": [1] * 6, "Entry": [1.0] * 6, "Exit": [1.0] * 6,
        "Reason": ["Max hold time reached", "Stop-loss hit", "Session end (marked at last price)",
                   "Recovered after restart (marked at last snapshot price)",
                   "Deterioration exit (x)", "Max hold time reached"],
        "P/L (Rs)": [50.0, -30.0, 900.0, -700.0, -10.0, 20.0],
    })


def test_best_and_worst_trades_leave_out_marks():
    ranked = dd.ranked_trades(_closed_frame(), n=2)
    assert list(ranked["best"]["Symbol"]) == ["A", "F"]
    assert list(ranked["worst"]["Symbol"]) == ["B", "E"]
    assert ranked["used"] == 4 and ranked["marks"] == 2


def test_best_and_worst_can_be_limited_to_one_book():
    ranked = dd.ranked_trades(_closed_frame(), n=5, book="filtered")
    assert list(ranked["best"]["Symbol"]) == ["F", "E"]
    assert ranked["used"] == 2 and ranked["marks"] == 0


def test_ranked_trades_with_no_trades():
    ranked = dd.ranked_trades(pd.DataFrame(columns=dd.CLOSED_TRADE_COLUMNS))
    assert ranked["best"].empty and ranked["worst"].empty
    assert (ranked["used"], ranked["marks"]) == (0, 0)


def test_csv_opens_cleanly_in_excel_and_cannot_run_formulas():
    df = pd.DataFrame({"Symbol": ["NSE:A-EQ", "=cmd|' /C calc'!A0"], "P/L (Rs)": [-12.5, 3.0]})
    data = dd.to_csv_bytes(df)
    assert data.startswith(b"\xef\xbb\xbf")                      # UTF-8 byte-order mark
    text = data.decode("utf-8-sig")
    assert "'=cmd" in text and ",=cmd" not in text
    assert "-12.5" in text and "'-12.5" not in text              # real negative numbers stay numbers
    assert text.splitlines()[0] == "Symbol,P/L (Rs)"


def test_rupee_formatting():
    assert dd.format_rupees(1234.4) == "₹1,234"
    assert dd.format_rupees(-1234.6) == "-₹1,235"
    assert dd.format_rupees(-0.4) == "₹0"
    assert dd.format_rupees(None) == "" and dd.format_rupees(float("nan")) == ""


def test_status_line_says_whether_the_numbers_are_current():
    assert dd.status_line(3).startswith("🟢 Live")
    assert dd.status_line(60).startswith("🟡 Delayed")
    stale = dd.status_line(1800)
    assert stale.startswith("🔴 Not updating") and "30 min ago" in stale
    assert "2.5 h ago" in dd.status_line(9000)


def test_no_usable_database_gives_no_rows():
    class Broken:
        def recent_closed_positions(self, user_id, limit):
            raise RuntimeError("database is locked")

    assert dd.recent_rows(None, 1) is None
    assert dd.recent_rows(Broken(), 1) is None


def test_gauge_stays_amber_until_there_are_enough_trades():
    pytest.importorskip("plotly")
    from dashboard import charts
    few = charts.win_rate_gauge(0.9, charts.MIN_TRADES_FOR_COLOUR - 1)
    assert few.data[0].gauge.bar.color == charts.AMBER
    assert charts.win_rate_gauge(0.6, charts.MIN_TRADES_FOR_COLOUR).data[0].gauge.bar.color == charts.GOOD
    assert charts.win_rate_gauge(0.4, charts.MIN_TRADES_FOR_COLOUR).data[0].gauge.bar.color == charts.BAD
    assert "7 trades" in charts.win_rate_gauge(0.5, 7).data[0].title.text


def test_donut_and_pnl_chart():
    pytest.importorskip("plotly")
    from dashboard import charts
    assert list(charts.verdict_donut({"ACCEPT": 0, "AVOID": 0}).data[0].labels) == ["No verdicts yet"]
    assert list(charts.verdict_donut({"ACCEPT": 3, "AVOID": 1}).data[0].values) == [3, 1]
    assert charts.pnl_chart(pd.DataFrame(columns=["Time", "Basic", "Filtered"])) is None
    curve = dd.pnl_curve([row("basic", 5.0, ts(30, 10)), row("filtered", -2.0, ts(30, 11))], ts(30, 15))
    figure = charts.pnl_chart(curve)
    assert [trace.name for trace in figure.data] == ["Basic", "Filtered"]

def test_gauge_labels_have_room_and_the_donut_fills_its_card():
    pytest.importorskip("plotly")
    from dashboard import charts
    gauge = charts.win_rate_gauge(0.5, 10)
    assert list(gauge.data[0].gauge.axis.tickvals) == [0, 50, 100]
    assert gauge.layout.margin.l >= 40 and gauge.layout.margin.r >= 40   # "100" was clipped to "10"
    assert charts.verdict_donut({"ACCEPT": 1, "AVOID": 1}).layout.height >= 280