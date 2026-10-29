"""The entry-filter report links crossovers to Basic trades and compares ACCEPT with AVOID."""
import pytest
from sqlalchemy import create_engine, text

from storage import entry_filter as ef


def event(i, symbol="AAA", signal="BUY", ts=1000.0, decision="ACCEPT", probability=0.9):
    return {"id": i, "symbol": symbol, "signal": signal, "ts": ts,
            "decision": decision, "probability": probability}


def trade(symbol="AAA", signal="BUY", entry_ts=1000.0, pnl=10.0,
          reason="Max hold time reached"):
    return {"symbol": symbol, "signal": signal, "entry_ts": entry_ts,
            "pnl": pnl, "exit_reason": reason}


def outcome(decision, probability, pnl):
    return {"decision": decision, "probability": probability, "pnl": pnl}


def test_event_is_linked_to_the_basic_trade_by_symbol_signal_and_time():
    events = [event(1, "AAA", "BUY", 1000.0, "ACCEPT", 0.9)]
    trades = [
        trade("AAA", "BUY", 1000.0, pnl=25.0),
        trade("BBB", "BUY", 1000.0, pnl=-99.0),    # other symbol
        trade("AAA", "SELL", 1000.0, pnl=-99.0),   # other signal
        trade("AAA", "BUY", 1500.0, pnl=-99.0),    # other time
    ]
    matched, marks, unmatched = ef.match_outcomes(events, trades)
    assert matched == [{"decision": "ACCEPT", "probability": 0.9, "pnl": 25.0}]
    assert (marks, unmatched) == (0, 0)


def test_trades_closed_by_a_mark_are_left_out():
    events = [event(1, "AAA", ts=1000.0), event(2, "BBB", ts=2000.0)]
    trades = [
        trade("AAA", entry_ts=1000.0, reason="Session end (marked at last price)"),
        trade("BBB", entry_ts=2000.0,
              reason="Recovered after restart (marked at last snapshot price)"),
    ]
    matched, marks, unmatched = ef.match_outcomes(events, trades)
    assert matched == []
    assert (marks, unmatched) == (2, 0)


def test_crossovers_without_a_basic_trade_are_counted():
    matched, marks, unmatched = ef.match_outcomes([event(1), event(2, "BBB")], [trade()])
    assert len(matched) == 1
    assert (marks, unmatched) == (0, 1)


def test_auc_values():
    assert ef.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert ef.auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert ef.auc([0.5], [0.5]) == 0.5


def test_group_statistics():
    matched = [outcome("ACCEPT", 0.9, 10.0), outcome("ACCEPT", 0.8, -4.0),
               outcome("AVOID", 0.1, 0.0), outcome("AVOID", 0.2, 6.0)]
    result = ef.analyse(matched, resamples=200)
    assert result["accept"]["n"] == 2 and result["accept"]["wins"] == 1
    assert result["accept"]["total_pnl"] == 6.0
    assert result["avoid"]["n"] == 2 and result["avoid"]["wins"] == 1   # breakeven is not a win
    assert result["avoid"]["total_pnl"] == 6.0


def test_perfect_separation_gives_a_positive_difference_and_auc_of_one():
    matched = ([outcome("ACCEPT", 0.9, 10.0)] * 10 + [outcome("AVOID", 0.1, -10.0)] * 10)
    result = ef.analyse(matched, resamples=200)
    assert result["diff"] == 20.0
    assert result["diff_ci"][0] > 0
    assert result["auc"] == 1.0
    assert result["auc_ci"] == (1.0, 1.0)


def test_bootstrap_is_reproducible():
    matched = [outcome("ACCEPT" if i % 3 else "AVOID", 0.1 * (i % 9), (-1) ** i * (i + 1))
               for i in range(40)]
    assert ef.analyse(matched, resamples=200) == ef.analyse(matched, resamples=200)


def test_empty_report_says_so():
    report = ef.format_report(ef.analyse([], marks=0, unmatched=3))
    assert "No outcomes to report yet." in report
    assert "Crossovers saved: 3" in report


def test_small_groups_get_a_warning():
    matched = [outcome("ACCEPT", 0.9, 5.0), outcome("AVOID", 0.1, -5.0)]
    report = ef.format_report(ef.analyse(matched, resamples=200))
    assert "want at least 30 per group" in report


def test_load_data_reads_only_basic_trades_and_respects_since():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE crossover_events (id INTEGER PRIMARY KEY, symbol TEXT, "
                          "signal TEXT, ts FLOAT, decision TEXT, probability FLOAT)"))
        conn.execute(text("CREATE TABLE paper_positions (id INTEGER PRIMARY KEY, book TEXT, "
                          "symbol TEXT, signal TEXT, entry_ts FLOAT, pnl FLOAT, exit_reason TEXT)"))
        for i, ts in enumerate((100.0, 1000.0, 2000.0), start=1):
            conn.execute(text("INSERT INTO crossover_events VALUES (:i, 'AAA', 'BUY', :ts, 'ACCEPT', 0.9)"),
                         {"i": i, "ts": ts})
        for book, ts in (("basic", 100.0), ("basic", 1000.0), ("filtered", 1000.0)):
            conn.execute(text("INSERT INTO paper_positions (book, symbol, signal, entry_ts, pnl, exit_reason) "
                              "VALUES (:b, 'AAA', 'BUY', :ts, 5.0, 'Max hold time reached')"),
                         {"b": book, "ts": ts})
    events, trades = ef.load_data(engine, since_ts=500.0)
    assert [e["id"] for e in events] == [2, 3]
    assert [t["entry_ts"] for t in trades] == [1000.0]