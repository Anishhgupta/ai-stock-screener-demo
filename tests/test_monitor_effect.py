"""The monitor-effect report: Filtered trades compared with their Basic twins."""
import math
import statistics
from datetime import datetime, timezone

import pytest

import trading.paper_engine as pe
from brokers.base import Tick
from config import CONFIG
from ml.predict import Decision
from storage.db import make_engine
from storage.monitor_effect import (
    IST, analyze, format_report, load_rows, pair_trades, since_to_epoch,
)
from storage.repository import Repository
from trading.paper_engine import Book, PaperTradingEngine
from types import SimpleNamespace

T0 = 1_800_000_000.0


def row(book, symbol="AAA", pnl=0.0, reason="Target hit", entry_ts=T0,
        price=100.0, signal="BUY"):
    return {"book": book, "symbol": symbol, "signal": signal, "entry_ts": entry_ts,
            "entry_price": price, "exit_ts": entry_ts + 100, "exit_reason": reason, "pnl": pnl}


def cut(symbol, filtered_pnl, basic_pnl, basic_reason="Target hit", entry_ts=T0):
    """A Filtered trade cut by the monitor, plus its Basic twin."""
    return [
        row("filtered", symbol, filtered_pnl, "Deterioration exit (model fell)", entry_ts),
        row("basic", symbol, basic_pnl, basic_reason, entry_ts),
    ]


# ----------------------------------------------------------------- pairing

def test_a_filtered_trade_is_paired_with_the_basic_trade_on_the_same_crossover():
    pairs, unmatched = pair_trades([row("filtered", pnl=-5), row("basic", pnl=600)])
    assert len(pairs) == 1 and unmatched == []
    assert pairs[0][0]["book"] == "filtered" and pairs[0][1]["book"] == "basic"


@pytest.mark.parametrize("basic", [
    row("basic", symbol="OTHER"),
    row("basic", signal="SELL"),
    row("basic", price=101.0),
    row("basic", entry_ts=T0 + 60),     # a different crossover in the same symbol
])
def test_trades_from_a_different_crossover_are_not_paired(basic):
    pairs, unmatched = pair_trades([row("filtered"), basic])
    assert pairs == [] and len(unmatched) == 1


def test_each_basic_trade_is_used_once_and_the_closest_in_time_wins():
    rows = [
        row("basic", entry_ts=T0 + 3, pnl=1),
        row("basic", entry_ts=T0 + 1, pnl=2),
        row("filtered", entry_ts=T0),
        row("filtered", entry_ts=T0 + 0.5),
    ]
    pairs, unmatched = pair_trades(rows)
    assert len(pairs) == 2 and unmatched == []
    assert sorted(b["pnl"] for _, b in pairs) == [1, 2]


def test_since_leaves_out_earlier_filtered_trades():
    rows = [row("filtered", entry_ts=T0 - 1000), row("basic", entry_ts=T0 - 1000),
            row("filtered", entry_ts=T0), row("basic", entry_ts=T0)]
    pairs, unmatched = pair_trades(rows, since_ts=T0 - 10)
    assert len(pairs) == 1 and pairs[0][0]["entry_ts"] == T0


# ----------------------------------------------------------------- analysis

def test_monitor_effect_is_filtered_minus_basic_on_the_trades_it_cut():
    rows = (cut("A", -5.0, 600.0, entry_ts=T0)          # monitor cost 605
            + cut("B", -40.0, -300.0, entry_ts=T0 + 100)  # monitor saved 260
            + cut("C", 0.0, 0.0, entry_ts=T0 + 200))      # no difference
    c = analyze(rows)["monitor_cuts"]
    assert c["n"] == 3
    assert c["filtered_pnl"] == pytest.approx(-45.0)
    assert c["basic_pnl"] == pytest.approx(300.0)
    assert c["effect"] == pytest.approx(-345.0)
    assert (c["helped"], c["hurt"], c["same"]) == (1, 1, 1)
    assert (c["basic_wins"], c["basic_losses"], c["basic_flat"]) == (1, 1, 1)


def test_mean_and_standard_error_of_the_per_trade_effect():
    rows = (cut("A", 0.0, 10.0, entry_ts=T0) + cut("B", 0.0, -20.0, entry_ts=T0 + 100)
            + cut("C", 0.0, 40.0, entry_ts=T0 + 200))
    diffs = [-10.0, 20.0, -40.0]
    c = analyze(rows)["monitor_cuts"]
    assert c["mean_effect"] == pytest.approx(statistics.fmean(diffs))
    assert c["std_error"] == pytest.approx(statistics.stdev(diffs) / math.sqrt(3))
    assert analyze(cut("A", 0.0, 10.0))["monitor_cuts"]["std_error"] is None   # one trade


def test_trades_that_both_books_exited_the_same_way_must_match():
    rows = [row("filtered", "A", 600, "Target hit", T0), row("basic", "A", 600, "Target hit", T0),
            row("filtered", "B", 100, "Max hold time reached", T0 + 100),
            row("basic", "B", 90, "Max hold time reached", T0 + 100)]   # differs
    other = analyze(rows)["other_exits"]
    assert other["n"] == 2 and other["mismatches"] == 1


def test_basic_twins_closed_by_a_session_end_mark_are_counted():
    rows = cut("A", -5.0, 50.0, basic_reason="Session end (marked at last price)")
    assert analyze(rows)["monitor_cuts"]["basic_session_end"] == 1


def test_unmatched_filtered_trades_are_counted_not_dropped():
    rows = cut("A", -5.0, 600.0) + [row("filtered", "LONELY", 0.0, "Deterioration exit (x)", T0 + 500)]
    s = analyze(rows)
    assert (s["filtered_trades"], s["paired"], s["unmatched"]) == (2, 1, 1)


# ------------------------------------------------------------------- report

def test_report_states_the_effect_and_warns_about_small_samples():
    text = format_report(analyze(cut("A", -5.0, 600.0)))
    assert "Cut early by the monitor: 1 trades" in text
    assert "Monitor effect (Filtered - Basic): Rs -605.0" in text
    assert "positive = the monitor helped" in text
    assert "too few to draw conclusions" in text


def test_report_explains_missing_twins():
    rows = cut("A", -5.0, 600.0) + [row("filtered", "LONELY", 0.0, "Deterioration exit (x)", T0 + 500)]
    assert "no recorded Basic twin" in format_report(analyze(rows))


def test_report_messages_for_empty_inputs():
    assert format_report(analyze([])) == "No closed Filtered trades saved yet."
    only_filtered = [row("filtered", "A", 0.0, "Deterioration exit (x)")]
    assert "none has a matching Basic trade" in format_report(analyze(only_filtered))


def test_report_names_the_date_filter_when_it_leaves_nothing():
    text = format_report(analyze([row("filtered"), row("basic")], since_ts=T0 + 10**6), "2030-01-01")
    assert text == "No closed Filtered trades entered on or after 2030-01-01."


def test_since_date_means_midnight_indian_time():
    expected = datetime(2026, 10, 15, 0, 0, tzinfo=IST).timestamp()
    assert since_to_epoch("2026-10-15") == expected
    assert expected == datetime(2026, 10, 14, 18, 30, tzinfo=timezone.utc).timestamp()


# ----------------------------------------- against the real engine and database

def test_the_real_engine_produces_pairs_the_report_can_match(tmp_path, monkeypatch):
    """Monitor cuts a flat Filtered trade; the Basic twin then rides to its target."""
    monkeypatch.setattr(pe, "check_deterioration", lambda *a, **k: "model fell")
    repo = Repository(make_engine(f"sqlite:///{tmp_path / 'effect.db'}"))
    uid = repo.ensure_user()
    engine = PaperTradingEngine(
        model=None, monitor_grace_seconds=10, monitor_confirm_seconds=5,
        on_close=lambda pos: repo.record_closed_position(uid, pos))

    def tick(ts, ltp):
        return Tick(symbol="AAA", ts=ts, ltp=ltp, ltq=1, total_traded_qty=1,
                    bid_price=ltp, bid_qty=1, ask_price=ltp, ask_qty=1)

    event = SimpleNamespace(symbol="AAA", signal="BUY", features=None)
    engine.on_crossover(event, Decision(symbol="AAA", signal="BUY", probability=0.9,
                                        decision="ACCEPT", reasons=["x"]), 100.0, ts=T0)
    live = object()
    engine.on_tick(tick(T0 + 11, 100.0), live)      # warning starts
    engine.on_tick(tick(T0 + 16, 100.0), live)      # persisted, flat: monitor exit (P/L 0)
    engine.on_tick(tick(T0 + 30, 100.0 * (1 + CONFIG.target_pct + 0.001)), live)  # Basic target

    summary = analyze(load_rows(repo, uid))
    cuts = summary["monitor_cuts"]
    assert (summary["paired"], summary["unmatched"], cuts["n"]) == (1, 0, 1)
    assert cuts["filtered_pnl"] == 0
    assert cuts["basic_pnl"] > 0
    assert cuts["effect"] == pytest.approx(-cuts["basic_pnl"])
    assert (cuts["helped"], cuts["hurt"]) == (0, 1)