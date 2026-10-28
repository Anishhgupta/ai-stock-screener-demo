"""The engine stops trading at the session close.

On 26 Oct 2026 the engine kept managing positions on post-close ticks: max-hold
exits at 15:39 and 15:46, and session-end marks at 15:59, none of them
realistic fills. With session_close=(15, 30) the engine opens nothing from the
first tick at or after 15:30 IST, evaluates no exits, and marks every open
position at the last price seen before the close.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from config import CONFIG
from trading.paper_engine import Book, PaperTradingEngine, SESSION_END_REASON

IST = timezone(timedelta(hours=5, minutes=30))


def at(hour, minute, second=0, day=27):
    return datetime(2026, 10, day, hour, minute, second, tzinfo=IST).timestamp()


def tick(symbol, price, ts):
    return SimpleNamespace(symbol=symbol, ltp=price, ts=ts)


def event(symbol, signal="BUY"):
    return SimpleNamespace(symbol=symbol, signal=signal, features=None)


def accept():
    return SimpleNamespace(decision="ACCEPT", probability=0.9, reasons=["test"])


def make_engine(**kwargs):
    return PaperTradingEngine(None, session_close=(15, 30), **kwargs)


def test_entry_before_close_opens_both_books():
    engine = make_engine()
    engine.on_crossover(event("AAA"), accept(), 100.0, ts=at(15, 29, 50))
    assert (Book.BASIC, "AAA") in engine.open_positions
    assert (Book.FILTERED, "AAA") in engine.open_positions


def test_entry_at_or_after_close_is_ignored():
    recorded = []
    engine = make_engine(on_crossover_event=recorded.append)
    engine.on_crossover(event("AAA"), accept(), 100.0, ts=at(15, 30, 0))
    engine.on_crossover(event("BBB"), accept(), 100.0, ts=at(15, 45))
    assert engine.open_positions == {}
    assert engine.trade_log == []
    assert recorded == []


@pytest.mark.parametrize("post_close_price", [105.0, 90.0])
def test_first_post_close_tick_marks_at_last_pre_close_price(post_close_price):
    engine = make_engine()
    engine.on_crossover(event("AAA"), accept(), 100.0, ts=at(15, 20))
    engine.on_crossover(event("BBB"), accept(), 100.0, ts=at(15, 25))
    engine.on_tick(tick("AAA", 100.4, at(15, 29, 59)), None)
    # A post-close price that would have hit the target (105) or the stop (90)
    # must not be used: everything is marked at the pre-close price.
    engine.on_tick(tick("AAA", post_close_price, at(15, 30, 1)), None)

    assert engine.open_positions == {}
    assert len(engine.closed_positions) == 4
    for pos in engine.closed_positions:
        assert pos.exit_reason == SESSION_END_REASON
        if pos.symbol == "AAA":
            assert pos.exit_price == 100.4
            assert pos.exit_ts == at(15, 29, 59)
        else:
            # BBB had no tick before the close, so it is marked at its entry price.
            assert pos.exit_price == 100.0


def test_max_hold_that_expires_after_the_close_is_a_session_end_mark():
    engine = make_engine()
    entry_ts = at(15, 29) - (CONFIG.max_hold_minutes - 1) * 60
    engine.on_crossover(event("AAA"), accept(), 100.0, ts=entry_ts)
    engine.on_tick(tick("AAA", 100.1, at(15, 29)), None)
    # Past the close, and past the max hold time: it must be a session-end
    # mark, not a "Max hold time reached" exit on a post-close tick.
    engine.on_tick(tick("AAA", 100.1, entry_ts + (CONFIG.max_hold_minutes + 1) * 60), None)

    assert {p.exit_reason for p in engine.closed_positions} == {SESSION_END_REASON}
    assert all(p.exit_ts == at(15, 29) for p in engine.closed_positions)


def test_latest_tick_ts_includes_post_close_ticks():
    engine = make_engine()
    engine.on_tick(tick("AAA", 100.0, at(15, 29)), None)
    engine.on_tick(tick("AAA", 100.0, at(15, 41)), None)
    assert engine.latest_tick_ts() == at(15, 41)


def test_post_close_tick_with_no_open_positions_is_harmless():
    engine = make_engine()
    engine.on_tick(tick("AAA", 100.0, at(15, 31)), None)
    assert engine.closed_positions == []
    assert engine.latest_tick_ts() == at(15, 31)


def test_close_all_after_the_cutoff_close_returns_zero():
    engine = make_engine()
    engine.on_crossover(event("AAA"), accept(), 100.0, ts=at(15, 20))
    engine.on_tick(tick("AAA", 100.0, at(15, 31)), None)
    assert engine.open_positions == {}
    assert engine.close_all() == 0


def test_trading_resumes_the_next_morning():
    engine = make_engine()
    engine.on_crossover(event("AAA"), accept(), 100.0, ts=at(15, 20))
    engine.on_tick(tick("AAA", 100.0, at(15, 31)), None)
    assert engine.open_positions == {}
    engine.on_crossover(event("AAA"), accept(), 100.0, ts=at(9, 20, day=28))
    assert (Book.BASIC, "AAA") in engine.open_positions


def test_without_session_close_nothing_changes():
    engine = PaperTradingEngine(None)
    engine.on_crossover(event("AAA"), accept(), 100.0, ts=at(15, 45))
    assert (Book.BASIC, "AAA") in engine.open_positions
    engine.on_tick(tick("AAA", 105.0, at(15, 46)), None)
    assert {p.exit_reason for p in engine.closed_positions} == {"Target hit"}


def test_session_close_can_be_set_and_cleared_after_construction():
    engine = PaperTradingEngine(None)
    engine.session_close = (15, 30)
    engine.on_crossover(event("AAA"), accept(), 100.0, ts=at(15, 45))
    assert engine.open_positions == {}
    engine.session_close = None
    engine.on_crossover(event("AAA"), accept(), 100.0, ts=at(15, 45))
    assert (Book.BASIC, "AAA") in engine.open_positions