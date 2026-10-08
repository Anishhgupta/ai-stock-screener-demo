"""
A crossover that cannot open a position (one is already open in that
symbol, or the book is full) must be logged as SKIPPED, not OPEN.
Found on a live Fyers run: the decision log showed a basic SELL "OPEN"
that never existed because a BUY in the same symbol was still open.
"""
import time
from types import SimpleNamespace

from config import CONFIG
from ml.predict import Decision
from storage.db import make_engine
from storage.repository import Repository
from trading.paper_engine import Book, PaperTradingEngine

PRICE = 100.0
START = time.time() + 10 * 24 * 3600


def event(symbol="AAA", signal="BUY"):
    return SimpleNamespace(symbol=symbol, signal=signal, features=None)


def decision(kind, symbol="AAA", signal="BUY"):
    if kind == "ACCEPT":
        return Decision(symbol=symbol, signal=signal, probability=0.9,
                        decision="ACCEPT", reasons=["strong bids"])
    return Decision(symbol=symbol, signal=signal, probability=0.2,
                    decision="AVOID", reasons=["Low model confidence"])


def basic_entries(engine):
    return [e for e in engine.trade_log if e.book == Book.BASIC]


def filtered_entries(engine):
    return [e for e in engine.trade_log if e.book == Book.FILTERED]


def test_second_crossover_in_an_open_symbol_is_skipped_not_opened():
    engine = PaperTradingEngine(model=None)
    engine.on_crossover(event(signal="BUY"), decision("AVOID"), PRICE, ts=START)
    engine.on_crossover(event(signal="SELL"), decision("AVOID", signal="SELL"),
                        PRICE + 5, ts=START + 60)

    first, second = basic_entries(engine)
    assert first.outcome == "OPEN"
    assert second.outcome == "SKIPPED"
    assert second.reasons == ["Position already open in this symbol"]
    # The original position is untouched.
    pos = engine.open_positions[(Book.BASIC, "AAA")]
    assert (pos.signal, pos.entry_price) == ("BUY", PRICE)


def test_symbol_can_be_entered_again_after_its_position_closes():
    engine = PaperTradingEngine(model=None)
    engine.on_crossover(event(), decision("AVOID"), PRICE, ts=START)
    engine._close(engine.open_positions[(Book.BASIC, "AAA")], PRICE, START + 10, "test close")
    engine.on_crossover(event(signal="SELL"), decision("AVOID", signal="SELL"),
                        PRICE, ts=START + 20)

    assert [e.outcome for e in basic_entries(engine) if e.outcome != "LOSS"] == ["OPEN", "OPEN"]
    assert engine.open_positions[(Book.BASIC, "AAA")].signal == "SELL"


def test_a_full_book_skips_new_entries():
    engine = PaperTradingEngine(model=None)
    cap = CONFIG.max_concurrent_positions
    for i in range(cap):
        engine.on_crossover(event(f"S{i}"), decision("AVOID", f"S{i}"), PRICE, ts=START + i)
    engine.on_crossover(event("OVERFLOW"), decision("AVOID", "OVERFLOW"), PRICE, ts=START + cap)

    last = basic_entries(engine)[-1]
    assert last.outcome == "SKIPPED"
    assert last.reasons == ["Max concurrent positions reached"]
    assert (Book.BASIC, "OVERFLOW") not in engine.open_positions
    assert len([k for k in engine.open_positions if k[0] == Book.BASIC]) == cap


def test_an_accepted_signal_that_cannot_open_is_skipped_in_the_filtered_book():
    engine = PaperTradingEngine(model=None)
    engine.on_crossover(event(), decision("ACCEPT"), PRICE, ts=START)
    engine.on_crossover(event(signal="SELL"), decision("ACCEPT", signal="SELL"),
                        PRICE, ts=START + 60)

    first, second = filtered_entries(engine)
    assert first.outcome == "OPEN"
    assert second.outcome == "SKIPPED"
    assert second.decision == "ACCEPT"           # the model's verdict is kept
    assert second.reasons[0] == "Position already open in this symbol"
    assert "strong bids" in second.reasons       # ...with its original reasons


def test_skipped_entries_are_not_counted_as_trades():
    engine = PaperTradingEngine(model=None)
    engine.on_crossover(event(), decision("AVOID"), PRICE, ts=START)
    engine.on_crossover(event(signal="SELL"), decision("AVOID", signal="SELL"),
                        PRICE, ts=START + 60)

    assert engine.summary(Book.BASIC)["open_trades"] == 1
    assert engine.summary(Book.BASIC)["total_trades"] == 0


def test_skipped_entries_reach_the_database(tmp_path):
    repo = Repository(make_engine(f"sqlite:///{tmp_path / 'skip.db'}"))
    uid = repo.ensure_user()
    engine = PaperTradingEngine(
        model=None, on_log=lambda entry: repo.record_trade_log(uid, entry))
    engine.on_crossover(event(), decision("AVOID"), PRICE, ts=START)
    engine.on_crossover(event(signal="SELL"), decision("AVOID", signal="SELL"),
                        PRICE, ts=START + 60)

    stored = [r for r in repo.trade_log(uid) if r["book"] == "basic"]
    assert sorted(r["outcome"] for r in stored) == ["OPEN", "SKIPPED"]