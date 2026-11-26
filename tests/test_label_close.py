"""Training labels must stop at the 15:15 close, as the live engine does, and only when asked to."""
import inspect
from datetime import datetime, timedelta, timezone

import pytest

import main
from brokers.base import Tick
from ml import labeling

IST = timezone(timedelta(hours=5, minutes=30))


def ist(hour, minute, second=0, day=13):                   # Friday 13 Nov 2026
    return datetime(2026, 11, day, hour, minute, second, tzinfo=IST).timestamp()


def tick(symbol, ts, ltp):
    return Tick(symbol=symbol, ts=ts, ltp=ltp, ltq=1, total_traded_qty=1,
                bid_price=ltp, bid_qty=1, ask_price=ltp, ask_qty=1)


class Features:
    def to_feature_vector(self):
        return [0.0] * 11


class Event:
    def __init__(self, signal):
        self.signal, self.features = signal, Features()


class ScriptedEngine:
    """Fires a crossover on exactly these (symbol, timestamp, signal) ticks."""

    def __init__(self, *fires):
        self.fires = {(s, t): sig for s, t, sig in fires}

    def on_tick(self, t):
        sig = self.fires.get((t.symbol, t.ts))
        return Event(sig) if sig else None


def late_day(symbol, jump_to):
    """Entry at 14:40, flat, 100.10 just before 15:15, then the closing-auction jump."""
    ticks = [tick(symbol, ist(14, 40), 100.0)]
    ticks += [tick(symbol, ist(14, 40) + 60 * i, 100.0) for i in range(1, 34)]     # flat, one tick a minute, to 15:13
    ticks += [tick(symbol, ist(15, 14, 59), 100.10),
              tick(symbol, ist(15, 20), jump_to), tick(symbol, ist(15, 25), jump_to)]
    return ticks


def label(ticks, *fires, session_close=None, stats=None):
    return labeling.build_training_set_from_ticks(ticks, ScriptedEngine(*fires), session_close=session_close, stats=stats)


CLOSE = (15, 15)


def test_a_stop_hit_only_by_the_auction_jump_is_not_a_stop_once_the_replay_stops_at_the_close():
    ticks = late_day("A", jump_to=98.0)
    fire = ("A", ist(14, 40), "BUY")
    old, = label(ticks, fire)
    new, = label(ticks, fire, session_close=CLOSE)
    assert old.label == 0 and old.outcome_pct == pytest.approx(-0.006)         # decided by the 15:20 jump
    assert new.label == 1 and new.outcome_pct == pytest.approx(0.001)          # marked at the last price before 15:15


def test_a_short_that_only_won_on_the_auction_jump_is_not_a_win_at_the_close():
    ticks = late_day("A", jump_to=98.0)
    fire = ("A", ist(14, 40), "SELL")
    old, = label(ticks, fire)
    new, = label(ticks, fire, session_close=CLOSE)
    assert old.label == 1 and old.outcome_pct == pytest.approx(0.012)
    assert new.label == 0 and new.outcome_pct == pytest.approx(-0.001)


def test_a_stop_or_target_hit_before_the_close_still_decides_the_label():
    for price, expected in ((101.3, 0.012), (99.3, -0.006)):
        ticks = [tick("A", ist(14, 50), 100.0), tick("A", ist(15, 0), price), tick("A", ist(15, 14, 59), 100.0)]
        fire = ("A", ist(14, 50), "BUY")
        assert label(ticks, fire)[0].outcome_pct == pytest.approx(expected)
        assert label(ticks, fire, session_close=CLOSE)[0].outcome_pct == pytest.approx(expected)


def test_a_tick_exactly_at_the_close_is_not_seen():
    ticks = [tick("A", ist(14, 50), 100.0), tick("A", ist(15, 14, 59), 100.10), tick("A", ist(15, 15, 0), 98.0)]
    new, = label(ticks, ("A", ist(14, 50), "BUY"), session_close=CLOSE)
    assert new.outcome_pct == pytest.approx(0.001)


def test_a_crossover_that_cannot_reach_the_close_is_labelled_exactly_as_before():
    ticks = [tick("A", ist(10, 0), 100.0)] + [tick("A", ist(10, m), 100.0 + 0.01 * m) for m in range(1, 50)]
    fire = ("A", ist(10, 0), "BUY")
    stats = labeling.new_stats()
    old, = label(ticks, fire)
    new, = label(ticks, fire, session_close=CLOSE, stats=stats)
    assert (old.label, old.outcome_pct) == (new.label, new.outcome_pct)
    assert stats["window_crossed_close"] == 0 and stats["label_changed"] == 0 and stats["crossovers"] == 1


def test_crossovers_at_or_after_the_close_are_skipped_only_when_asked():
    ticks = [tick("B", ist(15, 14, 59), 100.0), tick("B", ist(15, 15, 0), 100.0),
             tick("B", ist(15, 16, 0), 100.0), tick("B", ist(15, 25, 0), 100.0)]
    fires = [("B", ist(15, 14, 59), "BUY"), ("B", ist(15, 15, 0), "BUY"), ("B", ist(15, 16, 0), "BUY")]
    stats = labeling.new_stats()
    kept = label(ticks, *fires, session_close=CLOSE, stats=stats)
    assert [e.ts for e in kept] == [ist(15, 14, 59)]                              # 15:15:00 itself is already too late
    assert stats["skipped_after_close"] == 2 and stats["crossovers"] == 1
    assert len(label(ticks, *fires)) == 3                                          # off by default: nothing is skipped


def test_the_stats_count_what_the_close_changed():
    ticks = (late_day("A", 98.0) + late_day("B", 98.0) + late_day("E", 98.0)
             + [tick("C", ist(15, 20), 100.0), tick("C", ist(15, 25), 100.0)]
             + [tick("D", ist(14, 50), 100.0), tick("D", ist(15, 0), 101.3),            # D: target hit before the
                tick("D", ist(15, 14, 59), 101.3), tick("D", ist(15, 20), 90.0)])       # close, so its label stays
    stats = labeling.new_stats()
    label(ticks, ("A", ist(14, 40), "BUY"), ("B", ist(14, 40), "BUY"), ("E", ist(14, 40), "SELL"),
          ("C", ist(15, 20), "BUY"), ("D", ist(14, 50), "BUY"), session_close=CLOSE, stats=stats)
    # A and B were stopped out by the jump but are profitable at the close; E, a short, won only on the jump
    assert stats == {"crossovers": 4, "skipped_after_close": 1, "window_crossed_close": 4,
                     "label_changed": 3, "changed_to_win": 2, "changed_to_loss": 1}


def test_the_close_is_that_days_own_close():
    assert labeling._close_ts(ist(10, 0, day=16), CLOSE) == ist(15, 15, day=16)
    assert labeling._close_ts(ist(15, 14, 59, day=13), CLOSE) == ist(15, 15, day=13)


def test_the_option_reaches_the_csv_route(tmp_path):
    ticks = late_day("A", 98.0)
    path = tmp_path / "ticks.csv"
    path.write_text("symbol,ts,ltp,ltq,total_traded_qty,bid_price,bid_qty,ask_price,ask_qty\n" + "\n".join(
        f"{t.symbol},{t.ts!r},{t.ltp},1,1,{t.ltp},1,{t.ltp},1" for t in ticks), encoding="utf-8")
    engine = lambda: ScriptedEngine(("A", ist(14, 40), "BUY"))
    assert labeling.build_training_set(path, engine=engine())[0].label == 0
    assert labeling.build_training_set(path, engine=engine(), session_close=CLOSE)[0].label == 1


def test_it_is_off_by_default_so_the_demos_synthetic_clock_is_untouched():
    for function in (labeling.build_training_set, labeling.build_training_set_from_ticks):
        assert inspect.signature(function).parameters["session_close"].default is None


def test_the_close_time_matches_the_live_engines():
    assert labeling.MARKET_CLOSE_IST == main.MARKET_CLOSE_IST