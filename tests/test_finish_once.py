"""The end-of-session routine must run once, even when it is called twice.

Ctrl+C and `docker stop` run it from the SIGINT handler, and the SystemExit
that handler raises then runs the `finally` block in App.run(), which calls
it again. Seen live on 26 Oct 2026: two "Leaving N open position(s)" lines
and two reports from a single stop.
"""
from datetime import datetime

import pytest

import main


class FakeTrader:
    def __init__(self, last_tick_ts, open_positions):
        self._last_tick_ts = last_tick_ts
        self.open_positions = open_positions
        self.close_all_calls = 0

    def latest_tick_ts(self):
        return self._last_tick_ts

    def close_all(self):
        self.close_all_calls += 1
        closed = len(self.open_positions)
        self.open_positions = {}
        return closed


def _ist(hour, minute):
    return datetime(2026, 10, 26, hour, minute, tzinfo=main.IST).timestamp()


def _make_app(monkeypatch, trader):
    saved = []

    def fake_save_report(t, path):
        saved.append(path)
        return {}

    monkeypatch.setattr(main, "save_report", fake_save_report)
    monkeypatch.setattr(main, "print_report", lambda report: None)
    app = main.App.__new__(main.App)
    app.trader = trader
    app._write_state = lambda: None
    return app, saved


def test_finish_after_close_runs_once(monkeypatch):
    trader = FakeTrader(_ist(15, 45), {"a": 1, "b": 2, "c": 3})
    app, saved = _make_app(monkeypatch, trader)
    app._finish()
    app._finish()
    assert trader.close_all_calls == 1
    assert len(saved) == 1


def test_finish_mid_session_prints_leaving_once(monkeypatch, capsys):
    trader = FakeTrader(_ist(11, 0), {"a": 1, "b": 2, "c": 3, "d": 4})
    app, saved = _make_app(monkeypatch, trader)
    app._finish()
    app._finish()
    assert capsys.readouterr().out.count("Leaving 4 open position(s)") == 1
    assert trader.close_all_calls == 0
    assert len(saved) == 1


def test_finish_can_retry_after_a_failure(monkeypatch):
    trader = FakeTrader(_ist(11, 0), {})
    app, _ = _make_app(monkeypatch, trader)
    calls = {"n": 0}

    def flaky_save_report(t, path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")
        return {}

    monkeypatch.setattr(main, "save_report", flaky_save_report)
    with pytest.raises(RuntimeError):
        app._finish()
    app._finish()
    assert calls["n"] == 2
    app._finish()
    assert calls["n"] == 2