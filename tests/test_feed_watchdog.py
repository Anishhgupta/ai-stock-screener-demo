"""The session loop itself: a feed that goes silent ends the session with the right exit code."""
from datetime import datetime, timedelta, timezone

import main

IST = timezone(timedelta(hours=5, minutes=30))


def when(day, hour, minute, month=11):
    return datetime(2026, month, day, hour, minute, 0, tzinfo=IST)


class RunBroker:
    def __init__(self):
        self.calls = []

    def connect(self):
        self.calls.append("connect")

    def historical_candles(self, *args, **kwargs):
        raise NotImplementedError

    def subscribe(self, callback):
        pass

    def disconnect(self):
        self.calls.append("disconnect")


class RunTrader:
    open_positions = {"a": 1, "b": 2}

    def latest_tick_ts(self):
        return None


def silent_feed_app(monkeypatch, now):
    """An app whose feed has been silent for decades, with the clock fixed at `now`."""
    monkeypatch.setattr(main.os_signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "save_report", lambda trader, path: {})
    monkeypatch.setattr(main, "print_report", lambda report: None)
    app = main.App.__new__(main.App)
    app.broker, app.trader = RunBroker(), RunTrader()
    app._last_tick_at = 0.0
    monkeypatch.setattr(app, "_feed_lost_exit_code", lambda: main.App._feed_lost_exit_code(app, now))
    return app


def test_a_feed_lost_in_market_hours_ends_the_session_with_code_3_and_keeps_the_positions(monkeypatch, capsys):
    app = silent_feed_app(monkeypatch, when(13, 14, 56))
    assert main.run_to_exit_code(app, None) == 3
    out = capsys.readouterr().out
    assert "Leaving 2 open position(s) saved" in out
    assert "code 3" in out and "restart" in out
    assert app.broker.calls[0] == "connect" and "disconnect" in app.broker.calls


def test_a_feed_that_goes_quiet_after_the_close_ends_the_session_normally(monkeypatch, capsys):
    app = silent_feed_app(monkeypatch, when(13, 16, 2))
    assert main.run_to_exit_code(app, None) == 0
    assert "code 3" not in capsys.readouterr().out


def test_a_feed_lost_on_a_weekend_is_not_a_reason_to_restart(monkeypatch):
    app = silent_feed_app(monkeypatch, when(14, 11, 0))
    assert main.run_to_exit_code(app, None) == 0