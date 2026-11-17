"""A lost connection during market hours ends in a restartable exit; a restart waits out a network drop."""
import ast
import inspect
from datetime import datetime, timedelta, timezone

import pytest

import main

IST = timezone(timedelta(hours=5, minutes=30))


def when(day, hour, minute, month=11):
    return datetime(2026, month, day, hour, minute, 0, tzinfo=IST)


class FakeTrader:
    def __init__(self, newest_tick=None):
        self.newest_tick = newest_tick

    def latest_tick_ts(self):
        return self.newest_tick


def app_with(newest_tick=None):
    app = main.App.__new__(main.App)          # no broker, database or model needed
    app.trader = FakeTrader(newest_tick)
    return app


def test_a_silent_feed_during_market_hours_asks_for_a_restart():
    assert app_with()._feed_lost_exit_code(when(13, 14, 56)) == main.EXIT_FEED_LOST == 3     # Friday 14:56
    assert app_with()._feed_lost_exit_code(when(13, 9, 0)) == 3                             # the opening minute
    assert app_with()._feed_lost_exit_code(when(13, 15, 14)) == 3                           # last minute before the cutoff


def test_a_silent_feed_outside_market_hours_is_a_normal_end():
    assert app_with()._feed_lost_exit_code(when(13, 15, 15)) == 0       # the cutoff itself
    assert app_with()._feed_lost_exit_code(when(13, 16, 2)) == 0        # the feed going quiet after the close
    assert app_with()._feed_lost_exit_code(when(13, 8, 59)) == 0
    assert app_with()._feed_lost_exit_code(when(14, 11, 0)) == 0        # Saturday
    assert app_with()._feed_lost_exit_code(when(15, 11, 0)) == 0        # Sunday


def test_ticks_from_after_the_cutoff_also_mean_the_session_is_over():
    after_close = when(13, 15, 20).timestamp()
    assert app_with(after_close)._feed_lost_exit_code(when(13, 14, 0)) == 0


class FlakyBroker:
    def __init__(self, failures):
        self.failures = list(failures)
        self.attempts = 0

    def connect(self):
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)


def connecting_app(broker):
    app = main.App.__new__(main.App)
    app.broker = broker
    return app


def test_a_first_try_that_works_never_waits():
    sleeps = []
    broker = FlakyBroker([])
    connecting_app(broker)._connect_with_retries(sleep=sleeps.append)
    assert broker.attempts == 1 and sleeps == []


def test_a_network_drop_is_waited_out_between_tries(capsys):
    sleeps = []
    broker = FlakyBroker([OSError("[Errno -2] Name or service not known")] * 3)
    connecting_app(broker)._connect_with_retries(sleep=sleeps.append)
    assert broker.attempts == 4 and sleeps == [main.CONNECT_WAIT_SECONDS] * 3
    assert "Name or service not known" in capsys.readouterr().out


def test_an_expired_token_fails_at_once_without_waiting():
    sleeps = []
    broker = FlakyBroker([RuntimeError("Fyers auth failed: {'s': 'error', 'code': -16, "
                                       "'message': 'Token is Invalid or Expired'}")])
    with pytest.raises(RuntimeError, match="Token is Invalid"):
        connecting_app(broker)._connect_with_retries(sleep=sleeps.append)
    assert broker.attempts == 1 and sleeps == []


def test_giving_up_after_the_last_attempt_raises_the_last_error():
    sleeps = []
    broker = FlakyBroker([OSError(f"down {i}") for i in range(main.CONNECT_ATTEMPTS)])
    with pytest.raises(OSError, match=f"down {main.CONNECT_ATTEMPTS - 1}"):
        connecting_app(broker)._connect_with_retries(sleep=sleeps.append)
    assert broker.attempts == main.CONNECT_ATTEMPTS and len(sleeps) == main.CONNECT_ATTEMPTS - 1


@pytest.mark.parametrize("message, expected", [
    ("Fyers auth failed: {'message': 'Token is Invalid or Expired'}", True),
    ("invalid token supplied", True),
    ("[Errno -2] Name or service not known", False),
    ("Max retries exceeded with url: /api/v3/profile", False)])
def test_only_a_wrong_login_counts_as_a_login_problem(message, expected):
    assert main.is_login_problem(RuntimeError(message)) is expected


def test_the_exit_code_a_session_ends_with_is_the_apps_own():
    class Lost:
        exit_code = 3

        def run(self, duration):
            pass

    assert main.run_to_exit_code(Lost(), None) == 3


def test_the_broker_is_connected_in_one_place_only():
    tree = ast.parse(inspect.getsource(main))
    app_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "App")
    callers = []
    for method in app_class.body:
        if isinstance(method, ast.FunctionDef):
            for call in ast.walk(method):
                if (isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "connect"
                        and "self.broker" in ast.unparse(call.func)):
                    callers.append(method.name)
    assert callers == ["_connect_with_retries"]