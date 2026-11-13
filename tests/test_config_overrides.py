import ast
import inspect
from dataclasses import asdict

import pytest

import config
from config import ConfigError, ScreenerConfig, apply_env_overrides, parse_setting, parse_watchlist


def test_without_settings_nothing_changes():
    assert apply_env_overrides(ScreenerConfig(), {}) == ScreenerConfig()
    assert apply_env_overrides(ScreenerConfig(), {"SCREENER_MAX_POSITIONS": "  "}) == ScreenerConfig()


def test_each_setting_is_applied_in_the_unit_the_engine_uses():
    cfg = apply_env_overrides(ScreenerConfig(), {
        "SCREENER_CAPITAL_PER_TRADE": "75,000", "SCREENER_STOP_LOSS_PERCENT": "0.8",
        "SCREENER_TARGET_PERCENT": "1.6", "SCREENER_MAX_HOLD_MINUTES": "30",
        "SCREENER_MAX_POSITIONS": "5"})
    assert cfg.capital_per_trade == 75000.0
    assert cfg.stop_loss_pct == 0.008 and cfg.target_pct == 0.016
    assert cfg.max_hold_minutes == 30 and cfg.max_concurrent_positions == 5


def test_only_the_named_settings_change():
    base = asdict(ScreenerConfig())
    changed = asdict(apply_env_overrides(ScreenerConfig(), {"SCREENER_MAX_HOLD_MINUTES": "20"}))
    assert {k for k in base if base[k] != changed[k]} == {"max_hold_minutes"}


@pytest.mark.parametrize("name, bad", [
    ("SCREENER_CAPITAL_PER_TRADE", "lots"), ("SCREENER_CAPITAL_PER_TRADE", "999"),
    ("SCREENER_CAPITAL_PER_TRADE", "nan"), ("SCREENER_STOP_LOSS_PERCENT", "0"),
    ("SCREENER_STOP_LOSS_PERCENT", "50"), ("SCREENER_TARGET_PERCENT", "inf"),
    ("SCREENER_MAX_HOLD_MINUTES", "45.5"), ("SCREENER_MAX_HOLD_MINUTES", "0"),
    ("SCREENER_MAX_POSITIONS", "1000")])
def test_a_bad_value_is_refused_with_a_message_naming_the_setting(name, bad):
    with pytest.raises(ConfigError) as problem:
        apply_env_overrides(ScreenerConfig(), {name: bad})
    assert name in str(problem.value)


def test_parse_setting_returns_the_field_and_the_value_in_its_own_unit():
    assert parse_setting("SCREENER_STOP_LOSS_PERCENT", " 0.6 ") == ("stop_loss_pct", 0.006)
    assert parse_setting("SCREENER_MAX_POSITIONS", "10") == ("max_concurrent_positions", 10)


def test_a_watchlist_is_tidied_and_checked():
    assert parse_watchlist(" nse:sbin-eq, NSE:PNB-EQ ,nse:sbin-eq,, NSE:M&M-EQ") == [
        "NSE:SBIN-EQ", "NSE:PNB-EQ", "NSE:M&M-EQ"]
    for bad in ("SBIN", "NSE:SBIN", "NSE:SBIN-EQ,oops", ",,,", ""):
        with pytest.raises(ConfigError):
            parse_watchlist(bad)
    with pytest.raises(ConfigError):
        parse_watchlist(",".join(f"NSE:S{i}-EQ" for i in range(config.MAX_WATCHLIST + 1)))


def test_the_watchlist_and_broker_settings_are_applied():
    cfg = apply_env_overrides(ScreenerConfig(), {"SCREENER_WATCHLIST": "NSE:SBIN-EQ,NSE:PNB-EQ", "BROKER": " Fyers "})
    assert cfg.watchlist == ["NSE:SBIN-EQ", "NSE:PNB-EQ"] and cfg.broker == "fyers"
    with pytest.raises(ConfigError):
        apply_env_overrides(ScreenerConfig(), {"BROKER": "zerodha"})


def test_importing_the_config_never_applies_the_environment():
    tree = ast.parse(inspect.getsource(config))
    at_import = [node for node in tree.body
                 if "apply_env_overrides" in ast.dump(node) and not isinstance(node, ast.FunctionDef)]
    assert at_import == []