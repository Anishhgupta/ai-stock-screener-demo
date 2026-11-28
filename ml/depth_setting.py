"""
The minimum-depth setting for command-line runs that replay saved ticks (ml.labeling, ml.train).

The depth screen reads SCREENER_MIN_DEPTH_QTY from the environment. Docker and main.py load
.env, but these commands do not, so without this a run silently falls back to the literal
10 lakh of the original spec, which almost no real NSE order book reaches: the replay then
finds almost no crossovers and reports a result that means nothing.
"""
from __future__ import annotations

import os
from pathlib import Path

from envfile import read_env

DEPTH_ENV = "SCREENER_MIN_DEPTH_QTY"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def depth_setting(env_file=None, environ=None):
    """
    (value, source) for the minimum depth. The environment wins; otherwise the value is taken
    from .env and put into the environment so the depth screen sees it. (None, "not set") if
    neither has one.
    """
    environ = os.environ if environ is None else environ
    env_file = ENV_FILE if env_file is None else env_file
    if environ.get(DEPTH_ENV):
        return environ[DEPTH_ENV], "the environment"
    value = read_env(env_file).get(DEPTH_ENV)
    if value:
        environ[DEPTH_ENV] = value
        return value, str(env_file)
    return None, "not set"


def announce_depth(env_file=None, environ=None, out=print):
    """Say which minimum depth is in force, or warn loudly that none is. Returns the value or None."""
    value, source = depth_setting(env_file, environ)
    if value:
        out(f"Minimum depth: {value} shares (from {source}).")
    else:
        out("WARNING: SCREENER_MIN_DEPTH_QTY is not set, so the depth screen uses the literal 10 lakh "
            "shares of the original spec, which almost no real order book reaches. The replay will find "
            "almost no crossovers and the result will mean nothing. Run python setup_wizard.py, or "
            "set SCREENER_MIN_DEPTH_QTY=2000 in this window, and run again.")
    return value