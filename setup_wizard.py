"""
Guided first-run setup. It asks a few questions and writes your answers to the
.env file in this folder.

    python setup_wizard.py            ask the questions and write .env
    python setup_wizard.py --check    read-only: show what is set (never the values)

Secrets (passwords, API secrets) are typed hidden, are never printed and never
leave this computer: the only place they are written is .env, which git ignores.
Nothing here places an order. The screener is paper trading only.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

import envfile
from config import (BROKERS, ConfigError, ScreenerConfig, apply_env_overrides,
                    parse_setting, parse_watchlist)

ENV_PATH = Path(__file__).resolve().parent / ".env"
DEFAULT_DEPTH = "2000"
MAX_TRIES = 3

BROKER_NAMES = {
    "mock": "Mock (no credentials needed, simulated prices)",
    "fyers": "Fyers",
    "angelone": "Angel One",
}
# (setting name, question, is it a secret?)
BROKER_FIELDS = {
    "fyers": (
        ("FYERS_CLIENT_ID", "Fyers App ID (looks like ABC12345-100)", False),
        ("FYERS_SECRET_ID", "Fyers Secret ID", True),
        ("FYERS_REDIRECT_URI", "Redirect URL you set on your Fyers app (starts with https://)", False),
    ),
    "angelone": (
        ("ANGEL_API_KEY", "Angel One API key", True),
        ("ANGEL_CLIENT_ID", "Angel One client ID", False),
        ("ANGEL_PASSWORD", "Angel One password or PIN", True),
        ("ANGEL_TOTP_SECRET", "Angel One TOTP secret", True),
    ),
}
SECRET_KEYS = {key for fields in BROKER_FIELDS.values() for key, _, secret in fields if secret}
SETTING_QUESTIONS = (
    ("SCREENER_CAPITAL_PER_TRADE", "Rupees to put into each paper trade"),
    ("SCREENER_STOP_LOSS_PERCENT", "Stop-loss, in percent of the entry price"),
    ("SCREENER_TARGET_PERCENT", "Profit target, in percent of the entry price"),
    ("SCREENER_MAX_HOLD_MINUTES", "Longest a trade is held, in minutes"),
    ("SCREENER_MAX_POSITIONS", "Most positions each book may hold at once"),
)
INTRO = """Welcome. This sets up the screener by asking a few questions and saving the answers
to .env in this folder.

  * It places no orders. The screener is paper trading only.
  * Passwords and API secrets are typed hidden, are never shown again, and are
    written only to .env on this computer.
  * Press Enter to accept the suggestion in [brackets].
"""


class Console:
    """Talks to the person at the keyboard. Tests swap this for a scripted stand-in."""

    def say(self, text: str = "") -> None:
        print(text)

    def ask(self, question: str, default: str = "") -> str:
        hint = f" [{default}]" if default else ""
        return input(f"{question}{hint}: ").strip() or default

    def ask_secret(self, question: str) -> str:
        return getpass.getpass(f"{question} (hidden): ").strip()

    def confirm(self, question: str, default: bool = False) -> bool:
        answer = input(f"{question} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        return default if not answer else answer.startswith("y")


def default_settings() -> dict:
    c = ScreenerConfig()
    return {
        "SCREENER_CAPITAL_PER_TRADE": f"{c.capital_per_trade:g}",
        "SCREENER_STOP_LOSS_PERCENT": f"{c.stop_loss_pct * 100:g}",
        "SCREENER_TARGET_PERCENT": f"{c.target_pct * 100:g}",
        "SCREENER_MAX_HOLD_MINUTES": f"{c.max_hold_minutes}",
        "SCREENER_MAX_POSITIONS": f"{c.max_concurrent_positions}",
    }


def check_depth(text: str) -> int:
    try:
        number = int(str(text).replace(",", ""))
    except ValueError:
        raise ConfigError("Enter a whole number, for example 2000.") from None
    if not 0 <= number <= 10_000_000:
        raise ConfigError("Enter a number between 0 and 10,000,000.")
    return number


def check_url(text: str) -> str:
    if not str(text).lower().startswith(("http://", "https://")):
        raise ConfigError("The redirect URL must start with https:// (or http://).")
    return text


def ask_valid(io, question: str, default: str, check) -> str:
    """Ask until the answer passes `check` (which raises ConfigError); give up after a few tries."""
    for _ in range(MAX_TRIES):
        answer = io.ask(question, default)
        try:
            check(answer)
            return answer
        except ConfigError as problem:
            io.say(f"  {problem}")
    io.say(f"  Keeping {default or 'the current value'}.")
    return default


def choose_broker(io, current: str) -> str:
    io.say("Which broker will supply prices?")
    for number, name in enumerate(BROKERS, 1):
        io.say(f"  {number}) {BROKER_NAMES[name]}")
    default = current if current in BROKERS else "mock"
    for _ in range(MAX_TRIES):
        answer = io.ask("Choose 1, 2 or 3", str(BROKERS.index(default) + 1)).lower()
        if answer in BROKERS:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(BROKERS):
            return BROKERS[int(answer) - 1]
        io.say("  Please type 1, 2 or 3.")
    return default


def normalised_watchlist(text: str):
    try:
        return ",".join(parse_watchlist(text))
    except ConfigError:
        return None


def warn_if_not_ignored(io, env_path: Path) -> None:
    gitignore = env_path.parent / ".gitignore"
    lines = set()
    if gitignore.exists():
        lines = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}
    if not lines & {".env", "/.env", ".env*", "*.env"}:
        io.say("\nWARNING: .gitignore does not list .env. Add a line with .env to it before you "
               "commit anything, or your secrets could be published.")


def next_steps(io, broker: str) -> None:
    io.say("\nNext:")
    step = 1
    if broker == "fyers":
        io.say("  1. Each trading morning run: python fyers_login.py   (it saves that day's token into .env)")
        step = 2
    io.say(f"  {step}. Start it: docker compose --profile live up -d")
    io.say("     (after you change .env later, run docker compose --profile live down first, then up -d)")
    io.say(f"  {step + 1}. Open http://localhost:8502")
    io.say("Never share or commit your .env file.")


def run(io, env_path) -> int:
    env_path = Path(env_path)
    existing = envfile.read_env(env_path)
    updates = {}

    io.say(INTRO)
    broker = choose_broker(io, existing.get("BROKER", "mock"))
    updates["BROKER"] = broker

    for key, label, secret in BROKER_FIELDS.get(broker, ()):
        have = bool(existing.get(key))
        if secret:
            hint = " (press Enter to keep the one already saved)" if have else ""
            value = io.ask_secret(f"{label}{hint}")
        elif key == "FYERS_REDIRECT_URI":
            value = ask_valid(io, label, existing.get(key, ""), check_url)
        else:
            value = io.ask(label, existing.get(key, ""))
        if value:
            updates[key] = value
        elif not have:
            io.say(f"  Skipped {key}. The screener cannot log in until it is set; run this again to add it.")

    io.say("\nReal NSE order books rarely hold the 10 lakh shares the original spec asked for, so the "
           "screener needs a smaller minimum depth or it screens almost every stock out. 2000 works well.")
    depth = ask_valid(io, "Smallest total bid and ask depth, in shares",
                      existing.get("SCREENER_MIN_DEPTH_QTY") or DEFAULT_DEPTH, check_depth)
    updates["SCREENER_MIN_DEPTH_QTY"] = depth.replace(",", "")

    defaults = default_settings()
    if io.confirm("\nChange the trading settings (amount per trade, stop-loss, target, hold time)?", False):
        io.say("  These decide what the experiment measures. Results saved before a change are not "
               "comparable with results after it.")
        for name, question in SETTING_QUESTIONS:
            current = existing.get(name) or defaults[name]
            answer = ask_valid(io, question, current, lambda text, n=name: parse_setting(n, text)).strip()
            if answer != current:
                updates[name] = answer

    count = len(ScreenerConfig().watchlist)
    if io.confirm(f"Use your own list of stocks instead of the built-in {count}?", False):
        io.say("  Type symbols like NSE:SBIN-EQ, separated by commas.")
        text = ask_valid(io, "Symbols", existing.get("SCREENER_WATCHLIST", ""), parse_watchlist)
        listed = normalised_watchlist(text) if text else None
        if listed:
            updates["SCREENER_WATCHLIST"] = listed
    elif existing.get("SCREENER_WATCHLIST") and io.confirm(
            "A custom watchlist is saved. Go back to the built-in list?", False):
        updates["SCREENER_WATCHLIST"] = ""

    changed = {k: v for k, v in updates.items() if existing.get(k) != v}
    if not changed:
        io.say("\nEverything is already set the way you chose. Nothing to change.")
        return 0
    io.say(f"\nThis will be written to {env_path}:")
    for key, value in changed.items():
        shown = "(hidden)" if key in SECRET_KEYS else (value if value != "" else "(cleared)")
        io.say(f"  {key} = {shown}")
    if not io.confirm("Write these now?", True):
        io.say("Nothing was written.")
        return 1
    envfile.update_env(env_path, changed)
    io.say("\nSaved.")
    warn_if_not_ignored(io, env_path)
    next_steps(io, broker)
    return 0


def check(io, env_path) -> int:
    """Read-only: which settings are present (never their values) and whether the numbers are valid."""
    env_path = Path(env_path)
    if not env_path.exists():
        io.say(f"No .env file at {env_path}. Run: python setup_wizard.py")
        return 1
    values = envfile.read_env(env_path)
    broker = (values.get("BROKER") or "mock").lower()
    problems = []
    io.say(f"Checking {env_path} (values are never shown)\n")
    io.say(f"  BROKER: {broker}")
    for key, _, _ in BROKER_FIELDS.get(broker, ()):
        present = bool(values.get(key))
        io.say(f"  {key}: {'set' if present else 'MISSING'}")
        if not present:
            problems.append(f"{key} is missing")
    if broker == "fyers":
        present = bool(values.get("FYERS_ACCESS_TOKEN"))
        io.say("  FYERS_ACCESS_TOKEN: " + ("set (valid for one trading day only)" if present
                                           else "MISSING (run python fyers_login.py)"))
        if not present:
            problems.append("FYERS_ACCESS_TOKEN is missing")
    depth = values.get("SCREENER_MIN_DEPTH_QTY")
    if depth:
        try:
            check_depth(depth)
            io.say("  SCREENER_MIN_DEPTH_QTY: set")
        except ConfigError as problem:
            problems.append(f"SCREENER_MIN_DEPTH_QTY: {problem}")
    else:
        io.say("  SCREENER_MIN_DEPTH_QTY: not set (Docker uses 2000; a run outside Docker needs it)")
    try:
        apply_env_overrides(ScreenerConfig(), values)
    except ConfigError as problem:
        problems.append(str(problem))
    if problems:
        io.say("\nProblems:")
        for problem in problems:
            io.say(f"  - {problem}")
        return 1
    io.say("\nAll good.")
    return 0


def main(argv=None, io=None, env_path=None) -> int:
    parser = argparse.ArgumentParser(description="Guided setup for the screener's .env file.")
    parser.add_argument("--check", action="store_true", help="show what is set without changing anything")
    parser.add_argument("--env", help="path to the .env file (default: the one next to this script)")
    args = parser.parse_args(argv)
    io = io or Console()
    path = Path(args.env) if args.env else (env_path or ENV_PATH)
    try:
        return check(io, path) if args.check else run(io, path)
    except (EOFError, KeyboardInterrupt):
        io.say("\nCancelled. Nothing was written.")
        return 1


if __name__ == "__main__":
    sys.exit(main())