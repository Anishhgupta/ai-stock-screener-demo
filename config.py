"""
Central configuration for the screening/paper-trading system.

Nothing secret lives here. Real credentials are read from environment
variables (or a local, git-ignored `.env`) at runtime — see brokers/base.py.
"""
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if getattr(sys, "frozen", False):
    # Running as a PyInstaller --onefile .exe: __file__ would resolve
    # inside the temporary extraction folder (sys._MEIPASS), which is
    # deleted when the exe exits. Anchor persistent data/model/log
    # folders next to the actual .exe on disk instead, so collected
    # ticks and trained models survive between runs.
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "store"
MODEL_DIR = BASE_DIR / "ml" / "model_store"
LOG_DIR = BASE_DIR / "logs"

for d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ScreenerConfig:
    # --- Universe filters ---
    timeframe_minutes: int = 1
    smma_fast: int = 20
    smma_slow: int = 120
    price_min: float = 30.0
    price_max: float = 500.0
    min_bid_qty: int = 10_00_000   # 10 lakh, per assignment spec. Verified
                                    # against a real live session (Aug 17
                                    # 2026): top-of-book quantity alone
                                    # almost never reaches this for NSE
                                    # names. brokers/*.py now sum quantity
                                    # across all 5 published depth levels
                                    # before this filter is applied -- see
                                    # README "On the 10 lakh depth threshold".
    min_ask_qty: int = 10_00_000   # 10 lakh -- same aggregate-depth basis
    exchange: str = "NSE"

    # --- ML decision thresholds ---
    accept_probability: float = 0.62      # >= this -> ACCEPT
    avoid_probability: float = 0.45       # <= this -> AVOID outright
    deteriorate_prob_drop: float = 0.15   # drop in live prob vs entry prob -> DETERIORATING
    ltq_accel_lookback: int = 5           # ticks used to compute LTQ acceleration
    imbalance_lookback: int = 10

    # --- Paper trading ---
    capital_per_trade: float = 50_000.0
    stop_loss_pct: float = 0.006          # 0.6%
    target_pct: float = 0.012             # 1.2%
    max_hold_minutes: int = 45
    max_concurrent_positions: int = 10

    # --- Broker ---
    broker: str = "mock"                  # "fyers" | "angelone" | "mock"
    watchlist: list = field(default_factory=lambda: [
        # Wider NSE watchlist for a real training day -- SMMA20/120 is a
        # slow, infrequent indicator, so ml/train.py needs enough symbols
        # in play to accumulate 30+ crossovers in a single session.
        # Live screening (price ₹30-500, min depth) still applies to
        # every symbol below at runtime -- anything outside the band or
        # too thin is auto-excluded, so it's fine to cast a wide net here.
        # Before market open, sanity-check symbol->token resolution with:
        #   python -m data.instrument_master SBIN PNB IOC ...
        "NSE:SBIN-EQ", "NSE:AXISBANK-EQ", "NSE:PNB-EQ", "NSE:BANKBARODA-EQ",
        "NSE:CANBK-EQ", "NSE:UNIONBANK-EQ", "NSE:IDFCFIRSTB-EQ", "NSE:INDIANB-EQ",
        "NSE:IOB-EQ", "NSE:UCOBANK-EQ", "NSE:MAHABANK-EQ",

        "NSE:TATASTEEL-EQ", "NSE:VEDL-EQ", "NSE:SAIL-EQ", "NSE:NATIONALUM-EQ",
        "NSE:HINDCOPPER-EQ", "NSE:JINDALSTEL-EQ", "NSE:NMDC-EQ",

        "NSE:IOC-EQ", "NSE:BPCL-EQ", "NSE:GAIL-EQ", "NSE:ONGC-EQ",
        "NSE:OIL-EQ", "NSE:PETRONET-EQ",

        "NSE:IDEA-EQ", "NSE:TATAPOWER-EQ", "NSE:NHPC-EQ", "NSE:NTPC-EQ",
        "NSE:PFC-EQ", "NSE:RECLTD-EQ", "NSE:SJVN-EQ", "NSE:JPPOWER-EQ",
        "NSE:RPOWER-EQ", "NSE:SUZLON-EQ",

        "NSE:IRFC-EQ", "NSE:HUDCO-EQ", "NSE:NBCC-EQ", "NSE:RVNL-EQ",
        "NSE:RAILTEL-EQ", "NSE:IRCON-EQ",

        "NSE:YESBANK-EQ", "NSE:IDBI-EQ", "NSE:ETERNAL-EQ",
    ])


CONFIG = ScreenerConfig()


# ---------------------------------------------------------------------
# Settings that can be changed from .env (see setup_wizard.py) without editing
# code. They are applied by main.py when the screener starts, never at import,
# so tests and the demo always see the defaults above.
# ---------------------------------------------------------------------
class ConfigError(ValueError):
    """A setting in .env or the environment is not valid."""


BROKERS = ("mock", "fyers", "angelone")
MAX_WATCHLIST = 200
WATCHLIST_PATTERN = re.compile(r"^[A-Z0-9&_-]+:[A-Z0-9&_-]+-EQ$")

# environment name: (field, kind, lowest, highest, scale into the field's own unit)
SETTING_RULES = {
    "SCREENER_CAPITAL_PER_TRADE": ("capital_per_trade", "float", 1_000.0, 10_000_000.0, 1.0),
    "SCREENER_STOP_LOSS_PERCENT": ("stop_loss_pct", "float", 0.1, 5.0, 0.01),
    "SCREENER_TARGET_PERCENT": ("target_pct", "float", 0.1, 10.0, 0.01),
    "SCREENER_MAX_HOLD_MINUTES": ("max_hold_minutes", "int", 1, 300, 1),
    "SCREENER_MAX_POSITIONS": ("max_concurrent_positions", "int", 1, 100, 1),
}


def parse_setting(name: str, raw) -> tuple:
    """(field, value in the field's unit) for one setting, or ConfigError saying what is wrong."""
    attr, kind, low, high, scale = SETTING_RULES[name]
    text = str(raw).strip().replace(",", "")
    try:
        number = float(text) if kind == "float" else int(text)
    except ValueError:
        what = "a number" if kind == "float" else "a whole number"
        raise ConfigError(f"{name} must be {what}, not {str(raw).strip()!r}.") from None
    if not low <= number <= high:
        raise ConfigError(f"{name} must be between {low:g} and {high:g}, not {text}.")
    return attr, (round(number * scale, 6) if kind == "float" else number)


def parse_watchlist(raw: str) -> list:
    """Symbols like NSE:SBIN-EQ separated by commas: upper-cased, duplicates dropped."""
    symbols = []
    for token in (t.strip().upper() for t in str(raw).split(",")):
        if not token:
            continue
        if not WATCHLIST_PATTERN.match(token):
            raise ConfigError(f"{token!r} is not a symbol in the form NSE:SBIN-EQ.")
        if token not in symbols:
            symbols.append(token)
    if not symbols:
        raise ConfigError("The watchlist is empty.")
    if len(symbols) > MAX_WATCHLIST:
        raise ConfigError(f"The watchlist has {len(symbols)} symbols; the most allowed is {MAX_WATCHLIST}.")
    return symbols


def apply_env_overrides(cfg, environ=None):
    """Apply any settings found in the environment to cfg (a ScreenerConfig) and return it."""
    env = os.environ if environ is None else environ
    for name in SETTING_RULES:
        raw = (env.get(name) or "").strip()
        if raw:
            attr, value = parse_setting(name, raw)
            setattr(cfg, attr, value)
    raw = (env.get("SCREENER_WATCHLIST") or "").strip()
    if raw:
        cfg.watchlist = parse_watchlist(raw)
    raw = (env.get("BROKER") or "").strip().lower()
    if raw:
        if raw not in BROKERS:
            raise ConfigError(f"BROKER must be one of {', '.join(BROKERS)}, not {raw!r}.")
        cfg.broker = raw
    return cfg