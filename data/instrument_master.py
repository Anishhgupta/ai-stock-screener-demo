"""
Builds the symbol -> exchange token mapping AngelOneClient needs to
subscribe to the websocket depth feed.

Angel One publishes a single JSON file listing every tradable instrument
across all exchanges/segments, refreshed daily. We fetch it once, filter
to NSE equity, and cache it locally so we're not re-downloading tens of
MB on every run.

Usage:
    from data.instrument_master import build_token_map
    token_map = build_token_map(["TATAMOTORS", "SBIN", "AXISBANK"])
    # -> {"TATAMOTORS": "3456", "SBIN": "3045", "AXISBANK": "5900"}

Then:
    from brokers.angelone_client import AngelOneClient
    client = AngelOneClient(symbols, token_map=token_map)
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from config import DATA_DIR

INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
)
CACHE_PATH = DATA_DIR / "angelone_instrument_master.json"
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # refresh once a day


def _download_master() -> list[dict]:
    with urllib.request.urlopen(INSTRUMENT_MASTER_URL, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data))
    return data


def _load_master(force_refresh: bool = False) -> list[dict]:
    if not force_refresh and CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            return json.loads(CACHE_PATH.read_text())
    return _download_master()


def build_token_map(symbols: list[str], force_refresh: bool = False) -> dict[str, str]:
    """
    symbols: bare trading symbols WITHOUT exchange prefix or -EQ suffix,
             e.g. "TATAMOTORS", "SBIN" -- matched against Angel One's
             `symbol` field which is typically "TATAMOTORS-EQ".
    Returns: {bare_symbol: token} for every match found on NSE cash (nse_cm).
    """
    master = _load_master(force_refresh=force_refresh)
    wanted = {s.upper().replace("NSE:", "").replace("-EQ", "") for s in symbols}

    token_map: dict[str, str] = {}
    for row in master:
        if row.get("exch_seg") != "NSE":
            continue
        sym = row.get("symbol", "")
        bare = sym.replace("-EQ", "").upper()
        if bare in wanted and row.get("instrumenttype", "") == "":
            # instrumenttype == "" identifies plain equity rows (as opposed
            # to futures/options entries that also carry the base symbol)
            token_map[bare] = row["token"]

    missing = wanted - set(token_map.keys())
    if missing:
        print(f"WARNING: no NSE equity token found for: {sorted(missing)}")
    return token_map


def build_token_map_for_full_names(full_symbols: list[str], force_refresh: bool = False) -> dict[str, str]:
    """
    Convenience wrapper returning a map keyed by the SAME "NSE:XXX-EQ"
    strings used in config.CONFIG.watchlist, matching what
    AngelOneClient.symbols / self.symbols expects.
    """
    bare_map = build_token_map(full_symbols, force_refresh=force_refresh)
    out = {}
    for full in full_symbols:
        bare = full.upper().replace("NSE:", "").replace("-EQ", "")
        if bare in bare_map:
            out[full] = bare_map[bare]
    return out


if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or ["TATAMOTORS", "SBIN", "AXISBANK"]
    mapping = build_token_map(syms, force_refresh=True)
    print(json.dumps(mapping, indent=2))
