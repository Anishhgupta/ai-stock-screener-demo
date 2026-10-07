"""
Entry point. Wires broker -> tick pipeline -> crossover detection ->
ML scoring -> paper trading -> live dashboard state, all in one process.

Run with the mock broker (no credentials needed) to see the whole system
working end to end:
    python main.py --broker mock --duration 120

Once you have a trained model and real broker creds exported as env vars:
    python main.py --broker fyers
    python main.py --broker angelone

Paper trades and the decision log are saved to a local SQLite database
(data/store/screener.db) so history survives restarts. Set SCREENER_DB_URL
to use another database, or pass --no-db to run without persistence.

The dashboard (Streamlit) reads the live snapshot this script saves to the
database about once a second (state.json is still written as a fallback) --
run it separately:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import argparse
import json
import logging
import signal as os_signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # reads .env in the current working directory, if present,
                # and exports ANGEL_*/FYERS_* into os.environ before any
                # broker adapter tries to read them via get_env()

from brokers.base import build_broker, Tick
from config import CONFIG, BASE_DIR
from data.tick_store import TickStore
from features.engineering import build_feature_snapshot
from ml.predict import CrossoverModel, Decision
from reports.performance import save_report, print_report
from signals.crossover import CrossoverEngine
from storage.db import make_engine
from storage.repository import Repository
from trading.paper_engine import PaperTradingEngine, Book

STATE_PATH = BASE_DIR / "dashboard" / "state.json"
logger = logging.getLogger(__name__)


class App:
    def __init__(self, broker_name: str, use_db: bool = True, db_url: str | None = None):
        self.engine = CrossoverEngine()
        self.model = CrossoverModel()
        self.trader = self._make_trader(use_db, db_url)
        self.broker = self._build_broker(broker_name)
        self._last_state_write = 0.0
        self._latest_by_symbol: dict[str, dict] = {}
        self._last_tick_at: float = time.time()

    def _make_trader(self, use_db: bool, db_url: str | None = None) -> PaperTradingEngine:
        """Paper-trading engine, optionally wired to the database."""
        if not use_db:
            self.repo, self.user_id = None, None
            return PaperTradingEngine(self.model)
        self.repo = Repository(make_engine(db_url))
        self.user_id = self.repo.ensure_user("local")
        return PaperTradingEngine(
            self.model,
            on_close=lambda pos: self.repo.record_closed_position(self.user_id, pos),
            on_log=lambda entry: self.repo.record_trade_log(self.user_id, entry),
        )

    def _build_broker(self, broker_name: str):
        if broker_name == "angelone":
            from data.instrument_master import build_token_map_for_full_names
            from brokers.angelone_client import AngelOneClient
            print("Fetching Angel One instrument master to resolve symbol tokens...")
            token_map = build_token_map_for_full_names(CONFIG.watchlist)
            if len(token_map) < len(CONFIG.watchlist):
                print(
                    f"WARNING: only resolved {len(token_map)}/{len(CONFIG.watchlist)} "
                    "watchlist symbols to tokens -- unresolved symbols won't stream."
                )
            return AngelOneClient(CONFIG.watchlist, token_map=token_map)
        return build_broker(broker_name, CONFIG.watchlist)

    def _decide(self, event) -> Decision:
        """
        ACCEPT/AVOID for a crossover. With no trained model, return a
        placeholder AVOID instead of skipping the trading logic, so the
        Basic book still trades every crossover.
        """
        if self.model.is_available:
            return self.model.decide(event)
        return Decision(
            symbol=event.symbol, signal=event.signal, probability=None,
            decision="AVOID", reasons=["No trained model available"],
        )

    def handle_tick(self, tick: Tick):
        self._last_tick_at = time.time()
        event = self.engine.on_tick(tick)

        live_feat = build_feature_snapshot(
            symbol=tick.symbol,
            recent_ticks=self.engine.tick_store.recent(tick.symbol, 50),
            recent_candles=self.engine.candles.closed_candles(tick.symbol, 30),
            smma_pair=self.engine.smma.get(tick.symbol),
        )

        decision = None
        if event is not None:
            decision = self._decide(event)
            self.trader.on_crossover(event, decision, tick.ltp)

        self.trader.on_tick(tick, live_feat)

        # A crossover is a single-tick event, so signal/probability/decision/
        # reasons carry forward from the last real value for this symbol
        # instead of going blank on every ordinary tick.
        existing = self._latest_by_symbol.get(tick.symbol, {})
        self._latest_by_symbol[tick.symbol] = {
            "symbol": tick.symbol,
            "ltp": tick.ltp,
            "smma_fast": self.engine.smma.get(tick.symbol).fast.value,
            "smma_slow": self.engine.smma.get(tick.symbol).slow.value,
            "ltq": tick.ltq,
            "etq": tick.total_traded_qty,
            "bid_price": tick.bid_price, "bid_qty": tick.bid_qty,
            "ask_price": tick.ask_price, "ask_qty": tick.ask_qty,
            "imbalance": live_feat.bid_ask_imbalance if live_feat else None,
            "signal": event.signal if event is not None else existing.get("signal"),
            "probability": decision.probability if decision is not None else existing.get("probability"),
            "decision": decision.decision if decision is not None else existing.get("decision"),
            "reasons": decision.reasons if decision is not None else existing.get("reasons", []),
            "ts": tick.ts,
        }

        if time.time() - self._last_state_write > 1.0:
            self._write_state()
            self._last_state_write = time.time()

    def _write_state(self):
        state = {
            "updated_at": time.time(),
            "symbols": self._latest_by_symbol,
            "open_positions": [
                {
                    "book": p.book.value, "symbol": p.symbol, "signal": p.signal,
                    "entry_price": p.entry_price, "qty": p.qty,
                    "monitor_flag": p.monitor_flag, "entry_prob": p.entry_prob,
                }
                for p in self.trader.open_positions.values()
            ],
            "basic_summary": self.trader.summary(Book.BASIC),
            "filtered_summary": self.trader.summary(Book.FILTERED),
        }
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
        self._save_snapshot(state)

    def _save_snapshot(self, state: dict):
        """Mirror the dashboard state into the database (best effort)."""
        if self.repo is None:
            return
        try:
            self.repo.save_live_snapshot(self.user_id, state)
        except Exception:
            # A database hiccup must never kill the broker callback thread.
            logger.exception("Could not save live snapshot to the database; continuing")

    def _warm_start_from_history(self):
        """
        Attempt to prime SMMA20/120 from the broker's historical-candle
        REST endpoint (if it supports one) instead of starting cold and
        needing ~2 hours of live candles before crossovers can even be
        detected. Safe to call regardless of broker -- silently does
        nothing for brokers that don't implement historical_candles().
        """
        import time as _time
        print("Attempting to warm-start SMMA from historical candles...")
        warmed = 0
        for symbol in CONFIG.watchlist:
            try:
                candles = self.broker.historical_candles(
                    symbol, minutes_back=CONFIG.smma_slow + 30
                )
            except NotImplementedError:
                print("  Broker has no historical-candle support -- skipping warm-start entirely.")
                return
            except Exception as e:
                print(f"  Warm-start fetch failed for {symbol}: {e}")
                continue
            n = self.engine.warm_start(symbol, candles)
            if n > 0:
                warmed += 1
            _time.sleep(0.25)  # be polite to the broker's REST rate limits across 40+ symbols
        print(f"Warm-started SMMA for {warmed}/{len(CONFIG.watchlist)} symbols from historical data.")

    def run(self, duration_seconds: int | None):
        self.broker.connect()
        self._warm_start_from_history()
        stop_at = time.time() + duration_seconds if duration_seconds else None

        def _sigint(*_):
            print("\nStopping...")
            self.broker.disconnect()
            self._finish()
            sys.exit(0)

        os_signal.signal(os_signal.SIGINT, _sigint)

        # Real broker websockets (Fyers/Angel One) run their own blocking
        # event loop inside subscribe() and never return on their own, so
        # subscribe() always runs in a background thread here -- this is
        # what makes --duration work uniformly whether the broker's
        # subscribe() blocks (Fyers/Angel One) or returns immediately
        # after spawning its own thread (the mock broker).
        import threading
        sub_thread = threading.Thread(target=self.broker.subscribe, args=(self.handle_tick,), daemon=True)
        sub_thread.start()

        # Stale-feed watchdog: if the broker's websocket has died (max
        # reconnect retries exhausted, e.g. after market close) and no
        # ticks have arrived for this long, stop automatically instead of
        # sitting in an idle loop forever with no data coming in.
        STALE_FEED_TIMEOUT = 120  # seconds

        try:
            while stop_at is None or time.time() < stop_at:
                if time.time() - self._last_tick_at > STALE_FEED_TIMEOUT:
                    print(
                        f"\nNo ticks received for over {STALE_FEED_TIMEOUT}s -- "
                        "the feed appears to have disconnected (e.g. after "
                        "market close). Stopping automatically."
                    )
                    break
                time.sleep(0.5)
        finally:
            self.broker.disconnect()
            self._finish()

    def _finish(self):
        report = save_report(self.trader, BASE_DIR / "reports" / "latest_report.json")
        print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default=CONFIG.broker, choices=["mock", "fyers", "angelone"])
    parser.add_argument("--duration", type=int, default=None, help="Seconds to run (omit for indefinite/until Ctrl+C)")
    parser.add_argument("--no-db", action="store_true", help="Run without saving trades to the database")
    args = parser.parse_args()

    app = App(args.broker, use_db=not args.no_db)
    app.run(args.duration)