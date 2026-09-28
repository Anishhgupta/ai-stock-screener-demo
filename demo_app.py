"""
Standalone, self-contained DEMO of the SMMA + LTQ + Bid/Ask AI Screener.

Unlike the full project (main.py + a separate dashboard/app.py reading a
shared state file), this single file runs EVERYTHING in-process.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Calibrated against real live NSE depth data
os.environ.setdefault("SCREENER_MIN_DEPTH_QTY", "2000")

from brokers.mock_client import MockClient
from config import CONFIG
from demo.synthetic_history import generate_all_symbol_histories
from features.engineering import build_feature_snapshot
from ml.predict import CrossoverModel, Decision
from signals.crossover import CrossoverEngine
from trading.paper_engine import Book, PaperTradingEngine

st.set_page_config(page_title="SmartAlgo AI Screener", layout="wide")


# ---------------------------------------------------------------------
# In-memory, thread-safe shared state.
# ---------------------------------------------------------------------
@dataclass
class DemoState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: bool = False
    symbols: dict = field(default_factory=dict)
    basic_summary: dict = field(default_factory=dict)
    filtered_summary: dict = field(default_factory=dict)
    open_positions: list = field(default_factory=list)
    updated_at: float = 0.0
    
    # New tracking arrays for interactive charts
    pnl_history: list = field(default_factory=list)
    decision_counts: dict = field(default_factory=lambda: {"ACCEPT": 0, "AVOID": 0})
    last_chart_update: float = 0.0


@st.cache_resource
def get_demo_state() -> DemoState:
    return DemoState()


def start_background_engine(state: DemoState):
    """Runs once per server process (guarded by state.started)."""
    engine = CrossoverEngine(persist_ticks=False)
    from demo.train_demo_model import DEMO_MODEL_PATH
    model = CrossoverModel(model_path=DEMO_MODEL_PATH)
    
    trader = PaperTradingEngine(model)
    broker = MockClient(CONFIG.watchlist, tick_interval=0.05, speed=1.0, sim_seconds_per_loop=5.0)

    histories = generate_all_symbol_histories(CONFIG.watchlist, num_candles=150)
    ending_prices = {}
    for symbol, (candles, ending_price) in histories.items():
        engine.warm_start(symbol, candles)
        ending_prices[symbol] = ending_price
    broker.seed_prices(ending_prices)

    def handle_tick(tick):
        event = engine.on_tick(tick)
        live_feat = build_feature_snapshot(
            symbol=tick.symbol,
            recent_ticks=engine.tick_store.recent(tick.symbol, 50),
            recent_candles=engine.candles.closed_candles(tick.symbol, 30),
            smma_pair=engine.smma.get(tick.symbol),
        )

        decision = None
        if event is not None:
            if model.is_available:
                decision = model.decide(event)
            else:
                decision = Decision(
                    symbol=event.symbol, signal=event.signal, probability=None,
                    decision="AVOID", reasons=["No trained model available"],
                )
            trader.on_crossover(event, decision, tick.ltp)
            
            # Log AI Decisions for the Donut Chart
            with state.lock:
                if decision.decision in state.decision_counts:
                    state.decision_counts[decision.decision] += 1

        trader.on_tick(tick, live_feat)

        with state.lock:
            existing = state.symbols.get(tick.symbol, {})
            state.symbols[tick.symbol] = {
                "symbol": tick.symbol,
                "ltp": tick.ltp,
                "smma_fast": engine.smma.get(tick.symbol).fast.value,
                "smma_slow": engine.smma.get(tick.symbol).slow.value,
                "ltq": tick.ltq,
                "etq": tick.total_traded_qty,
                "bid_price": tick.bid_price, "bid_qty": tick.bid_qty,
                "ask_price": tick.ask_price, "ask_qty": tick.ask_qty,
                "imbalance": live_feat.bid_ask_imbalance if live_feat else None,
                "signal": event.signal if event else existing.get("signal"),
                "probability": decision.probability if decision else existing.get("probability"),
                "decision": decision.decision if decision else existing.get("decision"),
                "reasons": ", ".join(decision.reasons) if decision else existing.get("reasons", ""),
            }
            state.basic_summary = trader.summary(Book.BASIC)
            state.filtered_summary = trader.summary(Book.FILTERED)
            state.open_positions = [
                {"book": p.book.value, "symbol": p.symbol, "signal": p.signal,
                 "entry_price": p.entry_price, "qty": p.qty, "monitor_flag": p.monitor_flag}
                for p in trader.open_positions.values()
            ]
            state.updated_at = time.time()
            
            # Sample P/L data for the Line Chart roughly every second
            if state.updated_at - state.last_chart_update > 1.0:
                state.pnl_history.append({
                    "Time": state.updated_at,
                    "Basic Strategy": state.basic_summary.get("total_pnl", 0),
                    "AI Filtered": state.filtered_summary.get("total_pnl", 0)
                })
                if len(state.pnl_history) > 60:  # Keep the last 60 data points visible
                    state.pnl_history.pop(0)
                state.last_chart_update = state.updated_at

    broker.connect()
    broker.subscribe(handle_tick)


state = get_demo_state()
with state.lock:
    already_started = state.started
    state.started = True
if not already_started:
    start_background_engine(state)


# ---------------------------------------------------------------------
# UI Helpers for Plotly
# ---------------------------------------------------------------------
def make_gauge_chart(win_rate, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_rate * 100,
        number={'suffix': "%", 'font': {'color': 'white'}},
        title={'text': title, 'font': {'size': 16, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00e676" if win_rate >= 0.5 else "#ff1744"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 50], 'color': "rgba(255, 23, 68, 0.2)"},
                {'range': [50, 100], 'color': "rgba(0, 230, 118, 0.2)"}
            ]
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def make_donut_chart(counts):
    labels = list(counts.keys())
    values = list(counts.values())
    
    if sum(values) == 0:
        labels, values, colors = ["No Trades"], [1], ["#333333"]
    else:
        colors = ["#00e676" if l == "ACCEPT" else "#ff1744" for l in labels]
        
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker=dict(colors=colors))])
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=10), 
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False
    )
    fig.add_annotation(text="AI<br>Decisions", x=0.5, y=0.5, font_size=16, font_color="white", showarrow=False)
    return fig


# ---------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------
st.title("📈 SmartAlgo AI Screener")

with state.lock:
    symbols_snapshot = dict(state.symbols)
    basic = dict(state.basic_summary)
    filtered = dict(state.filtered_summary)
    positions = list(state.open_positions)
    pnl_history = list(state.pnl_history)
    decision_counts = dict(state.decision_counts)
    updated_at = state.updated_at

age = time.time() - updated_at if updated_at else None
st.caption(f"Live Market Data | Last tick: {age:.1f}s ago" if age is not None else "Starting up...")

# Top Row: KPIs and Gauges
col1, col2, col3 = st.columns([1.5, 1.5, 1])

with col1:
    with st.container(border=True):
        st.subheader("Basic Strategy")
        c1, c2 = st.columns(2)
        c1.metric("Total Trades", basic.get("total_trades", 0))
        c1.metric("P/L (Rs)", f"₹{basic.get('total_pnl', 0):.0f}")
        c2.plotly_chart(make_gauge_chart(basic.get("win_rate", 0), "Win Rate"), use_container_width=True, key="g1")

with col2:
    with st.container(border=True):
        st.subheader("🤖 AI Filtered Strategy")
        c1, c2 = st.columns(2)
        c1.metric("Total Trades", filtered.get("total_trades", 0))
        c1.metric("P/L (Rs)", f"₹{filtered.get('total_pnl', 0):.0f}")
        c2.plotly_chart(make_gauge_chart(filtered.get("win_rate", 0), "Win Rate"), use_container_width=True, key="g2")

with col3:
    with st.container(border=True):
        st.subheader("AI Decision Ratio")
        st.plotly_chart(make_donut_chart(decision_counts), use_container_width=True, key="d1")

# Middle Row: Live P/L Curve
if pnl_history:
    df_pnl = pd.DataFrame(pnl_history)
    df_pnl['Time'] = pd.to_datetime(df_pnl['Time'], unit='s')
    
    fig_line = px.line(df_pnl, x='Time', y=['Basic Strategy', 'AI Filtered'], 
                       color_discrete_sequence=["#ff9800", "#00e676"])
    
    fig_line.update_layout(
        title="Live Profit & Loss Over Time",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="#333", title="P/L (Rupees)"),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line, use_container_width=True, key="line1")

st.divider()

# Watchlist Dataframe
st.subheader("Watchlist — Live Screen")
if symbols_snapshot:
    df = pd.DataFrame(symbols_snapshot.values())
    display_cols = ["symbol", "ltp", "smma_fast", "smma_slow", "signal", "ltq", "etq",
                     "bid_price", "bid_qty", "ask_price", "ask_qty", "imbalance",
                     "probability", "decision", "reasons"]
    df = df[[c for c in display_cols if c in df.columns]]
    df.columns = ["Symbol", "LTP", "SMMA20", "SMMA120", "Signal", "LTQ", "ETQ",
                  "Bid Px", "Bid Qty", "Ask Px", "Ask Qty", "Bid/Ask Imbalance",
                  "AI Probability", "Decision", "Reasons"][:len(df.columns)]

    def highlight(row):
        if row.get("Decision") == "ACCEPT":
            return ["background-color: #0f5132; color: #75b798; font-weight: bold;"] * len(row)
        if row.get("Decision") == "AVOID":
            return ["background-color: #842029; color: #ea868f; font-weight: bold;"] * len(row)
        return [""] * len(row)

    st.dataframe(df.style.apply(highlight, axis=1), width="stretch", height=400)
else:
    st.info("Waiting for the first simulated ticks...")

st.subheader("Open Paper Positions")
if positions:
    st.dataframe(pd.DataFrame(positions), width="stretch")
else:
    st.caption("No open positions yet.")

from demo.train_demo_model import DEMO_MODEL_PATH
if not CrossoverModel(model_path=DEMO_MODEL_PATH).is_available:
    st.warning("No demo model file found. Run `python -m demo.train_demo_model` to generate it.")

time.sleep(2)
st.rerun()