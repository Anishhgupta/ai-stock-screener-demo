"""
Live dashboard. Reads the live snapshot that main.py saves to the
database about once a second (falling back to dashboard/state.json for
runs started with --no-db), plus saved trade history from the database.

Run alongside main.py:
    streamlit run dashboard/app.py
"""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR  # noqa: E402
from storage.dashboard_data import (  # noqa: E402
    all_time_summary, closed_trades_frame, decision_log_frame,
    load_live_state, watchlist_frame,
)
from storage.db import make_engine  # noqa: E402
from storage.repository import Repository  # noqa: E402

STATE_PATH = BASE_DIR / "dashboard" / "state.json"
REFRESH_SECONDS = 2

st.set_page_config(page_title="Stock Screener - Live", layout="wide")
st.title("Real-Time SMMA + LTQ + Bid/Ask AI Screener")

placeholder = st.empty()


@st.cache_resource
def get_repo():
    """Returns (repository, user_id), or (None, None) if the database is unavailable."""
    try:
        repo = Repository(make_engine())
        return repo, repo.ensure_user("local")
    except Exception:
        return None, None


def render(state, repo, user_id):
    with placeholder.container():
        if state is None:
            st.warning("Waiting for main.py to start writing state... run `python main.py --broker mock`")
            return

        age = time.time() - state["updated_at"]
        st.caption(f"Last update: {age:.1f}s ago")
        if repo is None:
            st.caption("Database unavailable: showing live state only, no saved history.")

        all_time = all_time_summary(repo, user_id)
        col1, col2 = st.columns(2)
        for col, key, book, label in (
            (col1, "basic_summary", "basic", "Basic Strategy"),
            (col2, "filtered_summary", "filtered", "AI/ML Filtered Strategy"),
        ):
            s = state.get(key)
            if s:
                with col:
                    st.subheader(label)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Trades", s["total_trades"])
                    m2.metric("Win rate", f"{s['win_rate']:.0%}")
                    m3.metric("Open", s["open_trades"])
                    m4.metric("P/L (Rs)", f"{s['total_pnl']:.0f}")
                    saved = all_time.get(book)
                    if saved:
                        st.caption(
                            f"All-time (saved): {saved['total_trades']} trades, "
                            f"{saved['win_rate']:.0%} win rate, P/L Rs {saved['total_pnl']:.0f}"
                        )

        st.subheader("Watchlist — Live Screen")
        df = watchlist_frame(state)
        if len(df):
            def highlight(row):
                if row["Decision"] == "ACCEPT":
                    return ["background-color: #d4f7d4"] * len(row)
                if row["Decision"] == "AVOID":
                    return ["background-color: #f7d4d4"] * len(row)
                return [""] * len(row)

            st.dataframe(df.style.apply(highlight, axis=1), width="stretch", height=400)
        else:
            st.info("No ticks received yet.")

        st.subheader("Open Paper Positions")
        positions = state.get("open_positions", [])
        if positions:
            import pandas as pd
            st.dataframe(pd.DataFrame(positions), width="stretch")
        else:
            st.caption("No open positions.")

        st.subheader("Closed Paper Trades (latest 50)")
        closed = closed_trades_frame(repo, user_id)
        if len(closed):
            st.dataframe(closed, width="stretch", height=300)
        else:
            st.caption("No closed trades saved yet.")

        st.subheader("Decision Log (latest 50)")
        log = decision_log_frame(repo, user_id)
        if len(log):
            st.dataframe(log, width="stretch", height=300)
        else:
            st.caption("No decisions saved yet.")


repo, user_id = get_repo()
render(load_live_state(repo, user_id, STATE_PATH), repo, user_id)

st.caption(f"Auto-refreshing every {REFRESH_SECONDS}s")
time.sleep(REFRESH_SECONDS)
st.rerun()