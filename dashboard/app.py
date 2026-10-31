"""
Live dashboard. Reads the live snapshot that main.py saves to the
database about once a second (falling back to dashboard/state.json for
runs started with --no-db), plus saved trade history from the database.

Run alongside main.py:
    streamlit run dashboard/app.py
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR  # noqa: E402
from dashboard.charts import (  # noqa: E402
    MIN_TRADES_FOR_COLOUR, pnl_chart, verdict_donut, win_rate_gauge,
)
from storage.dashboard_data import (  # noqa: E402
    CLOSED_TRADE_FORMATS, DECISION_LOG_FORMATS, EXPORT_ROW_LIMIT, IST,
    POSITION_FORMATS, WATCHLIST_FORMATS, all_time_summary, closed_trades_frame,
    decision_log_frame, filter_watchlist, format_rupees, load_live_state,
    pnl_curve, positions_frame, ranked_trades, recent_rows, sort_watchlist,
    status_line, style_table, summarise_today, tile_numbers, to_csv_bytes,
    verdict_counts, watchlist_frame,
)
from storage.db import make_engine  # noqa: E402
from storage.repository import Repository  # noqa: E402

STATE_PATH = BASE_DIR / "dashboard" / "state.json"
LIVE_REFRESH_SECONDS = 2
HISTORY_REFRESH_SECONDS = 10

st.set_page_config(page_title="SmartAlgo AI Screener", page_icon="📈", layout="wide")


@st.cache_resource
def get_repo():
    """Returns (repository, user_id), or (None, None) if the database is unavailable."""
    try:
        repo = Repository(make_engine())
        return repo, repo.ensure_user("local")
    except Exception:
        return None, None


repo, user_id = get_repo()

st.title("📈 SmartAlgo AI Screener")
st.caption("Live paper trading. No real orders are placed, and nothing here is investment advice.")


def load_table(loader, *args, **kwargs):
    """Run a table loader. Returns None if the database is briefly unavailable."""
    try:
        return loader(*args, **kwargs)
    except Exception:
        return None


def render_book_card(column, label, book, summary, saved):
    with column, st.container(border=True):
        st.subheader(label)
        left, right = st.columns(2)
        left.metric("Trades today", summary["total_trades"])
        left.metric("P/L today", format_rupees(summary["total_pnl"]))
        left.metric("Open now", summary["open_trades"])
        right.plotly_chart(win_rate_gauge(summary["win_rate"], summary["total_trades"]),
                           key=f"gauge-{book}")
        if summary["total_trades"] < MIN_TRADES_FOR_COLOUR:
            n = summary["total_trades"]
            st.caption(f"Only {n} trade{'' if n == 1 else 's'} so far: too few to read anything into.")
        if saved:
            st.caption(f"All time (saved): {saved['total_trades']} trades, "
                       f"{saved['win_rate']:.0%} win rate, P/L {format_rupees(saved['total_pnl'])}")


@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def live_panel():
    state = load_live_state(repo, user_id, STATE_PATH)
    if state is None:
        st.warning("Waiting for the screener to start writing data. "
                   "Start it with `python main.py --broker mock`, or with the live profile in Docker.")
        return

    st.caption(status_line(time.time() - state["updated_at"]))
    if repo is None:
        st.caption("Database unavailable: showing live state only, with no saved history.")

    rows = recent_rows(repo, user_id)
    today = summarise_today(rows, state["updated_at"]) if rows is not None else {}
    try:
        all_time = all_time_summary(repo, user_id)
    except Exception:
        all_time = {}
    if today:
        st.caption("The numbers for today count trades closed today (Indian time), read from the saved "
                   "history, so they survive a restart. Open positions are live.")

    basic_col, filtered_col, verdict_col = st.columns([1.5, 1.5, 1])
    for column, key, book, label in (
        (basic_col, "basic_summary", "basic", "Basic Strategy"),
        (filtered_col, "filtered_summary", "filtered", "🤖 AI Filtered Strategy"),
    ):
        summary = tile_numbers(state.get(key), today.get(book))
        if summary:
            render_book_card(column, label, book, summary, all_time.get(book))

    with verdict_col, st.container(border=True):
        st.subheader("Latest verdicts")
        st.plotly_chart(verdict_donut(verdict_counts(state)), key="donut")
        st.caption("The most recent ACCEPT or AVOID for each symbol.")

    figure = pnl_chart(pnl_curve(rows or [], state["updated_at"]))
    if figure is not None:
        st.plotly_chart(figure, key="pnl-curve")
    else:
        st.caption("No trades have closed yet today, so there is no profit and loss curve to draw.")

    st.divider()
    st.subheader("Watchlist: live screen")
    choice = st.radio("Show", ["All", "ACCEPT", "AVOID"], horizontal=True,
                      key="watchlist_filter", label_visibility="collapsed")
    full = sort_watchlist(watchlist_frame(state))
    shown = filter_watchlist(full, choice)
    if full.empty:
        st.info("No ticks received yet.")
    elif shown.empty:
        st.caption(f"No symbol has an {choice} verdict right now.")
    else:
        st.dataframe(style_table(shown, WATCHLIST_FORMATS, highlight_decisions=True),
                     width="stretch", height=400, hide_index=True)

    st.subheader("Open paper positions")
    positions = positions_frame(state)
    if len(positions):
        st.dataframe(style_table(positions, POSITION_FORMATS), width="stretch", hide_index=True)
    else:
        st.caption("No open positions.")


@st.fragment(run_every=HISTORY_REFRESH_SECONDS)
def history_panel():
    st.subheader("Closed paper trades (latest 50)")
    closed = load_table(closed_trades_frame, repo, user_id)
    if closed is None:
        st.caption("Saved history is briefly unavailable. It will try again in a few seconds.")
    elif len(closed):
        st.dataframe(style_table(closed, CLOSED_TRADE_FORMATS), width="stretch",
                     height=300, hide_index=True)
    else:
        st.caption("No closed trades saved yet.")

    st.subheader("Decision log (latest 50)")
    log = load_table(decision_log_frame, repo, user_id)
    if log is None:
        st.caption("Saved history is briefly unavailable. It will try again in a few seconds.")
    elif len(log):
        st.dataframe(style_table(log, DECISION_LOG_FORMATS), width="stretch",
                     height=300, hide_index=True)
    else:
        st.caption("No decisions saved yet.")


def results_and_export():
    st.divider()
    st.subheader("Results and export")
    st.caption("Past paper-trade results from your own saved history. They are not recommendations "
               "or a forecast, and paper trades are not real fills.")
    if repo is None:
        st.caption("Needs the database, which is not available.")
        return

    choice = st.radio("Book", ["Basic", "Filtered", "Both"], horizontal=True, key="results_book")
    closed = load_table(closed_trades_frame, repo, user_id, limit=EXPORT_ROW_LIMIT)
    log = load_table(decision_log_frame, repo, user_id, limit=EXPORT_ROW_LIMIT)
    if closed is None or log is None:
        st.caption("Saved history is briefly unavailable. Reload the page in a few seconds.")
        return
    ranked = ranked_trades(closed, n=5, book=None if choice == "Both" else choice.lower())
    used = ranked["used"]
    st.caption(f"{used} closed trade{'' if used == 1 else 's'} with a real exit. {ranked['marks']} closed "
               "by a session-end or recovery mark left out.")
    best_col, worst_col = st.columns(2)
    for column, title, key in ((best_col, "Best 5", "best"), (worst_col, "Worst 5", "worst")):
        with column:
            st.markdown(f"**{title}**")
            if ranked[key].empty:
                st.caption("No trades yet.")
            else:
                st.dataframe(style_table(ranked[key], CLOSED_TRADE_FORMATS),
                             width="stretch", hide_index=True)
    st.caption("A few trades say little about a strategy. Judge it by all of them, and by how many there are.")

    stamp = datetime.now(IST).strftime("%Y-%m-%d")
    left, right = st.columns(2)
    left.download_button("Download closed trades (CSV)", data=to_csv_bytes(closed),
                         file_name=f"closed_trades_{stamp}.csv", mime="text/csv",
                         disabled=closed.empty)
    right.download_button("Download decision log (CSV)", data=to_csv_bytes(log),
                          file_name=f"decision_log_{stamp}.csv", mime="text/csv",
                          disabled=log.empty)
    st.caption("The files contain everything saved, as of this page load. Reload the page for the latest.")


live_panel()
history_panel()
results_and_export()