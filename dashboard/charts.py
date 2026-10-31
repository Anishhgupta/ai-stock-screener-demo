"""
Plotly charts for the live dashboard. Text colours are left to the Streamlit
theme, so the charts read the same in light and dark mode.
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

BASIC_COLOR = "#ff9800"
FILTERED_COLOR = "#00e676"
GOOD, BAD, AMBER, GREY = "#00e676", "#ff1744", "#ffb300", "#555555"
# Below this many trades the win-rate gauge stays amber: colouring it green or
# red would suggest a verdict that a handful of trades cannot support.
MIN_TRADES_FOR_COLOUR = 30


def win_rate_gauge(win_rate: float, n_trades: int, title: str = "Win rate") -> go.Figure:
    if n_trades >= MIN_TRADES_FOR_COLOUR:
        colour = GOOD if win_rate >= 0.5 else BAD
    else:
        colour = AMBER
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_rate * 100,
        number={"suffix": "%", "valueformat": ".0f"},
        title={"text": f"{title} ({n_trades} trades)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 50, 100]},
            "bar": {"color": colour},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": "#333333",
            "steps": [
                {"range": [0, 50], "color": "rgba(255, 23, 68, 0.15)"},
                {"range": [50, 100], "color": "rgba(0, 230, 118, 0.15)"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=45, r=45, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def verdict_donut(counts: dict) -> go.Figure:
    accept, avoid = counts.get("ACCEPT", 0), counts.get("AVOID", 0)
    if accept + avoid == 0:
        labels, values, colours = ["No verdicts yet"], [1], [GREY]
    else:
        labels, values, colours = ["ACCEPT", "AVOID"], [accept, avoid], [GOOD, BAD]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.6,
                           marker=dict(colors=colours), sort=False))
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    fig.add_annotation(text=f"{accept} ACCEPT<br>{avoid} AVOID", x=0.5, y=0.5,
                       font_size=14, showarrow=False)
    return fig


def pnl_chart(curve):
    """Realised P/L through the day, or None when nothing has closed yet."""
    if curve is None or curve.empty:
        return None
    fig = px.line(curve, x="Time", y=["Basic", "Filtered"], line_shape="hv",
                  color_discrete_sequence=[BASIC_COLOR, FILTERED_COLOR])
    fig.update_layout(
        title="Realised profit and loss today (closed trades)",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="#333333", title="P/L (Rs)"),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig