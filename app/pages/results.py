# financial_market_forcasting1/app/pages/results.py
"""Results: EUR 1000 per strategy, month by month, with the pass marks."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import settings
from app.common import COLORS, WIN, load, need, ok


st.title("Results: EUR 1000 per strategy, month by month")
k = load("simulation", "kpis.csv")
need(k, 6)
st.caption("Development period" + (" + HOLDOUT" if settings.SHOW_HOLDOUT else
           f" ({int(settings.WARMUP_UNTIL[:4]) + 1} to {int(settings.HOLDOUT_FROM[:4]) - 1}), holdout hidden")
           + f". Pass marks: Sharpe above buy & hold, worst drop above {settings.PASS['max_drop']:.0%}, Sharpe at least {settings.PASS['sharpe']}.")
t = k[["strategy", "follows", "end_eur", "per_year", "max_drop", "sharpe", "months_up", "fees", "avg_invested"]].copy()
for col in ["beats hold", "drop ok", "sharpe ok"]:
    t[col] = k[col].map(ok) if col in k else ""
st.dataframe(t.style.format({"end_eur": "EUR {:,.0f}", "per_year": "{:+.1%}", "max_drop": "{:.0%}", "sharpe": "{:.2f}",
                             "months_up": "{:.0%}", "fees": "EUR {:,.0f}", "avg_invested": "{:.0%}"}),
             width="stretch", hide_index=True)
fig = go.Figure()
for i, s in enumerate(k["strategy"]):
    m = load("simulation", f"{s}_months.csv")
    if m is not None and len(m):
        fig.add_trace(go.Scatter(x=m["date"], y=m["equity_end"], name=s, line=dict(width=3 if s == "buy_and_hold" else 1.6,
                                 dash="dot" if s == "buy_and_hold" else None, color=COLORS[i % len(COLORS)])))
fig.add_hline(y=settings.START_EUR, line_color="#888", line_width=1)
fig.update_layout(height=430, template="plotly_white", yaxis_title="fund value (EUR)", title="Fund value per strategy")
st.plotly_chart(fig, width="stretch")
strat = st.selectbox("look closer at", k["strategy"].tolist())
m = load("simulation", f"{strat}_months.csv")
if m is not None and len(m):
    r = m["month_return"] * 100
    fig = go.Figure(go.Bar(x=m["date"], y=r, marker_color=np.where(r >= 0, WIN, "#d64545"),
                           hovertext=[f"{d:%Y-%m}: {v:+.1f}%" for d, v in zip(m["date"], r)], hoverinfo="text"))
    fig.update_layout(height=300, template="plotly_white", yaxis_title="month %", title=f"{strat}: every month, green up, red down")
    st.plotly_chart(fig, width="stretch")
    eq = m["equity_end"]
    dd = (eq / eq.cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=m["date"], y=dd, fill="tozeroy", line=dict(color="#d64545")))
    fig.add_hline(y=settings.PASS["max_drop"] * 100, line_dash="dash", annotation_text="pass mark")
    fig.update_layout(height=260, template="plotly_white", yaxis_title="% below its top", title="How deep it fell")
    st.plotly_chart(fig, width="stretch")
tr = load("simulation", f"{strat}_trades.csv")
if tr is not None and len(tr):
    st.subheader("Trades")
    st.dataframe(tr.iloc[::-1].head(60), width="stretch", hide_index=True)
