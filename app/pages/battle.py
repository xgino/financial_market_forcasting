# financial_market_forcasting1/app/pages/battle.py
"""Battle: month by month replay. Every model's guess as a line, what really happened, the winner."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import settings
from app.common import COLORS, WIN, load, need


st.title("Battle: month by month")
j, board, wins = load("battle", "journal.csv"), load("battle", "leaderboard.csv"), load("battle", "winners.csv")
need(j, 4)
c1, c2, c3 = st.columns([1, 2, 1])
asset = c1.selectbox("asset", sorted(j["asset"].unique()))
ranked = board["model"].tolist() if board is not None else sorted(j["model"].unique())
top_n = c3.slider("top N models", 1, len(ranked), min(5, len(ranked)))
chosen = c2.multiselect("models", ranked, default=ranked[:top_n])

ja = j[(j["asset"] == asset) & j["model"].isin(chosen)]
days = pd.DatetimeIndex(sorted(ja["date"].unique()))
firsts = pd.Series(days, index=days).groupby([days.year, days.month]).min()
if not settings.SHOW_HOLDOUT:
    firsts = firsts[firsts < pd.Timestamp(settings.HOLDOUT_FROM)]
months = list(firsts)
if not months:
    st.warning("no months to show")
    st.stop()
key = f"m_{asset}"
st.session_state.setdefault(key, len(months) // 2)
b1, b2, b3, b4 = st.columns([1, 1, 4, 2])
if b1.button("◀ previous"):
    st.session_state[key] = max(0, st.session_state[key] - 1)
if b2.button("next ▶"):
    st.session_state[key] = min(len(months) - 1, st.session_state[key] + 1)
st.session_state[key] = b3.select_slider("month", options=list(range(len(months))), value=st.session_state[key],
                                         format_func=lambda i: months[i].strftime("%Y-%m"))
reveal = b4.toggle("show what really happened", value=True)
d = months[st.session_state[key]]

prices = load("prices", f"{asset}.csv")
px = prices.set_index("date")["close_eur"]
today = ja[ja["date"] == d].set_index("model").reindex(chosen).dropna(subset=["p50"])
if today.empty:
    st.warning("no forecasts that day")
    st.stop()
end = pd.Timestamp(today["end_date"].dropna().iloc[0]) if today["end_date"].notna().any() else d
p0 = float(today["price"].iloc[0])
real_end = p0 * (1 + float(today["actual"].iloc[0])) if pd.notna(today["actual"].iloc[0]) else np.nan
today["guess_price"] = p0 * (1 + today["p50"])
today["miss"] = (today["guess_price"] - real_end).abs()
winner = today["miss"].idxmin() if reveal and np.isfinite(real_end) else None

fig = go.Figure()
before = px[(px.index <= d) & (px.index >= d - pd.DateOffset(days=180))]
fig.add_trace(go.Scatter(x=before.index, y=before, name="real price", line=dict(color="#444", width=1.6)))
if reveal:
    after = px[(px.index >= d) & (px.index <= end)]
    fig.add_trace(go.Scatter(x=after.index, y=after, name="what really happened",
                             line=dict(color="#444", width=1.6, dash="dot")))
for i, (m, r) in enumerate(today.iterrows()):
    col = WIN if m == winner else COLORS[i % len(COLORS)]
    width = 4 if m == winner else 2
    fig.add_trace(go.Scatter(x=[d, end], y=[p0, r["guess_price"]], mode="lines+markers", name=m,
                             line=dict(color=col, width=width),
                             hovertext=f"{m}: {r['p50']:+.1%} -> EUR {r['guess_price']:,.2f}", hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[end, end], y=[p0 * (1 + r["p10"]), p0 * (1 + r["p90"])], mode="lines",
                             showlegend=False, line=dict(color=col, width=8), opacity=0.15, hoverinfo="skip"))
title = f"{asset} {d.date()}: every model's guess for {end.date()}"
if winner:
    title += f"  |  winner: {winner}"
fig.update_layout(title=title, height=520, template="plotly_white", legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, width="stretch")

show = pd.DataFrame({"guess": today["p50"].map(lambda x: f"{x:+.1%}"),
                     "guess price": today["guess_price"].map(lambda x: f"EUR {x:,.2f}"),
                     "bad .. good": [f"{a:+.1%} .. {b:+.1%}" for a, b in zip(today["p10"], today["p90"])],
                     "situation": today["situation"]})
if reveal and np.isfinite(real_end):
    show["real"] = f"{today['actual'].iloc[0]:+.1%}"
    show["missed by"] = today["miss"].map(lambda x: f"EUR {x:,.2f}")
    show["winner"] = ["🏆" if m == winner else "" for m in today.index]
st.dataframe(show, width="stretch")
st.caption("A month's winner is mostly luck: one result per month. The leaderboard is the real ranking.")
