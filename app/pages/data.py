# financial_market_forcasting1/app/pages/data.py
"""Data: is it up to date (prices from Yahoo, macro from FRED), and every asset's price history."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import settings
from app.common import DATA, freshness, load

st.title("Data")
st.subheader("Is the data up to date?")
fr = freshness()
if len(fr):
    st.dataframe(fr, width="stretch", hide_index=True, column_config={"source": st.column_config.LinkColumn("source")})
rows = []
for name, sid in settings.MACRO_SERIES.items():
    m = load("macro", f"{name}.csv", usecols=["date"])
    last = m["date"].max() if m is not None else None
    rows.append({"macro": name, "last value": last.date() if last is not None else "missing",
                 "days old": (pd.Timestamp.today() - last).days if last is not None else None,
                 "source": f"https://fred.stlouisfed.org/series/{sid}"})
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={"source": st.column_config.LinkColumn("source")})
st.caption("Prices: Yahoo Finance, daily closes, converted to EUR. Macro: FRED (or Yahoo as back-up), each value only used "
           "from the next trading day. Press Update now in the sidebar to refresh.")

files = sorted(p.stem for p in (DATA / "prices").glob("*.csv")) if (DATA / "prices").exists() else []
if not files:
    st.info("No prices yet. Run step 1.")
    st.stop()
asset = st.selectbox("asset", files)
df = load("prices", f"{asset}.csv")
fig = go.Figure([go.Scatter(x=df["date"], y=df["close_eur"], name=asset, line=dict(width=1.2))])
fig.add_vline(x=pd.Timestamp(settings.HOLDOUT_FROM), line_dash="dash", line_color="red")
fig.update_layout(height=460, template="plotly_white", yaxis_title="EUR", title=f"{asset} (red line: holdout starts)")
st.plotly_chart(fig, width="stretch", alt=f"{asset} price history in EUR")
st.dataframe(df.tail(20).iloc[::-1], width="stretch", hide_index=True)
