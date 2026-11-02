# financial_market_forcasting1/app/pages/calculator.py
"""Calculator: try amounts, models and your own numbers. It only reads."""
import numpy as np
import pandas as pd
import streamlit as st

import settings
from app.common import load
from core.verdict import chance_up, firsts, models_on


st.title("Calculator")
st.caption("Try amounts and models. It only reads, it never changes anything.")
j, book, ch = load("battle", "journal.csv"), load("battle", "skillbook.csv"), load("selection", "choices.csv")
fee = settings.FEE_RATE
st.subheader("With the models, this month")
if j is None or ch is None:
    st.info("Run steps 4 and 5 first.")
else:
    c = st.columns(3)
    amount = c[0].number_input("amount (EUR)", min_value=10.0, value=1000.0, step=100.0)
    assets = sorted(j["asset"].unique())
    a = c[1].selectbox("asset", assets, index=assets.index("gold") if "gold" in assets else 0)
    how = c[2].radio("combine", ["each on its own", "average", "weighted by trust"], horizontal=False)
    d = firsts(j, a)[-1]
    mt, situation = models_on(j, book, a, d)
    pick = st.multiselect("models (default: top 3 by trust in today's situation)", mt.index.tolist(), default=mt.index[:3].tolist())
    rows = mt.loc[pick, ["p10", "p50", "p90", "trust"]] if pick else mt.iloc[0:0]
    if len(rows) and how != "each on its own":
        w = np.ones(len(rows)) if how == "average" else rows["trust"].clip(lower=0).fillna(0).to_numpy()
        w = w / w.sum() if w.sum() > 0 else np.ones(len(rows)) / len(rows)
        rows = pd.DataFrame([{c_: float((rows[c_] * w).sum()) for c_ in ["p10", "p50", "p90"]}], index=[how])
    out = []
    for mm, r in rows.iterrows():
        net = amount * (1 + r["p50"]) * (1 - fee) * (1 - fee)
        out.append({"model": mm, "expects": f"{r['p50']:+.1%}", "middle": f"EUR {amount * (1 + r['p50']):,.2f}",
                    "bad case": f"EUR {amount * (1 + r['p10']):,.2f}", "good case": f"EUR {amount * (1 + r['p90']):,.2f}",
                    "chance up": f"{chance_up(r['p10'], r['p50'], r['p90']):.0%}" if pd.notna(chance_up(r['p10'], r['p50'], r['p90'])) else "",
                    "middle after fees": f"EUR {net:,.2f}"})
    st.dataframe(pd.DataFrame(out), width="stretch", hide_index=True)
    st.caption(f"{a} on {d.date()} ({situation}), 20 trading days ahead, buy and sell fee {fee:.2%} each. "
               "Averaging models usually lowers the error a bit, but it never turns untrusted models into trusted ones.")

st.subheader("Your own numbers (no data used)")
c = st.columns(5)
amt = c[0].number_input("amount", min_value=10.0, value=1000.0, step=100.0, key="own_amt")
mid = c[1].number_input("expected move %", value=1.0, step=0.5) / 100
bad = c[2].number_input("bad case %", value=-6.0, step=0.5) / 100
good = c[3].number_input("good case %", value=7.0, step=0.5) / 100
f = c[4].number_input("fee per trade %", value=fee * 100, step=0.05) / 100
res = st.columns(4)
res[0].metric("middle after fees", f"EUR {amt * (1 + mid) * (1 - f) ** 2:,.2f}", f"{(1 + mid) * (1 - f) ** 2 - 1:+.2%}")
res[1].metric("bad case after fees", f"EUR {amt * (1 + bad) * (1 - f) ** 2:,.2f}")
res[2].metric("good case after fees", f"EUR {amt * (1 + good) * (1 - f) ** 2:,.2f}")
res[3].metric("move needed to break even", f"{1 / (1 - f) ** 2 - 1:+.2%}")
cu = chance_up(bad, mid, good)
if pd.notna(cu):
    st.caption(f"Rough chance of ending higher: {cu:.0%}. Expected value after fees: "
               f"EUR {amt * (1 + mid) * (1 - f) ** 2 - amt:+,.2f} on EUR {amt:,.0f}.")
