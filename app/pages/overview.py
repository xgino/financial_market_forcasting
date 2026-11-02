# financial_market_forcasting1/app/pages/overview.py
"""Overview: everything at a glance. The signal per asset, how fresh the data is, the best strategy, which models
have proven anything, and the judge's verdict. Details are on the other pages."""
import re

import pandas as pd
import streamlit as st

import settings
from app.common import (COST, DATA, HORIZON_LABEL, all_verdicts, freshness, horizons_ready, load, need, past_calls,
                        pick_strategies, proven, signal, suffix)

st.title("Overview")
hs = horizons_ready()
need(hs or None, 4)
k = load("simulation", "kpis.csv")
fr = freshness()

row = st.columns([2, 2, 3], vertical_alignment="bottom")
amount = row[0].number_input("your amount (EUR)", min_value=10.0, value=float(settings.NOTIFY.get("amount", 1000)), step=50.0)
h = row[1].segmented_control("looking ahead", hs, default=settings.HORIZON_DAYS if settings.HORIZON_DAYS in hs else hs[0],
                             format_func=lambda x: HORIZON_LABEL.get(x, f"{x} days"), required=True)
vs = all_verdicts(h)
day = max(d for d, *_ in vs.values())
last_price = fr["last price"].max() if len(fr) else None
stale = fr[fr["ok"] != "✅"] if len(fr) else fr
row[2].caption(f"Forecasts made {day:%d %b %Y} with only what was known that day. "
               + (f"Prices up to {last_price:%d %b %Y}" if last_price else "")
               + (" ✅" if stale.empty else f", ⚠️ {len(stale)} asset(s) stale: press Update now in the sidebar") + ".")
if h == settings.HORIZON_DAYS and day.to_period("M") < pd.Timestamp.today().to_period("M"):
    st.warning(f"The last monthly decision is from {day:%B %Y}. Press **Update now** in the sidebar.", icon=":material/update:")

# ----- the signal per asset -----
st.subheader(f"Signal per asset, next {HORIZON_LABEL.get(h, f'{h} days')}")
st.caption("How to read a card: the **best guess** is what the models expect (weighted by their track record), the **normal "
           "swing** is how much this asset moves anyway in that time. A guess much smaller than the swing is noise, not a "
           "signal, however the bad and good case look. BUY or WAIT only shows when models with a proven record agree.")
cards = st.columns(4)
for i, (a, (d, situation, v, mt)) in enumerate(vs.items()):
    s = signal(v, amount, h)
    px = load("prices", f"{a}.csv", usecols=["date", "close_eur"])
    spark = px["close_eur"].tail(120).round(2).tolist() if px is not None else None
    with cards[i % 4].container(border=True):
        top = st.container(horizontal=True, vertical_alignment="center")
        top.markdown(f"**{a}**")
        top.badge(s["label"], color=s["color"])
        st.metric("best guess", "-" if pd.isna(s["guess"]) else f"{s['guess']:+.1%}",
                  None if pd.isna(s["guess"]) else f"EUR {amount * s['after_costs']:+,.0f} after costs",
                  chart_data=spark, chart_type="area", delta_color="normal" if s["color"] != "gray" else "off",
                  help="Chart: the last 120 trading days. Delta: the best guess in euros on your amount, minus buy and sell costs.")
        if pd.notna(s["swing"]):
            st.caption(f"normal swing ±{s['swing']:.1%} · guess is {abs(s['ratio']):.2f}× the swing · "
                       f"trusted: {', '.join(mt.index[mt['trusted']]) or 'none'}")
st.caption(f"Costs to buy and later sell: {COST:.2%} of the amount (EUR {amount * COST:,.2f} on EUR {amount:,.0f}). "
           "Click **Forecasts** to see every past guess against what really happened.")

# ----- money: the strategies -----
st.subheader("Your money: the best strategy so far")
if k is None:
    st.info("Run step 6 to compare strategies.")
else:
    best = pick_strategies(k).iloc[0]
    hold = k.set_index("strategy").loc["buy_and_hold"] if "buy_and_hold" in set(k["strategy"]) else None
    P = settings.PASS
    with st.container(horizontal=True):
        st.metric(f"{best['strategy'].replace('_', ' ')}: per year", f"{best['per_year']:+.1%}",
                  None if hold is None else f"{best['per_year'] - hold['per_year']:+.1%} vs buy & hold", border=True)
        st.metric("worst drop", f"{best['max_drop']:.0%}",
                  None if hold is None else f"{best['max_drop'] - hold['max_drop']:+.0%} vs buy & hold", border=True)
        st.metric("sharpe (return per swing)", f"{best['sharpe']:.2f}",
                  None if hold is None else f"{best['sharpe'] - hold['sharpe']:+.2f} vs buy & hold", border=True)
        st.metric(f"EUR {settings.START_EUR:,.0f} became", f"EUR {best['end_eur']:,.0f}",
                  None if hold is None else f"EUR {best['end_eur'] - hold['end_eur']:+,.0f} vs buy & hold", border=True)
    uses = "uses no forecasts, only a fixed plan" if best["follows"] == "base_plan" else f"follows {best['follows'].replace('_', ' ')}"
    st.caption(f"Development years only (holdout hidden). It {uses}. Pass marks: Sharpe above buy & hold, worst drop above "
               f"{P['max_drop']:.0%}. All strategies are on the Results page.")

# ----- models: has anything proven itself? -----
st.subheader("Models: has anything proven itself?")
cols = st.columns(len(hs))
for c, hh in zip(cols, hs):
    board = load("battle", f"leaderboard{suffix(hh)}.csv")
    good = proven(board)
    with c.container(border=True):
        st.metric(HORIZON_LABEL.get(hh, f"{hh} days"), f"{len(good)} of {0 if board is None else len(board)}",
                  help="Models with an edge on the development years: trust ≥ 2 and the luck test passed.")
        st.caption("Proven: " + (", ".join(good) if good else "none yet") + ". Confirm in the holdout before trusting.")
calls = past_calls()
if len(calls):
    with st.expander("How did this monthly rule do in the past? (every asset, only what was known then)"):
        st.dataframe(calls.style.format({"avg_move_after": "{:+.2%}", "went_up": "{:.0%}", "avg_confidence": "{:.0f}"}),
                     width="stretch")
        st.caption("A call is only useful if what came after it differs from KEEP YOUR PLAN (which is 'any month').")

# ----- the judge -----
report = DATA / "judge" / "report.md"
if report.exists():
    m = re.search(r"\*\*Verdict:\*\*\s*(.+)", report.read_text())
    if m:
        st.info(m.group(1), title="Judge", icon=":material/gavel:")
