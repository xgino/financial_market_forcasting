# financial_market_forcasting1/app/pages/this_month.py
"""This month: the decision for one asset in detail. The call and its signal, why, every vote, the other horizons,
and below that the strategies and every model per asset."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import settings
from app.common import (COLORS, all_verdicts, freshness, horizons_ready, journal, live_models, load,
                        need, past_calls, pick_strategies, signal, skillbook)
from core.verdict import chance_up, firsts, models_on, track_record

CALL_STYLE = {"BUY": st.success, "WAIT": st.error, "KEEP YOUR PLAN": st.warning}


def other_horizons(asset, amount):
    """This week (5 days) and this quarter (60 days): the same vote, on each horizon's own battle."""
    label = {5: "This week (5 trading days): when to buy", 60: "This quarter (60 trading days): the direction"}
    others = [h for h in horizons_ready() if h != settings.HORIZON_DAYS]
    if not others:
        st.caption(f"Other horizons ({', '.join(str(h) for h in settings.HORIZONS if h != settings.HORIZON_DAYS)} days): "
                   "run step 4 to battle them.")
        return
    st.subheader("Other horizons")
    st.caption("The same vote, from each horizon's own battle and track record (step 4, settings.HORIZONS). "
               f"Information only: the monthly decision above and the strategies stay on {settings.HORIZON_DAYS} trading days.")
    for h in others:
        vs = all_verdicts(h)
        if asset not in vs:
            continue
        d, _, va, _ = vs[asset]
        s = signal(va, amount, h)
        with st.container(border=True):
            top = st.container(horizontal=True, vertical_alignment="center")
            top.markdown(f"**{label.get(h, f'{h} trading days')}** · {d:%d %b %Y}")
            top.badge(s["label"], color=s["color"])
            st.markdown(s["sentence"])
            st.caption(va["text"])


def now_details(j, book, ch, k, now, latest, amount):
    # ----- 1. which strategy -----
    ranked = pick_strategies(k)
    best3 = ranked["strategy"].head(3).tolist()
    st.subheader("1. Suggested strategy")
    strat = st.radio("best 3 (by pass marks, then Sharpe). The first one is the suggestion:", best3, horizontal=True)
    others = st.expander("or pick any other strategy")
    other = others.selectbox("strategy", ["(use the one above)"] + ranked["strategy"].tolist())
    if other != "(use the one above)":
        strat = other
    ks = k.set_index("strategy").loc[strat]
    hold = k.set_index("strategy").loc["buy_and_hold"] if "buy_and_hold" in set(k["strategy"]) else None
    P = settings.PASS
    c = st.columns(5)
    c[0].metric("per year", f"{ks['per_year']:+.1%}", None if hold is None else f"hold: {hold['per_year']:+.1%}", delta_color="off")
    c[1].metric("worst drop", f"{ks['max_drop']:.0%}", f"{'✅' if ks['max_drop'] >= P['max_drop'] else '❌'} limit {P['max_drop']:.0%}", delta_color="off")
    c[2].metric("sharpe", f"{ks['sharpe']:.2f}", f"{'✅' if ks['sharpe'] >= P['sharpe'] else '❌'} min {P['sharpe']} | hold {hold['sharpe']:.2f}" if hold is not None else None, delta_color="off")
    c[3].metric("months up", f"{ks['months_up']:.0%}")
    c[4].metric("follows", str(ks["follows"]).replace("_", " "))
    st.caption("All KPIs from the development period. Sharpe = return per unit of swings, the fairest way to compare.")

    # ----- 2. overview and one tab per asset -----
    st.subheader("2. This month")
    sel = ch[(ch["selector"] == ks["follows"]) & (ch["date"].dt.to_period("M") == latest.to_period("M"))].set_index("asset")
    n = now[now["strategy"] == strat].set_index("asset") if now is not None else pd.DataFrame()
    if len(n):
        n = n.assign(eur=n["share"] * amount)
    assets = sorted(sel.index)
    tabs = st.tabs(["All assets"] + assets)
    with tabs[0]:
        rows = []
        for a in assets:
            r = sel.loc[a]
            put = float(n["eur"].get(a, 0.0)) if len(n) else 0.0
            rows.append({"asset": a, "put in": put, "followed": r["main_model"],
                         "expects": "no opinion" if r["p50"] == 0 else f"{r['p50']:+.1%}",
                         "bad .. good": f"{r['p10']:+.1%} .. {r['p90']:+.1%}", "why": n["why"].get(a, "") if len(n) else "",
                         "situation": r["situation"]})
        t = pd.DataFrame(rows)
        cash = amount - t["put in"].sum()
        share = t.set_index("asset")["put in"] / amount
        mid = amount * (1 + (share * sel["p50"]).sum())
        lo = amount * (1 + (share * sel["p10"]).sum())
        hi = amount * (1 + (share * sel["p90"]).sum())
        m = st.columns(4)
        m[0].metric("invested", f"EUR {t['put in'].sum():,.0f}")
        m[1].metric("stays cash", f"EUR {cash:,.0f}")
        m[2].metric("expected in 20 days", f"EUR {mid:,.0f}")
        m[3].metric("bad .. good case", f"EUR {lo:,.0f} .. {hi:,.0f}")
        t["put in"] = t["put in"].map(lambda x: f"EUR {x:,.0f}")
        st.dataframe(t, width="stretch", hide_index=True)
        st.caption("Bad and good case assume everything goes bad (or good) together: the careful view.")
        st.markdown("**What every model expects this month** (middle guess, ★ = trusted in that asset's situation)")
        grid = {}
        for a in assets:
            d = sel.loc[a, "date"]
            mt, _ = models_on(j, book, a, d)
            grid[a] = {mm: f"{r['p50']:+.1%}{' ★' if r['trusted'] else ''}" for mm, r in mt.iterrows()}
        st.dataframe(pd.DataFrame(grid), width="stretch")

    for tab, a in zip(tabs[1:], assets):
        with tab:
            asset_tab(j, book, ch, sel, n, strat, a, amount)


def advice(p50):
    """What one model's forecast means as a decision, in plain words."""
    cost = 2 * (settings.FEE_RATE + settings.SPREAD)
    if p50 > cost + settings.MIN_EDGE:
        return "BUY", f"expects {p50:+.1%} in 20 trading days, more than the {cost:.2%} costs plus margin"
    if p50 < -settings.MIN_EDGE:
        return "WAIT", f"expects {p50:+.1%}: a down month, buying after the drop would be cheaper"
    return "NO CLEAR MOVE", f"expects {p50:+.1%}: smaller than the costs plus margin, no reason to act"


def asset_tab(j, book, ch, sel, n, strat, a, amount):
    r = sel.loc[a]
    d = r["date"]
    mt, situation = models_on(j, book, a, d)
    price = float(r["price"])
    put = float(n["eur"].get(a, 0.0)) if len(n) else 0.0
    c = st.columns(4)
    c[0].metric("price", f"EUR {price:,.2f}")
    c[1].metric("situation", str(situation))
    c[2].metric(f"{strat} puts in", f"EUR {put:,.0f}")
    c[3].metric("follows", r["main_model"])
    st.caption(n["why"].get(a, "") if len(n) else "")

    top = mt.head(settings.TOP_N)
    prices = load("prices", f"{a}.csv").set_index("date")["close_eur"]
    end = d + pd.offsets.BDay(settings.HORIZON_DAYS)
    fig = go.Figure()
    before = prices[(prices.index <= d) & (prices.index >= d - pd.DateOffset(days=180))]
    fig.add_trace(go.Scatter(x=before.index, y=before, name="price", line=dict(color="#888", width=1.6)))
    for i, (mm, row) in enumerate(top.iterrows()):
        col = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(x=[d, end], y=[price, price * (1 + row["p50"])], mode="lines+markers",
                                 name=f"{mm} {row['p50']:+.1%}", line=dict(color=col, width=3)))
        fig.add_trace(go.Scatter(x=[end, end], y=[price * (1 + row["p10"]), price * (1 + row["p90"])], mode="lines",
                                 showlegend=False, line=dict(color=col, width=10), opacity=0.2, hoverinfo="skip"))
    fig.update_layout(height=420, template="plotly_white", title=f"{a}: top {len(top)} models for {situation}, 20 trading days ahead",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    st.markdown(f"**What the top {len(top)} models would do with your money in {a}**")
    default = put if put > 0 else amount / max(len(sel), 1)
    mine = st.number_input(f"try an amount for {a} (EUR)", min_value=1.0, value=float(round(default, 2)), step=25.0, key=f"amt_{a}")
    cost = 2 * (settings.FEE_RATE + settings.SPREAD)
    rows = []
    for mm, row in top.iterrows():
        act, why = advice(row["p50"])
        rows.append({"model": mm, "decision": act, "why": why, "put in": f"EUR {mine:,.2f}" if act == "BUY" else "EUR 0 (wait)",
                     "middle after costs": f"EUR {mine * (1 + row['p50']) * (1 - cost):,.2f}",
                     "bad case": f"EUR {mine * (1 + row['p10']) * (1 - cost):,.2f}",
                     "good case": f"EUR {mine * (1 + row['p90']) * (1 - cost):,.2f}",
                     "trusted?": "★ yes" if row["trusted"] else ("seen years" if row["seen"] else f"no, IC {row['ic']:+.2f}" if pd.notna(row["ic"]) else "no")})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    follow = r["main_model"]
    final = (f"**Final decision ({strat}):** put EUR {put:,.0f} of your EUR {amount:,.0f} in {a}. "
             + ("No model is trusted in this situation, so this is the base plan, not a forecast."
                if follow == "base plan" else f"It follows {follow}, the most trusted model here."))
    st.markdown(final)
    st.caption("A model's advice only counts when it's trusted (★). Untrusted advice is shown so you can see it, not to follow.")

    st.markdown("**Why does a model say this?**")
    pick = st.selectbox("model", mt.index.tolist(), key=f"why_{a}")
    players, tl = live_models()
    if pick in players:
        import ast
        m = players[pick]
        doc = ast.get_docstring(ast.parse(m.file.read_text())) or m.about
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("*How it works*")
            st.write(doc.split("\n\n")[0])
            st.markdown("*Why this month*")
            try:
                for line in m.explain(tl.view(d), a):
                    st.write(line)
            except Exception as e:
                st.write(f"(no explanation available: {e})")
        with c2:
            st.markdown(f"*Its track record on {a}* (only results known on {d.date()})")
            stats, table = track_record(j, a, pick, d)
            if stats is None:
                st.write("No finished forecasts yet (or only in its seen years).")
            else:
                k = st.columns(3)
                k[0].metric("right direction", f"{stats['right']:.0%}" if pd.notna(stats["right"]) else "-",
                            f"always up: {stats['always_up']:.0%}" if pd.notna(stats["always_up"]) else None, delta_color="off")
                k[1].metric("inside its range", f"{stats['in_range']:.0%}", "should be ~80%", delta_color="off")
                k[2].metric("forecasts", f"{stats['forecasts']:,}")
                st.dataframe(table, width="stretch", hide_index=True)

    st.markdown("**This month, every model** (sorted by trust in this situation, only results known today)")
    show = pd.DataFrame({
        "expects": mt["p50"].map(lambda x: f"{x:+.1%}"),
        "expected price": (price * (1 + mt["p50"])).map(lambda x: f"EUR {x:,.2f}"),
        "bad .. good": [f"{x:+.1%} .. {y:+.1%}" for x, y in zip(mt["p10"], mt["p90"])],
        "chance up": [f"{chance_up(x, y, z):.0%}" if pd.notna(chance_up(x, y, z)) else "" for x, y, z in zip(mt["p10"], mt["p50"], mt["p90"])],
        "IC here": mt["ic"].map(lambda x: "" if pd.isna(x) else f"{x:+.2f}"),
        "trust here": mt["trust"].map(lambda x: "" if pd.isna(x) else f"{x:+.1f}"),
        "trusted?": ["★ yes" if t else ("seen years" if s_ else "no") for t, s_ in zip(mt["trusted"], mt["seen"].astype(bool))]})
    st.dataframe(show, width="stretch")

    fs = firsts(j, a)
    prev = [x for x in fs if x < d]
    if prev:
        pd_ = prev[-1]
        pm, psit = models_on(j, book, a, pd_)
        st.markdown(f"**Last month ({pd_.date()}, {psit}): every model vs what really happened**")
        if pm["actual"].notna().any():
            real = float(pm["actual"].dropna().iloc[0])
            pm["miss"] = (pm["p50"] - real).abs()
            win = pm["miss"].idxmin()
            st.dataframe(pd.DataFrame({
                "expected": pm["p50"].map(lambda x: f"{x:+.1%}"), "really": f"{real:+.1%}",
                "direction": ["✅" if np.sign(x) == np.sign(real) and x != 0 else ("no opinion" if x == 0 else "❌") for x in pm["p50"]],
                "inside range": ["✅" if lo_ <= real <= hi_ else "❌" for lo_, hi_ in zip(pm["p10"], pm["p90"])],
                "missed by": pm["miss"].map(lambda x: f"{x:.1%}"), "winner": ["🏆" if mm == win else "" for mm in pm.index]}),
                width="stretch")
        else:
            st.info("Last month's 20 trading days aren't over yet.")


st.title("This month")
need(journal(), 4)
j, book = journal(), skillbook()
ch, k, now = load("selection", "choices.csv"), load("simulation", "kpis.csv"), load("simulation", "now.csv")
need(ch, 5)
need(k, 6)
latest = ch["date"].max()
fr = freshness()
stale = fr[fr["ok"] != "✅"] if len(fr) else fr
st.caption(f"Decision made {latest.date()} with only what was known that day, for the next {settings.HORIZON_DAYS} trading days. "
           + ("Prices are up to date ✅" if stale.empty else f"⚠️ prices are {int(stale['days old'].max())} days old: press Update now in the sidebar")
           + f" (last price {fr['last price'].max() if len(fr) else '?'}, Yahoo Finance).")

assets = sorted(ch["asset"].unique())
mine = [a for a in settings.NOTIFY.get("assets", []) if a in assets] or assets[:1]
c = st.columns([2, 2, 3])
asset = c[0].selectbox("asset", assets, index=assets.index(mine[0]))
amount = c[1].number_input("your amount (EUR)", min_value=10.0, value=float(settings.NOTIFY.get("amount", 1000)), step=50.0)
d, situation, v, mt = all_verdicts()[asset]
s = signal(v, amount)

# ----- the decision card -----
action = {"BUY": f"Put your EUR {amount:,.0f} in {asset} this month (at the next open).",
          "WAIT": f"Don't put money in {asset} this month. Check again on the next decision day.",
          "KEEP YOUR PLAN": f"No forecast-based reason to change your plan. If you buy {asset} every month anyway, "
                            "buy as usual. If you don't, there's no signal to start now either."}[v["call"]]
CALL_STYLE[v["call"]](f"**{action}**", title=f"{asset.upper()}: {v['call']} · {s['label']}")
lead = (v["who"] or list(mt.index[mt["trusted"]]) or [None])[0]
col = st.columns([4, 4, 3])
with col[0]:
    st.markdown("#### Why")
    st.write("• " + v["text"])
    today = j[(j["asset"] == asset) & (j["date"] == d)]
    if "stretch" in today and today["stretch"].notna().any():
        sit_st = load("situations", f"{asset}.csv")
        since = ""
        if sit_st is not None and "stretch_start" in sit_st:
            row = sit_st[sit_st["date"] <= d].tail(1)
            if len(row) and pd.notna(row["stretch_start"].iloc[0]):
                since = f" since {pd.Timestamp(row['stretch_start'].iloc[0]):%d %b %Y} ({int(row['stretch_days'].iloc[0])} days)"
        st.write(f"• Situation: **{situation}**. Stretch (pattern chunk): **{today['stretch'].dropna().iloc[0]}**{since}.")
    if lead and v["call"] == "WAIT":
        r = mt.loc[lead]
        st.write(f"• **{lead}** is the model to listen to for warnings. After its 'fall' calls (all assets, only results it "
                 f"couldn't have seen) the price fell {r['warn_hit_all']:.0%} of the time, against {r['warn_base_all']:.0%} "
                 f"on any day. Warn trust {np.nanmax([r['warn_trust'], r['warn_trust_all']]):.1f} (above {settings.MIN_TRUST} = probably not luck).")
    elif lead:
        stats, _ = track_record(j, asset, lead, d)
        if stats:
            st.write(f"• **{lead}** is the model to listen to here. On {asset} since {stats['since']:%Y} it got the direction "
                     f"right {stats['right']:.0%} of the time (always saying 'up' would be {stats['always_up']:.0%}), "
                     f"and the result was inside its range {stats['in_range']:.0%} of the time.")
        players, tl = live_models()
        if lead in players:
            drivers = [x for x in players[lead].explain(tl.view(d), asset) if x.startswith("•")][:3]
            if drivers:
                st.write(f"• What drives {lead} this month:")
                for x in drivers:
                    st.write("  " + x)
    else:
        st.write("• No model has earned trust for this asset yet (on results it couldn't have seen). "
                 "Until one does, the honest answer is: there's no proven edge, keep your plan.")
with col[1]:
    st.markdown(f"#### What it means for your EUR {amount:,.0f}")
    st.markdown(s["sentence"])
    if pd.notna(s["guess"]):
        with st.container(horizontal=True):
            st.metric("best guess", f"EUR {amount * (1 + s['guess']):,.0f}", f"EUR {amount * s['after_costs']:+,.0f} after costs",
                      border=True)
            st.metric("normal swing", f"± EUR {amount * s['swing']:,.0f}", f"guess = {abs(s['ratio']):.2f}× swing",
                      border=True, delta_color="off")
            cu = chance_up(v["p10"], v["p50"], v["p90"])
            st.metric("chance it ends higher", "-" if pd.isna(cu) else f"{cu:.0%}", border=True)
    calls = past_calls(asset)
    if v["call"] in calls.index:
        r = calls.loc[v["call"]]
        st.caption(f"The last {int(r['times'])} times this rule said {v['call']} on {asset} (only what was known then): "
                   f"it went up {r['went_up']:.0%} of the time, on average {r['avg_move_after']:+.1%} in the {settings.HORIZON_DAYS} "
                   "days after. All calls are in 'past success of this call'.")
with col[2]:
    st.markdown("#### How strong is the signal?")
    st.metric("signal strength", f"{v['confidence']}/100")
    st.progress(v["confidence"] / 100)
    st.caption("NOT the chance that buying is good. It says how much the models agree on a move beyond the costs, "
               "weighted by their track record. Low = nothing special this month: your own plan decides.")
    if st.toggle("past success of this call", key="past_" + asset):
        summ = past_calls(asset)
        if len(summ):
            st.dataframe(summ.style.format({"avg_move_after": "{:+.2%}", "went_up": "{:.0%}", "avg_confidence": "{:.0f}"}),
                         width="stretch")
            st.caption("Same rule, every past month on this asset, only what was known then (holdout hidden).")
with st.expander("Every vote behind this call"):
    st.dataframe(v["votes"], width="stretch")
other_horizons(asset, amount)

# ----- all assets at a glance -----
st.subheader("All assets this month")
rows = []
for a, (da, sit, va, mta) in all_verdicts().items():
    sa = signal(va, amount)
    rows.append({"asset": a, "call": va["call"], "signal": sa["label"], "best guess": va["p50"], "normal swing": sa["swing"],
                 "confidence": va["confidence"], "trusted models": ", ".join(mta.index[mta["trusted"]]) or "none", "situation": sit})
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
    "best guess": st.column_config.NumberColumn(format="percent"), "normal swing": st.column_config.NumberColumn(format="percent")})

with st.expander("Details: strategies and every model per asset"):
    now_details(j, book, ch, k, now, latest, amount)
