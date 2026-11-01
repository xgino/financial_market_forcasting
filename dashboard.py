# paper-fund-v2/dashboard.py
"""
The dashboard. Start it with:   streamlit run dashboard.py
It opens in your browser. Pages (left side):
  Run          start any step with a button and watch its output
  Battle       month by month replay: every model's guess as a line, the winner, previous / next
  Leaderboard  the ranking, wins over time, and how trust in a model moved
  Training     live view of a training run (refreshes by itself)
  Data         the prices
  Now          this month: what the best models expect, what each strategy would do with EUR 1000
  Results      every strategy month by month: fund value, green/red months, KPIs with pass marks
  Models       what every model is, what it was trained on, strengths, weaknesses, limits, its scores
  Email        the monthly email: settings, what was sent, a preview
  Calculator   try amounts, models and your own numbers (only reads)
  Judge        the plain-language report: what's good, what needs a look
Everything it shows comes from the CSV files in data/, so closing it loses nothing.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings  # noqa: E402

DATA = settings.DATA
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#9D755D", "#BAB0AC"]
WIN = "#2e9e5b"
st.set_page_config(page_title="Paper fund", layout="wide")


@st.cache_data
def csv(path, mtime, dates=("date", "end_date", "as_of")):
    df = pd.read_csv(path)
    for c in dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def load(*parts):
    p = DATA.joinpath(*parts)
    return csv(p, p.stat().st_mtime) if p.exists() else None


def need(df, step):
    if df is None:
        st.info(f"Nothing here yet. Run step {step} first (page Run, or: python main.py {step}).")
        st.stop()


# ---------- pages ----------

def run_main(args, label):
    cmd = [sys.executable, "main.py", *args]
    st.write(f"`{' '.join(cmd[1:])}` ({label})")
    box, lines = st.empty(), []
    proc = subprocess.Popen(cmd, cwd=settings.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        lines.append(line.rstrip("\n").split("\r")[-1])
        box.code("\n".join(lines[-25:]))
    proc.wait()
    st.cache_data.clear()
    (st.success if proc.returncode == 0 else st.error)("done" if proc.returncode == 0 else "stopped with an error")


def ok(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else ("✅" if bool(v) else "❌")


from core.verdict import chance_up, firsts, history, history_summary, models_on, track_record, verdict  # noqa: E402

def pick_strategies(k):
    """Best first: passes the most pass marks, then the best Sharpe."""
    t = k.copy()
    marks = [c for c in ["beats hold", "drop ok", "sharpe ok"] if c in t]
    t["passes"] = t[marks].apply(lambda r: sum(bool(x) for x in r if pd.notna(x) and x != ""), axis=1)
    return t.sort_values(["passes", "sharpe"], ascending=False)


def live_models():
    """The model files and the timeline, loaded once, for the explanations."""
    from core.contracts import Model
    from core.plugins import load_players
    from core.timeline import Timeline

    return load_players(settings.ROOT / "04_models", Model, settings.HORIZON_DAYS), Timeline.load()


@st.cache_data
@st.cache_data
def verdict_history(jm, bm):
    return history(load("battle", "journal.csv"), load("battle", "skillbook.csv"))


CALL_STYLE = {"BUY": st.success, "WAIT": st.error, "KEEP YOUR PLAN": st.warning}


def freshness():
    """Last price date per traded asset and how old it is."""
    today = pd.Timestamp.today().normalize()
    rows = []
    for a, spec in settings.ASSETS.items():
        p = DATA / "prices" / f"{a}.csv"
        if p.exists():
            last = load("prices", f"{a}.csv")["date"].max()
            age = (today - last).days
            rows.append({"asset": a, "last price": last.date(), "days old": age, "ok": "✅" if age <= 4 else "⚠️",
                         "source": f"https://finance.yahoo.com/quote/{spec['ticker']}"})
    return pd.DataFrame(rows)


def money_chart(amount, p10, p50, p90):
    lo, mid, hi = amount * (1 + p10), amount * (1 + p50), amount * (1 + p90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="lines", line=dict(color="#9ab", width=26), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[mid], y=[0], mode="markers+text", marker=dict(size=22, color="#333", symbol="diamond"),
                             text=[f"middle EUR {mid:,.0f}"], textposition="top center", showlegend=False))
    fig.add_vline(x=amount, line_dash="dash", line_color=WIN, annotation_text=f"you put in EUR {amount:,.0f}")
    fig.add_annotation(x=lo, y=0, text=f"bad case<br>EUR {lo:,.0f}", showarrow=False, yshift=-34)
    fig.add_annotation(x=hi, y=0, text=f"good case<br>EUR {hi:,.0f}", showarrow=False, yshift=-34)
    fig.update_layout(height=190, template="plotly_white", margin=dict(l=10, r=10, t=30, b=40),
                      yaxis=dict(visible=False), xaxis=dict(title=None, tickprefix="EUR "))
    return fig


def page_now():
    j, book = load("battle", "journal.csv"), load("battle", "skillbook.csv")
    ch, k, now = load("selection", "choices.csv"), load("simulation", "kpis.csv"), load("simulation", "now.csv")
    head = st.columns([3, 1])
    if head[1].button("Get new data + update all", type="primary"):
        run_main([], "every step, only what's new")
    need(ch, 5)
    need(k, 6)
    latest = ch["date"].max()
    head[0].header(f"Now: {latest:%B %Y}")
    fr = freshness()
    stale = fr[fr["ok"] != "✅"] if len(fr) else fr
    st.caption(f"Decision made {latest.date()} with only what was known that day, for the next 20 trading days. "
               + ("Prices are up to date ✅" if stale.empty else f"⚠️ prices are {int(stale['days old'].max())} days old, press update")
               + f" (last price {fr['last price'].max() if len(fr) else '?'}, Yahoo Finance).")
    if latest.to_period("M") < pd.Timestamp.today().to_period("M"):
        st.warning(f"Monthly update due: the last decision is from {latest:%B %Y}. Press 'Get new data + update all'.")

    assets = sorted(ch["asset"].unique())
    mine = [a for a in settings.NOTIFY.get("assets", []) if a in assets] or assets[:1]
    c = st.columns([2, 2, 3])
    asset = c[0].selectbox("asset", assets, index=assets.index(mine[0]))
    amount = c[1].number_input("your amount (EUR)", min_value=10.0, value=float(settings.NOTIFY.get("amount", 1000)), step=50.0)
    d = firsts(j, asset)[-1]
    mt, situation = models_on(j, book, asset, d)
    v = verdict(mt)

    # ----- the decision card -----
    action = {"BUY": f"Put your EUR {amount:,.0f} in {asset} this month (at the next open).",
              "WAIT": f"Don't put money in {asset} this month. Check again on the next decision day.",
              "KEEP YOUR PLAN": f"No forecast-based reason to change your plan. If you buy {asset} every month anyway, "
                                "buy as usual. If you don't, there's no signal to start now either."}[v["call"]]
    CALL_STYLE[v["call"]](f"## {asset.upper()}: {v['call']}\n**{action}**")
    lead = (v["who"] or list(mt.index[mt["trusted"]]) or [None])[0]
    col = st.columns([4, 4, 3])
    with col[0]:
        st.markdown("#### Why")
        st.write("• " + v["text"])
        if lead:
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
        st.markdown("#### What could happen to your money (20 trading days)")
        if pd.notna(v["p10"]):
            st.plotly_chart(money_chart(amount, v["p10"], v["p50"], v["p90"]), width="stretch")
            cu = chance_up(v["p10"], v["p50"], v["p90"])
            cost = 2 * (settings.FEE_RATE + settings.SPREAD)
            st.write(f"Chance of ending higher: about **{cu:.0%}**. Costs to buy and later sell: about "
                     f"EUR {amount * cost:,.2f}. Bad and good case: 1 in 10 chance it ends worse / better.")
    with col[2]:
        st.markdown("#### How strong is this month's signal?")
        st.metric("signal strength", f"{v['confidence']}/100")
        st.progress(v["confidence"] / 100)
        st.caption("This is NOT the chance that buying is good or bad. It says how much the models agree on a move "
                   "beyond the costs, weighted by their track record. Low = the models see nothing special this month, "
                   "so timing doesn't matter much: your own plan decides. High = an unusual month worth acting on.")
        if st.toggle("past success of this call", key="past_" + asset):
            jp, bp = DATA / "battle" / "journal.csv", DATA / "battle" / "skillbook.csv"
            hs = verdict_history(jp.stat().st_mtime, bp.stat().st_mtime)
            summ = history_summary(hs[hs["asset"] == asset])
            if len(summ):
                st.dataframe(summ.style.format({"avg_move_after": "{:+.2%}", "went_up": "{:.0%}", "avg_confidence": "{:.0f}"}),
                             width="stretch")
                st.caption("Same rule, every past month on this asset, only what was known then (holdout hidden).")
    with st.expander("Every vote behind this call"):
        st.dataframe(v["votes"], width="stretch")

    # ----- long term: no models, just this asset's own history -----
    st.subheader(f"Long-term view: {asset} in EUR (no models, just its history)")
    px = load("prices", f"{asset}.csv").set_index("date")["close_eur"]
    rows = []
    for label, h in (("1 month", 21), ("1 year", 252), ("3 years", 756), ("5 years", 1260), ("10 years", 2520)):
        r = (px.shift(-h) / px - 1).dropna()
        if len(r) > 250:
            rows.append({"held for": label, "ended higher": f"{(r > 0).mean():.0%}", "middle result": f"{r.median():+.0%}",
                         "worst 10% ended below": f"{r.quantile(0.1):+.0%}", "best 10% ended above": f"{r.quantile(0.9):+.0%}",
                         f"EUR {amount:,.0f} became (middle)": f"EUR {amount * (1 + r.median()):,.0f}", "start days": len(r)})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(f"Every possible buy day from {px.index[0]:%Y} to now, and what it was worth after holding. The longer you "
                   "hold, the less one month's timing matters. History is not a promise: this is one asset over ~20 years.")

    # ----- all assets at a glance -----
    st.subheader("All assets this month")
    rows = []
    for a in assets:
        mta, sit = models_on(j, book, a, firsts(j, a)[-1])
        va = verdict(mta)
        rows.append({"asset": a, "call": va["call"], "confidence": va["confidence"], "middle guess": f"{va['p50']:+.1%}",
                     "bad .. good": f"{va['p10']:+.1%} .. {va['p90']:+.1%}" if pd.notna(va["p10"]) else "",
                     "trusted models": ", ".join(mta.index[mta["trusted"]]) or "none", "situation": sit})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("Details: strategies and every model per asset"):
        now_details(j, book, ch, k, now, latest, amount)


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


def page_judge():
    st.header("Judge")
    if st.button("Run the judge now"):
        run_main(["7"], "step 7")
    p = DATA / "judge" / "report.md"
    if not p.exists():
        st.info("No report yet. Run step 7 (button above, or: python main.py 7).")
        return
    st.markdown(p.read_text())
    h = load("judge", "history.csv")
    if h is not None and len(h):
        st.subheader("Earlier reports")
        t = h.groupby(["time", "level"]).size().unstack(fill_value=0)
        st.dataframe(t.iloc[::-1], width="stretch")


def page_calc():
    st.header("Calculator")
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


def page_results():
    st.header("Results: EUR 1000 per strategy, month by month")
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


def page_run():
    st.header("Run")
    st.caption("Same as typing python main.py ... in the terminal. Every step only does what's missing.")
    from core.plugins import load_file, step_folders

    steps = [(n, f, load_file(f / "_step.py").STEP) for n, f in step_folders() if (f / "_step.py").exists()]
    c1, c2, c3 = st.columns([2, 2, 1])
    only = c1.text_input("only these players (optional, comma separated)")
    fresh = c2.checkbox("fresh (redo from scratch)")
    check = c3.checkbox("no-future check")
    cols = st.columns(len(steps) + 1)
    todo = None
    if cols[0].button("Run all", type="primary"):
        todo = []
    for i, (n, f, name) in enumerate(steps, start=1):
        if cols[i].button(f"{n:02d} {name}"):
            todo = [str(n)]
    if todo is not None:
        cmd = [sys.executable, "main.py", *todo] + (["--only", only] if only else []) + \
              (["--fresh"] if fresh else []) + (["--check"] if check else [])
        st.write("`" + " ".join(cmd[1:]) + "`")
        box, lines = st.empty(), []
        proc = subprocess.Popen(cmd, cwd=settings.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            lines.append(line.rstrip("\n").split("\r")[-1])
            box.code("\n".join(lines[-40:]))
        proc.wait()
        st.success("done") if proc.returncode == 0 else st.error("stopped with an error, see above")
    log = load("runs", "run_log.csv")
    if log is not None:
        st.subheader("Earlier runs")
        st.dataframe(log.iloc[::-1], width="stretch", hide_index=True)


def page_battle():
    st.header("Battle: month by month")
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
        return
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
        return
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


def page_leaderboard():
    st.header("Leaderboard")
    board, wins, book = load("battle", "leaderboard.csv"), load("battle", "winners.csv"), load("battle", "skillbook.csv")
    need(board, 4)
    st.caption(f"Development period ({int(settings.WARMUP_UNTIL[:4]) + 1} to {int(settings.HOLDOUT_FROM[:4]) - 1}), "
               "only forecasts a model couldn't have seen in its own training.")
    p = settings.PASS
    board = board.copy()
    yrs = board["years_positive"].astype(str).str.extract(r"(\d+) of (\d+)").astype(float)
    board["✓ trust"] = (board["trust"] >= p["trust"]).map(ok)
    board["✓ luck"] = (board["luck_beats_it"] < p["luck"]).map(ok)
    board["✓ years"] = (yrs[0] / yrs[1] >= p["years"]).map(ok)
    board["✓ skill"] = (board["skill"] > p["skill"]).map(ok)
    board["✓ range"] = board["range_hit"].between(*p["range"]).map(ok)
    board["passes"] = board[["✓ trust", "✓ luck", "✓ years", "✓ skill", "✓ range"]].apply(lambda r: (r == "✅").sum(), axis=1)
    st.caption(f"Pass marks: trust ≥ {p['trust']}, luck beats it < {p['luck']:.0%}, positive in ≥ {p['years']:.0%} of years, "
               f"skill > 0 (beats 'no change'), range hit {p['range'][0]:.0%} to {p['range'][1]:.0%}. A model needs all 5.")
    view = board[["rank", "model", "kind", "passes", "✓ trust", "✓ luck", "✓ years", "✓ skill", "✓ range", "periods",
                  "ic", "trust", "years_positive", "luck_beats_it", "direction", "always_up", "skill", "range_hit", "verdict"]]
    st.dataframe(view.style.format({"periods": "{:.0f}", "ic": "{:+.3f}", "trust": "{:+.1f}", "luck_beats_it": "{:.1%}",
                                    "direction": "{:.1%}", "always_up": "{:.1%}", "skill": "{:+.1%}",
                                    "range_hit": "{:.0%}"}, na_rep=""), width="stretch", hide_index=True)
    if wins is not None and len(wins):
        st.subheader("Monthly wins per year (closest price guess)")
        w = wins.assign(year=wins["date"].dt.year).groupby(["year", "winner"]).size().unstack(fill_value=0)
        fig = go.Figure([go.Bar(x=w.index, y=w[c], name=c) for c in w.columns])
        fig.update_layout(barmode="stack", height=380, template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    if book is not None and len(book):
        st.subheader("How trust moved over time")
        m = st.selectbox("model", board["model"].tolist())
        h = book[(book["model"] == m) & (book["scope"] == "all") & (book["situation"] == "all")]
        fig = go.Figure([go.Scatter(x=h["as_of"], y=h["trust"], name="trust")])
        fig.add_hline(y=settings.MIN_TRUST, line_dash="dash", annotation_text="trusted above this")
        fig.update_layout(height=320, template="plotly_white", yaxis_title="trust (IC x sqrt(periods))")
        st.plotly_chart(fig, width="stretch")


def page_training():
    st.header("Training")
    folder = DATA / "training"
    runs = sorted(p.name for p in folder.iterdir()) if folder.exists() else []
    if not runs:
        st.info("Nothing trained yet. Run step 3 (page Run, or: python main.py 3).")
        return
    name = st.selectbox("model", runs)
    live = st.toggle("live (refresh every 3 seconds)", value=False)
    log = load("training", name, "log.csv")
    if log is not None and len(log):
        params = [c for c in log.columns if c not in ("cutoff", "cv_loss", "rows", "best")]
        log["combo"] = log[params].astype(str).agg(", ".join, axis=1)
        fig = go.Figure()
        for combo, g in log.groupby("combo"):
            fig.add_trace(go.Scatter(x=g["cutoff"], y=g["cv_loss"], mode="lines+markers", name=combo))
        best = log[log["best"] == True]  # noqa: E712
        fig.add_trace(go.Scatter(x=best["cutoff"], y=best["cv_loss"], mode="markers", name="picked",
                                 marker=dict(size=13, symbol="star", color=WIN)))
        fig.update_layout(height=460, template="plotly_white", yaxis_title="error on the held back folds (lower = better)",
                          xaxis_title="checkpoint (trained with data up to this date)", legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{log['cutoff'].nunique()} checkpoints so far, {len(log)} tries. ★ = the combination that won "
                   "for that checkpoint. All scored on time-ordered folds of the training data only.")
    index = DATA / "checkpoints" / name / "index.csv"
    if index.exists():
        st.dataframe(pd.read_csv(index).drop(columns=["settings"], errors="ignore"), width="stretch", hide_index=True)
    if live:
        time.sleep(3)
        st.rerun()


def page_data():
    st.header("Data")
    page_data_status()
    folder = DATA / "prices"
    files = sorted(p.stem for p in folder.glob("*.csv")) if folder.exists() else []
    if not files:
        st.info("No prices yet. Run step 1.")
        return
    asset = st.selectbox("asset", files)
    df = load("prices", f"{asset}.csv")
    fig = go.Figure([go.Scatter(x=df["date"], y=df["close_eur"], name=asset, line=dict(width=1.2))])
    fig.add_vline(x=pd.Timestamp(settings.HOLDOUT_FROM), line_dash="dash", line_color="red")
    fig.update_layout(height=460, template="plotly_white", yaxis_title="EUR", title=f"{asset} (red line: holdout starts)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.tail(20).iloc[::-1], width="stretch", hide_index=True)


def page_models():
    st.header("Models: what is what")
    st.caption("Every model in 04_models, what kind it is, what it was trained on, what it's good and bad at, and how it "
               "scores right now (development period, only results it couldn't have seen).")
    from core.plugins import load_file

    players, _ = live_models()
    board = load("battle", "leaderboard.csv")
    board = board.set_index("model") if board is not None else pd.DataFrame()
    order = {"your own": 0, "pretrained": 1, "statistics": 2, "rule": 3, "baseline": 4}
    cards = []
    for name, m in players.items():
        card = dict(getattr(load_file(m.file), "CARD", {}))
        cards.append((min((v for k_, v in order.items() if k_ in card.get("type", "")), default=5), name, m, card))
    first_price = min((load("prices", f"{a}.csv")["date"].min() for a in settings.ALL_ASSETS
                       if (DATA / "prices" / f"{a}.csv").exists()), default=None)
    for _, name, m, card in sorted(cards, key=lambda x: (x[0], x[1])):
        with st.container(border=True):
            st.markdown(f"### {name}  ·  *{card.get('type', m.kind)}*")
            trained = card.get("trained", "")
            if trained == "DYNAMIC":
                idx = DATA / "checkpoints" / name / "index.csv"
                if idx.exists():
                    ix = pd.read_csv(idx)
                    n_assets = sum((DATA / "prices" / f"{a}.csv").exists() for a in settings.ALL_ASSETS)
                    trained = (f"Trained by you, walk-forward: {len(ix)} checkpoints ({ix['cutoff'].iloc[0]} to "
                               f"{ix['cutoff'].iloc[-1]}), each only on results finished before its date. Data: {n_assets} assets "
                               f"(traded + training-only) from {first_price:%Y} on, newest checkpoint {int(ix['rows'].iloc[-1]):,} rows. "
                               "Settings picked by GridSearchCV on time-ordered folds, then trained on all rows.")
                else:
                    trained = "Not trained yet: run step 3."
            c1, c2 = st.columns(2)
            c1.write(f"**What it is:** {m.about}")
            c1.write(f"**Trained on:** {trained}")
            if m.seen_until:
                c1.write(f"**Only trusted on results after** {m.seen_until} (it may have seen older prices while it was trained).")
            src = card.get("source", "")
            c1.write(f"**Source:** {src}")
            c2.write(f"**Strong at:** {card.get('strengths', '')}")
            c2.write(f"**Weak at:** {card.get('weaknesses', '')}")
            c2.write(f"**Limits:** {card.get('limits', '')}")
            if name in board.index:
                r = board.loc[name]
                k = st.columns(6)
                k[0].metric("rank", int(r["rank"]))
                k[1].metric("IC", "" if pd.isna(r["ic"]) else f"{r['ic']:+.3f}")
                k[2].metric("trust", "" if pd.isna(r["trust"]) else f"{r['trust']:+.1f}", "✅ ≥ 2" if pd.notna(r["trust"]) and r["trust"] >= 2 else "❌ < 2", delta_color="off")
                k[3].metric("years positive", str(r["years_positive"]) if pd.notna(r["years_positive"]) else "")
                k[4].metric("direction", "" if pd.isna(r["direction"]) else f"{r['direction']:.0%}", f"always up {r['always_up']:.0%}", delta_color="off")
                k[5].metric("range hit", f"{r['range_hit']:.0%}", "should be ~80%", delta_color="off")
                st.caption(f"Verdict: {r['verdict']}")
            elif m.seen_until:
                st.caption("Not on the 2010-2018 leaderboard: all those years are 'seen' for it. It earns trust from its release on.")


def page_email():
    st.header("Email")
    cfg = settings.NOTIFY
    st.write(f"Mails about **{', '.join(cfg['assets'])}**, only when the call is **{', '.join(cfg['only_when'])}** "
             f"with confidence ≥ {cfg['min_confidence']}. Change it in settings.py (NOTIFY).")
    ok = (settings.ROOT / "secrets.env").exists()
    st.write("Mail account: " + ("✅ secrets.env found" if ok else "❌ no secrets.env yet: copy secrets.example.env and fill it in"))
    if st.button("Make this month's email now (sends only if it matters)"):
        run_main(["8"], "step 8")
    sent = load("notify", "sent.csv")
    if sent is not None and len(sent):
        st.subheader("Sent")
        st.dataframe(sent.iloc[::-1], width="stretch", hide_index=True)
    prev = DATA / "notify" / "last_email.html"
    if prev.exists():
        st.subheader("Preview of this month's email")
        import streamlit.components.v1 as components
        components.html(prev.read_text(), height=700, scrolling=True)


def page_data_status():
    st.subheader("Is the data up to date?")
    fr = freshness()
    if len(fr):
        st.dataframe(fr, width="stretch", hide_index=True, column_config={"source": st.column_config.LinkColumn("source")})
    rows = []
    for name, sid in settings.MACRO_SERIES.items():
        f = DATA / "macro" / f"{name}.csv"
        if f.exists():
            last = pd.read_csv(f, parse_dates=["date"])["date"].max()
            rows.append({"macro": name, "last value": last.date(), "days old": (pd.Timestamp.today() - last).days,
                         "source": f"https://fred.stlouisfed.org/series/{sid}"})
        else:
            rows.append({"macro": name, "last value": "missing", "days old": None, "source": f"https://fred.stlouisfed.org/series/{sid}"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={"source": st.column_config.LinkColumn("source")})
    st.caption("Prices: Yahoo Finance, daily closes, converted to EUR. Macro: FRED (or Yahoo as back-up), each value only used "
               "from the next trading day. Press 'Get new data + update all' on the Now page to refresh.")


PAGES = {"Now": page_now, "Models": page_models, "Results": page_results, "Calculator": page_calc,
         "Judge": page_judge, "Email": page_email, "Run": page_run, "Battle": page_battle, "Leaderboard": page_leaderboard,
         "Training": page_training, "Data": page_data}
choice = st.sidebar.radio("page", list(PAGES))
st.sidebar.caption("Holdout " + ("SHOWN" if settings.SHOW_HOLDOUT else "hidden") + f" (from {settings.HOLDOUT_FROM})")
PAGES[choice]()