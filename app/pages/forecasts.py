# financial_market_forcasting1/app/pages/forecasts.py
"""Forecasts: the price, what the models expect ahead, and how a model's past guesses compared with what really
happened. Short term (5 days), a month (20) or a quarter (60), and the long term from the asset's own history.

Honesty: the past guesses are only shown for the development years. The holdout (settings.HOLDOUT_FROM on) is the
final test; looking at how models did there before it's opened would spoil it. Forecasts whose result isn't known
yet (the newest ones, pointing into the future) are always shown."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import settings
from app.common import COLORS, COST, HORIZON_LABEL, horizons_ready, journal, load, need, proven, suffix
from core.metrics import _ic

HOLDOUT = pd.Timestamp(settings.HOLDOUT_FROM)
st.title("Forecasts")
hs = horizons_ready()
need(hs or None, 4)

c = st.columns([2, 3, 3], vertical_alignment="bottom")
h = c[1].segmented_control("looking ahead", hs, default=settings.HORIZON_DAYS if settings.HORIZON_DAYS in hs else hs[0],
                           format_func=lambda x: HORIZON_LABEL.get(x, f"{x} days"), required=True)
j = journal(h)
assets = sorted(j["asset"].unique())
mine = [a for a in settings.NOTIFY.get("assets", []) if a in assets] or assets[:1]
asset = c[0].selectbox("asset", assets, index=assets.index(mine[0]))
board = load("battle", f"leaderboard{suffix(h)}.csv")
ranked = [m for m in (board["model"].tolist() if board is not None else []) if m != "naive"]
models = ranked + sorted(set(j["model"].unique()) - set(ranked) - {"naive"})
good = proven(board)
chosen = c[2].multiselect("models", models, default=(good or ranked)[:3],
                          help="Default: the models with a proven edge on this horizon (else the leaderboard's top 3).")

px = load("prices", f"{asset}.csv", usecols=["date", "close_eur"]).set_index("date")["close_eur"]
ja = j[(j["asset"] == asset) & j["model"].isin(chosen)].copy()
# the day each guess is about: its end_date, or (not over yet) h trading days after it was made
ja["target"] = ja["end_date"].fillna(ja["date"] + pd.offsets.BDay(h))
ja["lo"], ja["mid"], ja["hi"] = (ja["price"] * (1 + ja[q]) for q in ("p10", "p50", "p90"))


def band(fig, g, col, name, show_band=True):
    if show_band:
        fig.add_trace(go.Scatter(x=pd.concat([g["target"], g["target"][::-1]]), y=pd.concat([g["hi"], g["lo"][::-1]]),
                                 fill="toself", fillcolor=col, opacity=0.12, line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
    fig.add_trace(go.Scatter(x=g["target"], y=g["mid"], name=name, line=dict(color=col, width=2),
                             customdata=np.stack([g["date"].dt.strftime("%d %b %Y"), g["p50"] * 100], axis=-1),
                             hovertemplate="guessed on %{customdata[0]}: %{customdata[1]:+.1f}% → EUR %{y:,.2f}<extra>" + name
                                           + "</extra>"))


# ----- now: the last year and what's expected ahead -----
st.subheader(f"Now: {asset} and the next {HORIZON_LABEL.get(h, f'{h} days')}")
live = ja[ja["actual"].isna()]                                   # the result isn't known yet
last_day = px.index[-1]
fig = go.Figure()
recent = px[px.index >= last_day - pd.DateOffset(years=1)]
fig.add_trace(go.Scatter(x=recent.index, y=recent, name="real price", line=dict(color="#444", width=2)))
rows = []
for i, m in enumerate(chosen):
    g = live[live["model"] == m].sort_values("target")
    if g.empty:
        continue
    col = COLORS[i % len(COLORS)]
    band(fig, g, col, m)
    newest = g.iloc[-1]
    rows.append({"model": m, "proven": "✅" if m in good else "", "made on": newest["date"].date(),
                 "for": newest["target"].date(), "expects": newest["p50"], "after costs": newest["p50"] - COST,
                 "bad case": newest["p10"], "good case": newest["p90"], "price then": newest["mid"],
                 "seen years": "yes (not trusted)" if bool(newest["seen"]) else ""})
fig.add_vline(x=last_day, line_dash="dot", line_color="#888")
fig.add_annotation(x=last_day, y=1, yref="paper", text="last price", showarrow=False, yshift=10)
fig.update_layout(height=430, template="plotly_white", yaxis_title="EUR", legend=dict(orientation="h", y=-0.12),
                  margin=dict(t=30))
st.plotly_chart(fig, width="stretch", alt=f"{asset} price over the last year with each model's forecast for the next {h} days")
st.caption(f"Each coloured line is a model's guesses whose {h} days aren't over yet, drawn on the day they're about "
           "(the shaded band: its bad to good case, 8 in 10 results should land inside). Where the line goes past the "
           "dotted line is the forecast ahead.")
if rows:
    t = pd.DataFrame(rows)
    st.dataframe(t, width="stretch", hide_index=True, column_config={
        "expects": st.column_config.NumberColumn(format="percent"), "after costs": st.column_config.NumberColumn(format="percent"),
        "bad case": st.column_config.NumberColumn(format="percent"), "good case": st.column_config.NumberColumn(format="percent"),
        "price then": st.column_config.NumberColumn("expected price", format="€ %.2f")})

# ----- track record: every guess vs what really happened -----
st.subheader("Track record: every guess against what really happened")
done = ja[ja["actual"].notna() & ((ja["target"] < HOLDOUT) | settings.SHOW_HOLDOUT)]
if done.empty:
    st.info("No finished guesses in the development years for these models.")
else:
    c2 = st.columns([2, 3], vertical_alignment="bottom")
    pick = c2[0].selectbox("model", [m for m in chosen if m in set(done["model"])])
    g = done[done["model"] == pick].sort_values("target")
    honest = g[~g["seen"].astype(bool)]
    years = sorted(g["target"].dt.year.unique())
    span = c2[1].select_slider("years", options=years, value=(max(years[0], years[-1] - 2), years[-1]))
    g = g[g["target"].dt.year.between(*span)]
    gh = honest[honest["target"].dt.year.between(*span)]
    fig = go.Figure()
    real = px[(px.index >= g["target"].min()) & (px.index <= g["target"].max())]
    band(fig, g, COLORS[0], f"{pick}: guess for that day")
    fig.add_trace(go.Scatter(x=real.index, y=real, name="real price", line=dict(color="#222", width=1.6)))
    if g["seen"].astype(bool).any():
        s = g[g["seen"].astype(bool)]
        fig.add_vrect(x0=s["target"].min(), x1=s["target"].max(), fillcolor="#999", opacity=0.12, line_width=0,
                      annotation_text="seen years: it may know these prices", annotation_position="top left")
    fig.update_layout(height=430, template="plotly_white", yaxis_title="EUR", legend=dict(orientation="h", y=-0.12),
                      margin=dict(t=30))
    st.plotly_chart(fig, width="stretch", alt=f"{pick}'s past {h} day guesses for {asset} against the real price")
    st.caption(f"Every day {pick} guessed the price {h} trading days later; the blue line is that guess drawn on the day it "
               "was about, the black line what really happened. Close lines = good guesses. "
               + ("" if settings.SHOW_HOLDOUT else f"Hidden from {HOLDOUT:%Y} on: that's the holdout, the final test."))
    if gh.empty:
        st.warning(f"All of {pick}'s guesses in these years are from its seen years (it may have been trained on these prices), "
                   "so they say nothing about its skill. Its honest results start after its release, inside the holdout.",
                   icon=":material/visibility_off:")
    else:
        said = gh[gh["p50"] != 0]
        right = (np.sign(said["p50"]) == np.sign(said["actual"])).mean() if len(said) else np.nan
        up = (said["actual"] > 0).mean() if len(said) else np.nan
        miss, flat = (gh["p50"] - gh["actual"]).abs().mean(), gh["actual"].abs().mean()
        inside = ((gh["actual"] >= gh["p10"]) & (gh["actual"] <= gh["p90"])).mean()
        with st.container(horizontal=True):
            st.metric("right direction", "-" if pd.isna(right) else f"{right:.0%}",
                      None if pd.isna(right) else f"{right - up:+.0%} vs always 'up'", border=True,
                      help="How often it got up or down right. Always saying 'up' gets the second number for free.")
            st.metric("average miss", f"{miss:.1%}", f"{flat - miss:+.1%} vs guessing 'no change'", border=True,
                      help="How far its middle guess was from the result, on average. Positive delta = better than no change.")
            st.metric("inside its range", f"{inside:.0%}", f"{inside - 0.8:+.0%} vs the 80% it should be", border=True,
                      delta_color="off")
            st.metric("ranking skill (IC)", f"{_ic(gh['p50'].to_numpy(), gh['actual'].to_numpy()):+.3f}",
                      f"{len(gh):,} guesses", border=True, delta_color="off",
                      help="Do its bigger guesses come before bigger moves? 0 = no, above ~0.05 is useful (if it's not luck: "
                           "see the Leaderboard).")

# ----- long term: no models, just this asset's own history -----
st.subheader(f"Long term: {asset} held for years (no models, just its history)")
rows = []
for label, n in (("1 month", 21), ("1 year", 252), ("3 years", 756), ("5 years", 1260), ("10 years", 2520)):
    r = (px.shift(-n) / px - 1).dropna()
    if len(r) > 250:
        rows.append({"held for": label, "ended higher": (r > 0).mean(), "middle result": r.median(),
                     "worst 10% below": r.quantile(0.1), "best 10% above": r.quantile(0.9), "start days": len(r)})
if rows:
    t = pd.DataFrame(rows)
    pct = st.column_config.NumberColumn(format="percent")
    st.dataframe(t, width="stretch", hide_index=True, column_config={
        "ended higher": pct, "middle result": pct, "worst 10% below": pct, "best 10% above": pct})
    st.caption(f"Every possible buy day from {px.index[0]:%Y} to now and what it was worth after holding. The longer you hold, "
               "the less one month's timing matters. History is not a promise.")
