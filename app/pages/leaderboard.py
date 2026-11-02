# financial_market_forcasting1/app/pages/leaderboard.py
"""Leaderboard: the ranking per horizon, the warning record, whose results count for trust."""
import plotly.graph_objects as go
import streamlit as st

import settings
from app.common import DATA, load, need, ok



def warning_record(board):
    """Leaderboard section: is it a bad (or good) time to buy? How good each model's fall and rise calls were."""
    if "warn_trust" not in board:
        st.info("No warning record yet: run step 4 again to add it.")
        return
    p = settings.PASS
    st.subheader("Warning record: is it a bad time to buy?")
    st.caption("A fall call = its middle guess below "
               f"{-settings.MIN_EDGE:+.1%}; a rise call = above {2 * (settings.FEE_RATE + settings.SPREAD) + settings.MIN_EDGE:+.1%} "
               "(beats buy + sell costs), the same lines the monthly verdict uses. 'fell after' is how often the price really "
               "fell in the 20 days after its fall calls; 'any day' is how often it fell anyway. Only the difference is skill. "
               f"WAIT on the Now page listens to models with warn trust ≥ {settings.MIN_TRUST} and ≥ {settings.MIN_PERIODS} periods of calls.")
    t = board.copy()
    for k in ("warn", "rise"):
        t[f"✓ {k}"] = ((t[f"{k}_trust"] >= p["trust"]) & (t[f"{k}_periods"] >= settings.MIN_PERIODS)
                       & (t[f"{k}_luck"] < p["luck"])).map(ok)
    t = t.sort_values("warn_trust", ascending=False, na_position="last")
    view = t[["model", "kind", "✓ warn", "warn_periods", "warn_hit", "warn_base", "warn_move", "warn_trust", "warn_luck",
              "✓ rise", "rise_periods", "rise_hit", "rise_base", "rise_move", "rise_trust", "rise_luck"]].rename(columns={
        "warn_periods": "fall calls (periods)", "warn_hit": "fell after", "warn_base": "fell any day",
        "warn_move": "move after vs any day", "warn_trust": "warn trust", "warn_luck": "luck beats it",
        "rise_periods": "rise calls (periods)", "rise_hit": "rose after", "rise_base": "rose any day",
        "rise_move": "move after vs any day ", "rise_trust": "rise trust", "rise_luck": "luck beats it "})
    pct, pp = "{:.0%}", "{:+.2%}"
    st.dataframe(view.style.format({"fall calls (periods)": "{:.0f}", "fell after": pct, "fell any day": pct,
                                    "move after vs any day": pp, "warn trust": "{:+.1f}", "luck beats it": "{:.1%}",
                                    "rise calls (periods)": "{:.0f}", "rose after": pct, "rose any day": pct,
                                    "move after vs any day ": pp, "rise trust": "{:+.1f}", "luck beats it ": "{:.1%}"},
                                   na_rep=""), width="stretch", hide_index=True)
    st.caption("A ✅ needs: trust ≥ 2, enough calls, and luck (results shifted in time) beating it less than 5% of the time. "
               "A model that says 'fall' on most days just gets the 'any day' rate: no edge.")


st.title("Leaderboard")
hs = [h for h in settings.HORIZONS if h == settings.HORIZON_DAYS or (DATA / "battle" / f"leaderboard_h{h}.csv").exists()]
h = st.radio("horizon (trading days ahead)", hs, index=hs.index(settings.HORIZON_DAYS), horizontal=True) if len(hs) > 1 \
    else settings.HORIZON_DAYS
tag = "" if h == settings.HORIZON_DAYS else f"_h{h}"
board, book = load("battle", f"leaderboard{tag}.csv"), load("battle", f"skillbook{tag}.csv")
wins = load("battle", "winners.csv") if not tag else None
need(board, 4)
st.caption(f"Development period ({int(settings.WARMUP_UNTIL[:4]) + 1} to {int(settings.HOLDOUT_FROM[:4]) - 1}), "
           f"{h} trading days ahead, only forecasts a model couldn't have seen in its own training."
           + ("" if not tag else f" Decisions use {settings.HORIZON_DAYS} days; this horizon is information."))
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
warning_record(board)
pools = load("battle", "pooling.csv") if not tag else None
if pools is not None and len(pools):
    st.subheader("Whose results should count for trust?")
    st.caption(f"Trust on an asset can use its own results only, add its group at half weight ({', '.join(settings.GROUPS)}), "
               f"or count every asset the same. Now used: **{settings.EVIDENCE}** (settings.EVIDENCE). 'predicts' = does a "
               "model's trust on 1 January predict its IC on that asset that year (rank correlation, higher = better); "
               "'steady' = trust this year vs next. 'all' gives every asset the same trust, so it's always steady: "
               "only 'predicts' decides.")
    st.dataframe(pools.style.format({"pairs": "{:.0f}", "predicts": "{:+.3f}", "predicts_low": "{:+.3f}", "steady": "{:+.3f}"},
                                    na_rep=""), width="stretch", hide_index=True)
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
