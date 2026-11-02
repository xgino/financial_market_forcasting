# financial_market_forcasting1/app/pages/models.py
"""Models: what every model is, what it was trained on, strengths, weaknesses, limits, its scores."""
import pandas as pd
import streamlit as st

import settings
from app.common import DATA, live_models, load


st.title("Models: what is what")
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
            if pd.notna(r.get("warn_hit")):
                st.caption(f"Warning record: after its {r['warn_periods']:.0f} periods of 'fall' calls the price fell "
                           f"{r['warn_hit']:.0%} of the time (any day: {r['warn_base']:.0%}), warn trust {r['warn_trust']:+.1f}. "
                           + (f"After 'rise' calls it rose {r['rise_hit']:.0%} (any day {r['rise_base']:.0%})."
                              if pd.notna(r.get("rise_hit")) else ""))
        elif m.seen_until:
            st.caption("Not on the 2010-2018 leaderboard: all those years are 'seen' for it. It earns trust from its release on.")
