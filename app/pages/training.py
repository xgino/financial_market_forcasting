# financial_market_forcasting1/app/pages/training.py
"""Training: live view of a training run (checkpoints, tries, the picked settings)."""
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import DATA, WIN, load


st.title("Training")
folder = DATA / "training"
runs = sorted(p.name for p in folder.iterdir()) if folder.exists() else []
if not runs:
    st.info("Nothing trained yet. Run step 3 (page Run, or: python main.py 3).")
    st.stop()
name = st.selectbox("model", runs)
live = st.toggle("live (refresh every 3 seconds)", value=False)
log = load("training", name, "log.csv")
log = log.copy() if log is not None else None
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
