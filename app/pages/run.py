# financial_market_forcasting1/app/pages/run.py
"""Run: start any step in the background and follow its live output. The run is its own process: it keeps going
when you click to another page or close the dashboard, and a second run can't start while one is going."""
import streamlit as st

from app.common import load, start_run
from core import runs
from core.plugins import load_file, step_folders

st.title("Run")
st.caption("Same as typing python main.py ... in the terminal. Every step only does what's missing, so a stopped run "
           "continues where it was next time.")

steps = [(n, f, load_file(f / "_step.py").STEP) for n, f in step_folders() if (f / "_step.py").exists()]
with st.container(border=True):
    c = st.columns([3, 2, 2], vertical_alignment="bottom")
    only = c[0].text_input("only these players (optional, comma separated)", placeholder="e.g. theta,gbm")
    fresh = c[1].checkbox("fresh (redo from scratch)")
    check = c[2].checkbox("no-future check")
    extra = (["--only", only] if only else []) + (["--fresh"] if fresh else []) + (["--check"] if check else [])
    with st.container(horizontal=True):
        if st.button("Run steps 1-7", type="primary", icon=":material/play_arrow:"):
            start_run(["1-7", *extra], "steps 1-7")
        for n, f, name in steps:
            if st.button(f"{n:02d} {name}"):
                start_run([str(n), *extra], f"step {n}")
    st.caption("Steps 1-7 = new data up to the judge. Step 8 sends the email (only when something matters).")


@st.fragment(run_every=2)
def live_log():
    info = runs.running()
    with st.container(horizontal=True, vertical_alignment="center"):
        if info:
            st.badge("sleeping until " + info.get("next_run", "?") if info.get("state") == "sleeping" else "running",
                     color="orange", icon=":material/sync:")
            st.caption(f"`main.py {info['what']}` (pid {info['pid']}) since {info['started'].replace('T', ' ')}")
            st.button("Stop", icon=":material/stop:", on_click=runs.stop)
        else:
            st.badge("nothing running", color="gray", icon=":material/pause:")
    lines = runs.tail(40)
    if lines:
        st.code("\n".join(lines), language=None)
    else:
        st.caption("No run yet from the dashboard.")


st.subheader("Live output")
live_log()

log = load("runs", "run_log.csv")
if log is not None:
    st.subheader("Earlier runs")
    st.dataframe(log.iloc[::-1], width="stretch", hide_index=True)
