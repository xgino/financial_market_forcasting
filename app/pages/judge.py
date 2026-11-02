# financial_market_forcasting1/app/pages/judge.py
"""Judge: the plain-language report of step 7. What's good, what needs a look."""
import streamlit as st

from app.common import DATA, load, start_run

st.title("Judge")
if st.button("Run the judge now", icon=":material/gavel:"):
    start_run(["7"], "step 7")
p = DATA / "judge" / "report.md"
if not p.exists():
    st.info("No report yet. Run step 7 (button above, or: python main.py 7).")
    st.stop()
st.markdown(p.read_text())
h = load("judge", "history.csv")
if h is not None and len(h):
    st.subheader("Earlier reports")
    t = h.groupby(["time", "level"]).size().unstack(fill_value=0)
    st.dataframe(t.iloc[::-1], width="stretch")
