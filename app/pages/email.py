# financial_market_forcasting1/app/pages/email.py
"""Email: the monthly email (step 8). Its settings, what was sent, a preview."""
import streamlit as st

import settings
from app.common import DATA, load, start_run

st.title("Email")
cfg = settings.NOTIFY
st.write(f"Mails about **{', '.join(cfg['assets'])}**, only when the call is **{', '.join(cfg['only_when'])}** "
         f"with confidence ≥ {cfg['min_confidence']}. Change it in settings.py (NOTIFY).")
has_secrets = (settings.ROOT / "secrets.env").exists()
st.write("Mail account: " + ("✅ secrets.env found" if has_secrets else "❌ no secrets.env yet: copy secrets.example.env and fill it in"))
if st.button("Make this month's email now (sends only if it matters)", icon=":material/mail:"):
    start_run(["8"], "step 8")
sent = load("notify", "sent.csv")
if sent is not None and len(sent):
    st.subheader("Sent")
    st.dataframe(sent.iloc[::-1], width="stretch", hide_index=True)
prev = DATA / "notify" / "last_email.html"
if prev.exists():
    st.subheader("Preview of this month's email")
    st.iframe(prev, height=700)
