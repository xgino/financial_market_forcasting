# financial_market_forcasting1/dashboard.py
"""
The dashboard. Start it with:   venv/bin/streamlit run dashboard.py   (or streamlit run dashboard.py after
source venv/bin/activate; any other Python is stopped, see core/env.py)

Every page is its own file in app/pages/ (shared helpers: app/common.py), so a page is easy to change:
  Decide            Overview     KPIs: the signal per asset, the best strategy, which models are proven
                    Forecasts    price history + every past guess of a model + its forecast ahead, 5/20/60 days
                    This month   the decision for one asset in detail: why, votes, other horizons, strategies
                    Calculator   try amounts, models and your own numbers (only reads)
  How good is it    Results      every strategy month by month: fund value, green/red months, KPIs with pass marks
                    Leaderboard  the ranking per horizon, the warning record, whose results count for trust
                    Battle       month by month replay: every model's guess, what really happened, the winner
                    Models       what every model is, what it was trained on, strengths, weaknesses, its scores
                    Judge        the plain-language report: what's good, what needs a look
  Run and data      Run          start steps in the background, follow the live log, earlier runs
                    Training     live view of a training run
                    Data         the prices and whether they're fresh
                    Email        the monthly email: settings, what was sent, a preview
The sidebar shows whether a run is going and has Update now / Keep updating every day / Stop.
Everything it shows comes from the CSV files in data/, so closing it loses nothing (a run keeps going).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.env import wrong_python  # noqa: E402

st.set_page_config(page_title="financial_market_forcasting1", page_icon=":material/monitoring:", layout="wide")
if wrong_python():   # started from Anaconda or another Python: stop, its buttons would run steps with it too
    st.error("Start the dashboard from the venv in this folder.", title="Wrong Python")
    st.code(wrong_python())
    st.stop()

import settings  # noqa: E402
from app.common import run_status  # noqa: E402

PAGES = Path(__file__).resolve().parent / "app" / "pages"
page = lambda f, title, icon, **kw: st.Page(PAGES / f, title=title, icon=f":material/{icon}:", **kw)
nav = st.navigation({
    "Decide": [page("overview.py", "Overview", "dashboard", default=True), page("forecasts.py", "Forecasts", "show_chart"),
               page("this_month.py", "This month", "event"), page("calculator.py", "Calculator", "calculate")],
    "How good is it": [page("results.py", "Results", "savings"), page("leaderboard.py", "Leaderboard", "leaderboard"),
                       page("battle.py", "Battle", "swords"), page("models.py", "Models", "smart_toy"),
                       page("judge.py", "Judge", "gavel")],
    "Run and data": [page("run.py", "Run", "play_circle"), page("training.py", "Training", "model_training"),
                     page("data.py", "Data", "database"), page("email.py", "Email", "mail")],
})
with st.sidebar:
    run_status()
    st.caption("Holdout " + ("SHOWN" if settings.SHOW_HOLDOUT else "hidden") + f" (from {settings.HOLDOUT_FROM})")
nav.run()
