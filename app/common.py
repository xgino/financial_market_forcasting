# financial_market_forcasting1/app/common.py
"""
What every dashboard page shares: reading the CSV files in data/, the signal in plain words, the run
status in the sidebar. The pages themselves are in app/pages/, one file each: change a page there.
"""
import numpy as np
import pandas as pd
import streamlit as st

import settings
from core import runs
from core.verdict import firsts, history_summary, models_on, verdict

DATA = settings.DATA
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#9D755D", "#BAB0AC"]
WIN, LOSS = "#2e9e5b", "#d64545"
COST = 2 * (settings.FEE_RATE + settings.SPREAD)            # buy + sell, as a share of the amount
HORIZON_LABEL = {5: "week (5 days)", 20: "month (20 days)", 60: "quarter (60 days)"}
CALL_COLOR = {"BUY": "green", "WAIT": "red", "KEEP YOUR PLAN": "gray"}
JOURNAL_COLS = ["date", "end_date", "asset", "model", "kind", "situation", "stretch", "price", "p10", "p50", "p90",
                "actual", "seen"]


# ---------- reading data/ ----------

@st.cache_resource(max_entries=40)
def _csv(path, mtime, usecols=None):
    """Read once per version of the file (mtime). Shared between reruns: never change what it returns."""
    df = pd.read_csv(path, usecols=(lambda c: c in usecols) if usecols else None)   # missing columns are fine
    for c in ("date", "end_date", "as_of"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="ISO8601")
    return df


def load(*parts, usecols=None):
    """data/<parts>, or None if that step hasn't made it yet."""
    p = DATA.joinpath(*parts)
    if not p.exists():
        return None
    return _csv(p, p.stat().st_mtime, tuple(usecols) if usecols else None)


def suffix(h):
    return "" if h == settings.HORIZON_DAYS else f"_h{h}"


def journal(h=settings.HORIZON_DAYS):
    return load("battle", f"journal{suffix(h)}.csv", usecols=JOURNAL_COLS)


def skillbook(h=settings.HORIZON_DAYS):
    return load("battle", f"skillbook{suffix(h)}.csv")


def horizons_ready():
    """The horizons step 4 has battled (journal and skill book both there)."""
    return [h for h in settings.HORIZONS
            if (DATA / "battle" / f"journal{suffix(h)}.csv").exists() and (DATA / "battle" / f"skillbook{suffix(h)}.csv").exists()]


def need(df, step):
    if df is None:
        st.info(f"Nothing here yet. Run step {step} first (sidebar: Update now, or: python main.py {step}).",
                icon=":material/hourglass_empty:")
        st.stop()


def mtime(*parts):
    p = DATA.joinpath(*parts)
    return p.stat().st_mtime if p.exists() else 0


def ok(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else ("✅" if bool(v) else "❌")


def freshness():
    """Last price date per traded asset and how old it is."""
    today = pd.Timestamp.today().normalize()
    rows = []
    for a, spec in settings.ASSETS.items():
        px = load("prices", f"{a}.csv", usecols=["date"])
        if px is not None:
            last = px["date"].max()
            rows.append({"asset": a, "last price": last.date(), "days old": (today - last).days,
                         "ok": "✅" if (today - last).days <= 4 else "⚠️",
                         "source": f"https://finance.yahoo.com/quote/{spec['ticker']}"})
    return pd.DataFrame(rows)


def pick_strategies(k):
    """Best first: passes the most pass marks, then the best Sharpe."""
    t = k.copy()
    marks = [c for c in ["beats hold", "drop ok", "sharpe ok"] if c in t]
    t["passes"] = t[marks].apply(lambda r: sum(bool(x) for x in r if pd.notna(x) and x != ""), axis=1)
    return t.sort_values(["passes", "sharpe"], ascending=False)


@st.cache_resource
def live_models():
    """The model files and the timeline, loaded once, for the explanations."""
    from core.contracts import Model
    from core.plugins import load_players
    from core.timeline import Timeline

    return load_players(settings.ROOT / "04_models", Model, settings.HORIZON_DAYS), Timeline.load()


# ---------- the signal, in plain words ----------

def signal(v, amount, h=settings.HORIZON_DAYS):
    """What a verdict means for your money. The bad .. good range alone says little (it's mostly the normal
    swing of the market), so this compares the models' guess with that swing and with the costs.
    -> dict: label, color, guess, after_costs, swing, ratio, sentence (all shares of the amount)."""
    p10, p50, p90 = v["p10"], v["p50"], v["p90"]
    if pd.isna(p10) or pd.isna(p90):
        return {"label": "No forecast", "color": "gray", "guess": np.nan, "after_costs": np.nan, "swing": np.nan,
                "ratio": np.nan, "sentence": "No model has a forecast for this day yet."}
    swing = max((p90 - p10) / 2.563, 1e-9)                        # p10..p90 of a bell curve = 2.56 swings wide
    ratio = p50 / swing
    if v["call"] == "BUY":
        label = "Strong buy signal" if v["confidence"] >= 60 else "Buy signal"
    elif v["call"] == "WAIT":
        label = "Strong wait signal" if v["confidence"] >= 60 else "Wait signal"
    else:
        label = "No signal"
    size = "much smaller than" if abs(ratio) < 0.25 else ("smaller than" if abs(ratio) < 0.5 else "about as big as")
    sentence = (f"The models' best guess is **{p50:+.1%}** in {h} trading days (EUR {amount * p50:+,.0f} on EUR {amount:,.0f}), "
                f"EUR {amount * (p50 - COST):+,.0f} after buy and sell costs. The market's normal swing over that time is "
                f"**±{swing:.1%}** (EUR ±{amount * swing:,.0f}), so the guess is {size} the noise.")
    if v["call"] == "KEEP YOUR PLAN":
        sentence += (" No model with a proven record backs a move, so timing doesn't matter this time: follow your plan."
                     if not v["who"] else "")
    return {"label": label, "color": CALL_COLOR[v["call"]], "guess": p50, "after_costs": p50 - COST, "swing": swing,
            "ratio": ratio, "sentence": sentence}


@st.cache_data(max_entries=8)
def verdicts(h, _jm, _bm):
    """This decision day's verdict for every asset on horizon h. -> {asset: (date, situation, verdict, models table)}"""
    j, book = journal(h), skillbook(h)
    out = {}
    for a in sorted(j["asset"].unique()):
        d = firsts(j, a)[-1] if h == settings.HORIZON_DAYS else j.loc[j["asset"] == a, "date"].max()
        mt, situation = models_on(j, book, a, d)
        out[a] = (d, situation, verdict(mt, h), mt)
    return out


def all_verdicts(h=settings.HORIZON_DAYS):
    s = suffix(h)
    return verdicts(h, mtime("battle", f"journal{s}.csv"), mtime("battle", f"skillbook{s}.csv"))


def past_calls(asset=None):
    """The same monthly rule on every past month: what happened after each call (step 5, only what was known then)."""
    h = load("selection", "past_calls.csv")
    if h is None or h.empty:
        return pd.DataFrame()
    return history_summary(h[h["asset"] == asset] if asset else h)


def proven(board):
    """Models whose leaderboard verdict is an edge (passes trust and the luck test on the development years)."""
    if board is None or "verdict" not in board:
        return []
    return board.loc[board["verdict"].astype(str).str.startswith("edge"), "model"].tolist()


# ---------- runs (sidebar) ----------

def start_run(args, label):
    pid = runs.start(args)
    if pid:
        st.session_state["was_running"] = True
        st.toast(f"Started {label} in the background. It keeps going if you close this page.", icon=":material/play_arrow:")
    else:
        st.toast("A run is going already: wait for it, or stop it in the sidebar.", icon=":material/block:")


@st.fragment(run_every=3)
def run_status():
    """Sidebar: is a run going, what it's doing, start / stop. Refreshes by itself; when a run ends the page reloads
    with the new results."""
    info = runs.running()
    if info:
        st.session_state["was_running"] = True
        sleeping = info.get("state") == "sleeping"
        st.badge(f"sleeping until {info.get('next_run', '?')[11:]}" if sleeping else "running",
                 icon=":material/bedtime:" if sleeping else ":material/sync:", color="blue" if sleeping else "orange")
        st.caption(f"`main.py {info['what']}` since {info['started'][5:16].replace('T', ' ')}")
        last = runs.tail(1)
        if last and not sleeping:
            st.caption(last[-1][-90:])
        st.button("Stop", icon=":material/stop:", on_click=runs.stop, width="stretch")
        return
    if st.session_state.pop("was_running", False):
        st.cache_data.clear()
        st.rerun(scope="app")                                  # the run ended: show its results
    log = load("runs", "run_log.csv")
    if log is not None and len(log):
        last = log.iloc[-1]
        good = str(last["status"]) == "ok"
        st.badge("last run ok" if good else f"last run: {str(last['status'])[:40]}",
                 icon=":material/check_circle:" if good else ":material/error:", color="green" if good else "red")
        st.caption(f"{last['time'][5:16].replace('T', ' ')}, step {last['step']}")
    if st.button("Update now", icon=":material/refresh:", type="primary", width="stretch",
                 help="New prices, then steps 1 to 7. Only what's new is computed."):
        start_run(["1-7"], "steps 1-7")
        st.rerun(scope="fragment")
    if st.button("Keep updating every day", icon=":material/autorenew:", width="stretch",
                 help="Runs steps 1-7 now and again every 24 hours, until you press Stop."):
        start_run(["1-7", "--every", "24"], "steps 1-7 every 24 hours")
        st.rerun(scope="fragment")

