# financial_market_forcasting1/06_strategies/_step.py
"""
Step 6, what to do with the money. Every strategy gets EUR 1000 and runs month by month on what its
selector chose (only what was known each month), trades at the next open with fees.
Saves in data/simulation/:
  kpis.csv               every strategy's KPIs + pass marks (development period)
  <strategy>_months.csv  month by month: fund value, shares per asset, why
  <strategy>_trades.csv  every buy and sell
  now.csv                what each strategy would do TODAY with a fresh EUR 1000
"""
from pathlib import Path

import pandas as pd

import settings
from core import simulate, store
from core.plugins import load_players

STEP = "Strategies"
FOLDER = Path(__file__).resolve().parent


def passes(k, hold):
    p = settings.PASS
    return {"beats hold": k["sharpe"] > hold["sharpe"] + p["beat_hold"] if hold and k is not hold else None,
            "drop ok": k["max_drop"] >= p["max_drop"], "sharpe ok": k["sharpe"] >= p["sharpe"]}


def log_live(now, decided):
    """data/live/decisions_log.csv: what every strategy said, the first time it saw each month. Only ever
    added to, never rewritten: your real track record, even if models or settings change later."""
    from datetime import datetime

    p = store.path("live", "decisions_log.csv")
    old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["month"])
    month = str(decided.to_period("M"))
    if now.empty or month in set(old["month"].astype(str)):
        return
    new = now.assign(month=month, decided=str(decided.date()), logged_at=datetime.now().isoformat(timespec="seconds"))
    pd.concat([old, new], ignore_index=True).to_csv(p, index=False)
    print(f"  live record: {month} added to data/live/decisions_log.csv")


def run(args):
    prices = {a: store.read_prices(a) for a in settings.ASSETS}
    choices = store.read_csv(store.path("selection", "choices.csv"), dates=("date", "end_date"))
    strategies = load_players(FOLDER, simulate.Strategy, only=args.only)
    start = pd.Timestamp(settings.WARMUP_UNTIL) + pd.DateOffset(days=1)
    end = None if settings.SHOW_HOLDOUT else settings.HOLDOUT_FROM
    rows, latest = {}, choices["date"].max()
    now = []
    for name, s in strategies.items():
        months, trades = simulate.simulate(s, choices, prices, start, end)
        store.write_csv(months, store.path("simulation", store.tag(f"{name}_months") + ".csv"), index=False)
        store.write_csv(trades, store.path("simulation", store.tag(f"{name}_trades") + ".csv"), index=False)
        rows[name] = {"follows": s.s["follow"], **simulate.kpis(months)}
        # today: a fresh strategy with a fresh EUR 1000, on this month's picks
        fresh = type(s)(s.s)
        picks = choices[(choices["selector"] == s.s["follow"]) & (choices["date"].dt.to_period("M") == latest.to_period("M"))]
        if len(picks):
            target, why = fresh.targets(simulate.Fund(latest, settings.START_EUR, settings.START_EUR, {},
                                                      settings.START_EUR, settings.START_EUR), picks.set_index("asset"))
            for a, w in target.items():
                now.append({"strategy": name, "asset": a, "share": w, "eur": w * settings.START_EUR, "why": why.get(a, "")})
    k = pd.DataFrame(rows).T
    hold = rows.get("buy_and_hold")
    for name in k.index:
        for col, ok in passes(rows[name], hold).items():
            k.loc[name, col] = ok
    k.index.name = "strategy"
    store.write_csv(k, store.path("simulation", store.tag("kpis") + ".csv"))
    if not settings.SHOW_HOLDOUT:   # today's decision and the live record only from normal runs
        store.write_csv(pd.DataFrame(now), store.path("simulation", "now.csv"), index=False)
        log_live(pd.DataFrame(now), latest)

    show = pd.DataFrame(index=k.index)
    show["follows"] = k["follows"]
    show["end EUR"] = k["end_eur"].map(lambda x: f"{x:,.0f}")
    for c in ["total", "per_year", "max_drop", "worst_month"]:
        show[c] = k[c].map(lambda x: f"{x:+.1%}")
    show["sharpe"] = k["sharpe"].map(lambda x: f"{x:.2f}")
    show["months up"] = k["months_up"].map(lambda x: f"{x:.0%}")
    show["fees"] = k["fees"].map(lambda x: f"{x:,.0f}")
    show["invested"] = k["avg_invested"].map(lambda x: f"{x:.0%}")
    for c in ["beats hold", "drop ok", "sharpe ok"]:
        show[c] = k[c].map(lambda x: "" if x is None or pd.isna(x) else ("yes" if x else "NO"))
    shown = "INCLUDES HOLDOUT" if settings.SHOW_HOLDOUT else "development period, holdout hidden"
    print(f"\nEUR {settings.START_EUR} each, {shown}:")
    print(show.to_string())
    print(f"\nPass marks (settings.PASS): Sharpe above buy & hold, worst drop above {settings.PASS['max_drop']:.0%}, "
          f"Sharpe at least {settings.PASS['sharpe']}.")
    return {"strategies": len(strategies)}