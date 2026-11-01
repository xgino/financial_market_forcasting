# financial_market_forcasting1/07_judge/data_check.py
"""Is the data fresh, is the macro data there, is a monthly update due?"""
SETTINGS = {"enabled": True, "stale_days": 7}

import pandas as pd

import settings
from core import store
from core.contracts import Check


class DataCheck(Check):
    area = "Data"

    def run(self):
        out, today = [], pd.Timestamp.today().normalize()
        for a in settings.ASSETS:
            p = store.path("prices", f"{a}.csv")
            if not p.exists():
                out.append(("bad", f"{a}: no prices yet, run step 1"))
                continue
            last = store.read_prices(a).index.max()
            if (today - last).days > self.s["stale_days"]:
                out.append(("warn", f"{a}: last price {last.date()}, {(today - last).days} days old. Run the update."))
        if not store.read_macro():
            out.append(("warn", "No macro data (FRED). Your own models are missing interest rates, dollar and VIX."))
        ch = store.path("selection", "choices.csv")
        if ch.exists():
            last = store.read_csv(ch)["date"].max()
            if last.to_period("M") < today.to_period("M"):
                out.append(("warn", f"Monthly update due: the last decision is from {last:%B %Y}. Run: python main.py"))
            else:
                out.append(("good", f"This month's decision is there ({last.date()})."))
        if not any(level != "good" for level, _ in out):
            out.append(("good", "All prices are fresh."))
        return out