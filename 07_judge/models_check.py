# paper-fund-v2/07_judge/models_check.py
"""Which models pass every pass mark, and are there red flags (too good to be true, ranges that are off)?"""
SETTINGS = {"enabled": True}

import pandas as pd

import settings
from core import store
from core.contracts import Check


class ModelsCheck(Check):
    area = "Models"

    def run(self):
        p = store.path("battle", "leaderboard.csv")
        if not p.exists():
            return [("bad", "No leaderboard yet, run step 4.")]
        b, P, out = pd.read_csv(p), settings.PASS, []
        yrs = b["years_positive"].astype(str).str.extract(r"(\d+) of (\d+)").astype(float)
        passed = b[(b["trust"] >= P["trust"]) & (b["luck_beats_it"] < P["luck"]) & (yrs[0] / yrs[1] >= P["years"])
                   & (b["skill"] > P["skill"]) & b["range_hit"].between(*P["range"])]
        if len(passed):
            out.append(("good", f"Passes every pass mark: {', '.join(passed['model'])}. Confirm it in the holdout before trusting it."))
        else:
            best = b.dropna(subset=["ic"]).head(1)
            hint = f" Closest: {best['model'].iloc[0]} (IC {best['ic'].iloc[0]:+.3f}, trust {best['trust'].iloc[0]:+.1f})." if len(best) else ""
            out.append(("info", "No model passes every pass mark yet, so the strategies follow the base plan. "
                                "That's the honest answer, not a failure." + hint))
        for _, r in b.iterrows():
            if pd.notna(r["ic"]) and r["ic"] > 0.25 and r["trust"] >= P["trust"]:
                out.append(("bad", f"{r['model']}: IC {r['ic']:+.2f} is too good to be true for markets. Check for memory or a leak."))
            if pd.notna(r["range_hit"]) and not P["range"][0] <= r["range_hit"] <= P["range"][1]:
                way = "too narrow (too sure of itself)" if r["range_hit"] < P["range"][0] else "too wide"
                out.append(("warn", f"{r['model']}: range hit {r['range_hit']:.0%}, {way}."))
        return out