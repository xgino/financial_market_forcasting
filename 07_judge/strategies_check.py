# financial_market_forcasting1/07_judge/strategies_check.py
"""Which strategy is the suggestion, does it pass its marks, and how did the holdout go (if opened)?"""
SETTINGS = {"enabled": True}

import pandas as pd

import settings
from core import store
from core.contracts import Check


class StrategiesCheck(Check):
    area = "Strategies"

    def run(self):
        p = store.path("simulation", "kpis.csv")
        if not p.exists():
            return [("bad", "No strategy results yet, run step 6.")]
        k, out = pd.read_csv(p).set_index("strategy"), []
        marks = [c for c in ["beats hold", "drop ok", "sharpe ok"] if c in k]
        k["passes"] = k[marks].apply(lambda r: sum(bool(x) for x in r if pd.notna(x)), axis=1)
        best = k.sort_values(["passes", "sharpe"], ascending=False).index[0]
        r = k.loc[best]
        out.append(("info", f"Suggestion: {best} (follows {r['follows']}): {r['per_year']:+.1%} per year, worst drop "
                            f"{r['max_drop']:.0%}, Sharpe {r['sharpe']:.2f}, fees EUR {r['fees']:,.0f}."))
        if "buy_and_hold" in k.index:
            h = k.loc["buy_and_hold"]
            if r["per_year"] < h["per_year"]:
                out.append(("info", f"It makes less per year than buy & hold ({h['per_year']:+.1%}) but with "
                                    f"{'smaller' if r['max_drop'] > h['max_drop'] else 'similar'} drops: a calmer ride, not more money."))
        if r["follows"] == "base_plan":
            out.append(("info", "The suggestion uses no forecasts for direction, only the plan (and maybe the range)."))
        hp = store.path("simulation", "kpis_holdout.csv")
        if hp.exists():
            hk = pd.read_csv(hp).set_index("strategy")
            if best in hk.index:
                hr = hk.loc[best]
                level = "good" if hr["sharpe"] >= settings.PASS["sharpe"] else "bad"
                out.append((level, f"Holdout (final test) for {best}: {hr['per_year']:+.1%} per year, Sharpe {hr['sharpe']:.2f}, "
                                   f"worst drop {hr['max_drop']:.0%}."))
        else:
            out.append(("info", "The holdout is still closed. Open it once, when everything is final: python main.py 4-6 --holdout"))
        return out