# financial_market_forcasting1/07_judge/live_check.py
"""The live record: how many months, how it did against the base plan, and the rule for real money."""
SETTINGS = {"enabled": True, "base": "fixed_mix"}

import pandas as pd

import settings
from core import store
from core.contracts import Check


def realised(log, strategy):
    """Month return of what a strategy said live: shares x what each asset did until the next decision."""
    s = log[log["strategy"] == strategy]
    months = sorted(s["month"].unique())
    out = {}
    for m, nxt in zip(months, months[1:]):
        g = s[s["month"] == m]
        start, end = pd.Timestamp(g["decided"].iloc[0]), pd.Timestamp(log.loc[log["month"] == nxt, "decided"].iloc[0])
        r = 0.0
        for _, row in g.iterrows():
            px = store.read_prices(row["asset"])["close_eur"]
            a, b = px[:start], px[:end]
            if len(a) and len(b):
                r += row["share"] * (b.iloc[-1] / a.iloc[-1] - 1)
        out[m] = r
    return pd.Series(out, dtype=float)


class LiveCheck(Check):
    area = "Live record"

    def run(self):
        p = store.path("live", "decisions_log.csv")
        if not p.exists():
            return [("info", "No live record yet. It starts the first time step 6 runs in a month.")]
        log = pd.read_csv(p)
        months = log["month"].nunique()
        out = [("info", f"{months} live month(s) recorded, from {log['month'].min()} to {log['month'].max()}.")]
        need = settings.REAL_MONEY_RULE_MONTHS
        k = store.path("simulation", "kpis.csv")
        best = None
        if k.exists():
            kk = pd.read_csv(k)
            best = kk.sort_values("sharpe", ascending=False)["strategy"].iloc[0]
        if best and months >= 2:
            mine, base = realised(log, best), realised(log, self.s["base"])
            if len(mine):
                out.append(("info", f"Live so far, {best}: {mine.add(1).prod() - 1:+.1%} vs base plan "
                                    f"({self.s['base']}) {base.add(1).prod() - 1:+.1%} over {len(mine)} finished month(s)."))
        if months < need:
            out.append(("info", f"Rule for real money: at least {need} live months that beat the base plan. "
                                f"{need - months} to go."))
        else:
            out.append(("warn", f"{months} live months recorded. Only consider real money if the live record beat the "
                                "base plan, and start small."))
        return out