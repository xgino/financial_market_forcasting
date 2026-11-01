# financial_market_forcasting1/06_strategies/risk_balanced.py
"""Uses the part models ARE good at: the range. A wild asset (wide bad..good range) gets less money,
a calm one more, so each asset brings about the same risk. Same total invested as fixed_mix."""
SETTINGS = {"enabled": True, "follow": "base_plan", "invested": 0.8}

import numpy as np

from core.simulate import Strategy


class RiskBalanced(Strategy):
    about = "less money in wild assets, more in calm ones (sized by the range)"

    def targets(self, fund, picks):
        width = (picks["p90"] - picks["p10"]).clip(lower=0.01)
        inv = (1 / width).replace([np.inf], np.nan).dropna()
        w = self.s["invested"] * inv / inv.sum()
        return w.to_dict(), {a: f"range {width[a]:.0%} wide: {w[a]:.0%}" for a in w.index}