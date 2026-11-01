# financial_market_forcasting1/04_models/mean_reversion.py
"""Pulls back to the 200 day average: far above it expects a drop, far below it expects a rise."""
SETTINGS = {"enabled": True, "avg_days": 200, "speed": 0.1, "lookback": 252}
CARD = {  # shown on the dashboard page Models
    "type": 'rule',
    "trained": "No training. Uses each asset's own last 200 days.",
    "strengths": 'Expects stretched prices to come back to normal, which happens often after overreactions.',
    "weaknesses": 'Fights strong trends and loses there.',
    "limits": 'One fixed speed (10% of the gap) for every asset.',
    "source": 'Mean reversion: https://en.wikipedia.org/wiki/Mean_reversion_(finance)',
}

import numpy as np

from core.contracts import NAN3, Model, bell_range, log_returns


class MeanReversion(Model):
    kind = "rule"
    about = "Pulls back to the 200 day average."

    def predict(self, view, asset):
        c = view.closes(asset, self.s["avg_days"]).to_numpy()
        r = log_returns(view, asset, self.s["lookback"])
        if len(c) < self.s["avg_days"] or len(r) < 30:
            return NAN3
        gap = np.log(c[-1] / c.mean())  # how far above (+) or below (-) the average
        return bell_range(-self.s["speed"] * gap, r.std(ddof=1) * np.sqrt(self.horizon))

    def explain(self, view, asset):
        c = view.closes(asset, self.s["avg_days"]).to_numpy()
        gap = np.log(c[-1] / c.mean())
        side = "above" if gap > 0 else "below"
        return [f"The price is {abs(np.expm1(gap)):.1%} {side} its {self.s['avg_days']} day average.",
                f"It expects {self.s['speed']:.0%} of that gap to close in 20 trading days: {np.expm1(-self.s['speed'] * gap):+.2%}."]