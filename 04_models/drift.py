# paper-fund-v2/04_models/drift.py
"""Momentum: last year's average daily move just continues. Textbook name: random walk with drift."""
SETTINGS = {"enabled": True, "lookback": 252}
CARD = {  # shown on the dashboard page Models
    "type": 'rule',
    "trained": "No training. Uses each asset's own last year of prices.",
    "strengths": 'Simple momentum: catches long steady trends.',
    "weaknesses": 'Late at turning points, keeps betting on a trend that just ended.',
    "limits": "Direction only from last year's average move.",
    "source": 'Random walk with drift: https://otexts.com/fpp3/simple-methods.html',
}

import numpy as np

from core.contracts import NAN3, Model, bell_range, log_returns


class Drift(Model):
    kind = "rule"
    about = "Momentum: last year's average daily move just continues."

    def predict(self, view, asset):
        r = log_returns(view, asset, self.s["lookback"])
        if len(r) < 30:
            return NAN3
        return bell_range(r.mean() * self.horizon, r.std(ddof=1) * np.sqrt(self.horizon))

    def explain(self, view, asset):
        r = log_returns(view, asset, self.s["lookback"])
        return [f"Last year it moved {r.mean():+.3%} a day on average ({np.expm1(r.sum()):+.1%} over the year).",
                f"It assumes that just continues: {np.expm1(r.mean() * self.horizon):+.2%} over the next 20 trading days."]