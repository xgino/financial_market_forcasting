# financial_market_forcasting1/04_models/bounce.py
"""Bounce back: after a fall over the last 20 days it expects a rise, after a rise a dip."""
SETTINGS = {"enabled": True, "days": 20, "strength": 0.25, "only_drops": False, "lookback": 252}
CARD = {  # shown on the dashboard page Models
    "type": 'rule',
    "trained": "No training. Uses each asset's own last 20 days.",
    "strengths": 'Short-term reversal is one of the more documented small effects in stock indices.',
    "weaknesses": 'Small effect, easily eaten by costs.',
    "limits": 'Only looks at the last 20 days.',
    "source": 'Short-term reversal: https://en.wikipedia.org/wiki/Mean_reversion_(finance)',
}

import numpy as np

from core.contracts import NAN3, Model, bell_range, log_returns


class Bounce(Model):
    kind = "rule"
    about = "Bounce back: expects part of the last 20 days' move to come back."

    def predict(self, view, asset):
        c = view.closes(asset, self.s["days"] + 1).to_numpy()
        r = log_returns(view, asset, self.s["lookback"])
        if len(c) <= self.s["days"] or len(r) < 30:
            return NAN3
        move = np.log(c[-1] / c[0])
        if self.s["only_drops"]:
            move = min(move, 0.0)
        return bell_range(-self.s["strength"] * move, r.std(ddof=1) * np.sqrt(self.horizon))

    def explain(self, view, asset):
        c = view.closes(asset, self.s["days"] + 1).to_numpy()
        move = np.log(c[-1] / c[0])
        return [f"Over the last {self.s['days']} days it moved {np.expm1(move):+.1%}.",
                f"It expects {self.s['strength']:.0%} of that to come back: {np.expm1(-self.s['strength'] * (min(move, 0) if self.s['only_drops'] else move)):+.2%}."]