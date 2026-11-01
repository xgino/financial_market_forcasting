# paper-fund-v2/04_models/naive.py
"""No change: the middle guess is always 0% (today's price). The range comes from last year's swings
on a bell curve. It knows nothing about direction on purpose: every other model has to beat it."""
SETTINGS = {"enabled": True, "lookback": 252}
CARD = {  # shown on the dashboard page Models
    "type": 'baseline (rule)',
    "trained": "No training. Uses each asset's own last year of prices.",
    "strengths": 'The honest floor: it never pretends to know the direction. Its range is well calibrated.',
    "weaknesses": 'No opinion at all, by design.',
    "limits": 'Only useful as the bar every other model has to beat.',
    "source": 'Textbook naive forecast: https://otexts.com/fpp3/simple-methods.html',
}

import numpy as np

from core.contracts import NAN3, Model, bell_range, log_returns


class Naive(Model):
    kind = "rule"
    about = "No change: middle guess always 0%, range from last year's swings. The floor."

    def predict(self, view, asset):
        r = log_returns(view, asset, self.s["lookback"])
        if len(r) < 30:
            return NAN3
        return bell_range(0.0, r.std(ddof=1) * np.sqrt(self.horizon))

    def explain(self, view, asset):
        r = log_returns(view, asset, self.s["lookback"])
        sd = r.std(ddof=1)
        return ["Middle guess: no change, today's price. It never guesses a direction on purpose.",
                f"Daily swings over the last year: {sd:.2%}, about {sd * np.sqrt(self.horizon):.1%} over 20 trading days.",
                "Bad and good case = about 1.28 times that below and above today's price."]