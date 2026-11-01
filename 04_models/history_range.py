# financial_market_forcasting1/04_models/history_range.py
"""No bell curve: bad, middle and good case read straight from real past 20 day results."""
SETTINGS = {"enabled": True, "years": 3}
CARD = {  # shown on the dashboard page Models
    "type": 'statistics',
    "trained": "No training. Uses each asset's own last 3 years of 20 day results.",
    "strengths": 'No bell curve assumption: the range comes straight from real past results.',
    "weaknesses": 'Slow: its middle guess barely changes month to month (that also fooled the scores once).',
    "limits": 'Assumes the next 20 days look like the last 3 years.',
    "source": 'Empirical quantiles',
}

import numpy as np

from core.contracts import NAN3, Model


class HistoryRange(Model):
    kind = "stats"
    about = "Range read straight from real past 20 day results (no bell curve)."

    def predict(self, view, asset):
        h = self.horizon
        c = view.closes(asset, int(self.s["years"] * 252) + h).to_numpy()
        if len(c) < 252:
            return NAN3
        past = c[h:] / c[:-h] - 1  # every 20 day result that already finished by today
        return np.quantile(past, [0.1, 0.5, 0.9])

    def explain(self, view, asset):
        h = self.horizon
        c = view.closes(asset, int(self.s["years"] * 252) + h).to_numpy()
        past = c[h:] / c[:-h] - 1
        return [f"Every 20 day result of the last {self.s['years']} years ({len(past)} of them): "
                f"{(past > 0).mean():.0%} went up, the middle one was {np.median(past):+.1%}.",
                f"10% were worse than {np.quantile(past, 0.1):+.1%}, 10% better than {np.quantile(past, 0.9):+.1%}. That's the forecast."]