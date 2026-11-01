# financial_market_forcasting1/06_strategies/follow_similar.py
"""follow_best, but following best_in_similar_days: the model that did best on the past days most like
today, instead of the best in today's situation box. Same base and tilt, so the two compare fairly."""
SETTINGS = {"enabled": True, "follow": "best_in_similar_days", "base": 0.6, "tilt": 0.3, "min_edge": 0.005}

from core.simulate import Strategy, tilt


class FollowSimilar(Strategy):
    about = "base 60% plus or minus 30% when the model trusted on similar days expects a clear move"

    def targets(self, fund, picks):
        return tilt(picks, self.s["base"], self.s["tilt"], self.s["min_edge"])
