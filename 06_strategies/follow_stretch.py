# financial_market_forcasting1/06_strategies/follow_stretch.py
"""follow_best, but following best_in_stretch: the model trusted most in today's stretch (pattern chunk).
Same base and tilt as follow_best and follow_similar, so the three compare fairly."""
SETTINGS = {"enabled": True, "follow": "best_in_stretch", "base": 0.6, "tilt": 0.3, "min_edge": 0.005}

from core.simulate import Strategy, tilt


class FollowStretch(Strategy):
    about = "base 60% plus or minus 30% when the model trusted in today's stretch expects a clear move"

    def targets(self, fund, picks):
        return tilt(picks, self.s["base"], self.s["tilt"], self.s["min_edge"])
