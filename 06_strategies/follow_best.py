# financial_market_forcasting1/06_strategies/follow_best.py
"""Base plan plus a tilt: more in an asset when the most trusted model for its situation expects a rise
clearly bigger than the fees, less when it expects a fall. No trusted model = just the base."""
SETTINGS = {"enabled": True, "follow": "best_in_situation", "base": 0.6, "tilt": 0.3, "min_edge": 0.005}

from core.simulate import Strategy, tilt


class FollowBest(Strategy):
    about = "base 60% plus or minus 30% when a trusted model expects a clear move"

    def targets(self, fund, picks):
        return tilt(picks, self.s["base"], self.s["tilt"], self.s["min_edge"])