# financial_market_forcasting1/06_strategies/follow_bandit.py
"""Same tilt as follow_best, but following the reward-and-punish selector (bandit)."""
SETTINGS = {"enabled": True, "follow": "bandit", "base": 0.6, "tilt": 0.3, "min_edge": 0.005}

from core.simulate import Strategy, tilt


class FollowBandit(Strategy):
    about = "base 60% plus or minus 30% following the bandit"

    def targets(self, fund, picks):
        return tilt(picks, self.s["base"], self.s["tilt"], self.s["min_edge"])