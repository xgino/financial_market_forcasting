# paper-fund-v2/06_strategies/follow_overall.py
"""Base 60% plus or minus a 30% tilt, following the model trusted most over all situations (best_overall).
This is the money test for a model like gbm: does following it actually beat the base plan after costs?"""
SETTINGS = {"enabled": True, "follow": "best_overall", "base": 0.6, "tilt": 0.3, "min_edge": 0.005}

from core.simulate import Strategy, tilt


class FollowOverall(Strategy):
    about = "base 60% plus or minus 30% following the most trusted model overall"

    def targets(self, fund, picks):
        return tilt(picks, self.s["base"], self.s["tilt"], self.s["min_edge"])