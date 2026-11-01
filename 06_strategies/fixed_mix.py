# paper-fund-v2/06_strategies/fixed_mix.py
"""The base plan with money: always the same share invested (spread evenly), the rest cash, put back in
balance once a month. No forecasts used at all."""
SETTINGS = {"enabled": True, "follow": "base_plan", "invested": 0.8}

from core.simulate import Strategy


class FixedMix(Strategy):
    about = "always 80% invested, spread evenly, rebalanced monthly (no forecasts)"

    def targets(self, fund, picks):
        w = self.s["invested"] / len(picks)
        return {a: w for a in picks.index}, {a: f"keep {w:.0%}" for a in picks.index}