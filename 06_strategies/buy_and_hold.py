# paper-fund-v2/06_strategies/buy_and_hold.py
"""Buy every asset in equal parts on day one and never touch it again. The benchmark."""
SETTINGS = {"enabled": True, "follow": "base_plan"}

from core.simulate import Strategy


class BuyAndHold(Strategy):
    about = "equal parts on day one, never trade again (the benchmark)"

    def targets(self, fund, picks):
        if getattr(self, "_done", False):
            return {}, {}
        self._done = True
        return {a: 1 / len(picks) for a in picks.index}, {a: "buy once, hold" for a in picks.index}