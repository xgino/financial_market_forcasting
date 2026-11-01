# paper-fund-v2/06_strategies/cppi_floor.py
"""Your "house money" idea, real name CPPI: protect a floor and only risk a multiple of what is above it.
floor = floor_share x the highest the fund ever was (ratchet): profits get partly locked in as it grows.
At the floor it sits fully in cash. The risky part is spread like follow_best (tilt by trusted models)."""
SETTINGS = {"enabled": True, "follow": "best_in_situation", "floor_share": 0.85, "multiplier": 3,
            "max_invested": 1.0, "base": 0.6, "tilt": 0.3, "min_edge": 0.005}

from core.simulate import Strategy, tilt


class CPPIFloor(Strategy):
    about = "protect 85% of the best level so far, risk 3x what is above it"

    def targets(self, fund, picks):
        floor = self.s["floor_share"] * fund.peak
        risky = min(self.s["multiplier"] * max(fund.equity - floor, 0), self.s["max_invested"] * fund.equity)
        share = risky / fund.equity if fund.equity else 0
        w, why = tilt(picks, self.s["base"], self.s["tilt"], self.s["min_edge"])
        total = sum(w.values()) or 1
        return ({a: share * v / total for a, v in w.items()},
                {a: f"floor EUR {floor:,.0f}, risk {share:.0%} of fund | {why[a]}" for a in w})