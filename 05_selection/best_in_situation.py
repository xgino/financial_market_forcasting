# paper-fund-v2/05_selection/best_in_situation.py
"""Follows the model the skill book trusts most for TODAY'S situation (only results known today count).
No model trusted yet = base plan. Trusted = settings.MIN_TRUST with settings.MIN_PERIODS."""
SETTINGS = {"enabled": True}

from core.selection import Selector, mix, plain, trusted


class BestInSituation(Selector):
    about = "the most trusted model for today's situation, else base plan"

    def choose(self, ctx):
        t = trusted(ctx)
        if t.empty:
            return plain(ctx, f"no model trusted yet in {ctx.situation}")
        m = t.index[0]
        return mix(ctx, {m: 1.0}, f"most trusted in {ctx.situation}: IC {t.at[m, 'ic']:+.2f}, trust {t.at[m, 'trust']:.1f}")