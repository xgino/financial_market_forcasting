# paper-fund-v2/05_selection/best_overall.py
"""Follows the model the skill book trusts most over ALL situations together (only results known today).
Per situation the results get split into 6 boxes, which makes trust slow to earn. Overall, a model with a
steady small edge (like gbm) can earn trust sooner. No model trusted overall = base plan."""
SETTINGS = {"enabled": True}

from core.selection import Selector, mix, plain, trusted


class BestOverall(Selector):
    about = "the most trusted model over all situations together, else base plan"

    def choose(self, ctx):
        t = trusted(ctx, ctx.trust_all)
        if t.empty:
            return plain(ctx, "no model trusted overall yet")
        m = t.index[0]
        return mix(ctx, {m: 1.0}, f"most trusted overall: IC {t.at[m, 'ic']:+.2f}, trust {t.at[m, 'trust']:.1f}")