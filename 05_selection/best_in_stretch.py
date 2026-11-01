# financial_market_forcasting1/05_selection/best_in_stretch.py
"""Follows the model the skill book trusts most for TODAY'S STRETCH (core/stretches.py): the pattern chunk
the price line is in, cut where its behaviour changed, instead of fixed 120/20 day windows. Same bar as
best_in_situation (settings.MIN_TRUST with settings.MIN_PERIODS). No model trusted = base plan."""
SETTINGS = {"enabled": True}

from core.selection import Selector, mix, plain, trusted


class BestInStretch(Selector):
    about = "the most trusted model for today's stretch (pattern chunk), else base plan"

    def choose(self, ctx):
        if ctx.stretch is None or ctx.by_label is None:
            return plain(ctx, "no stretch label yet")
        t = trusted(ctx, ctx.by_label(f"stretch:{ctx.stretch}"))
        if t.empty:
            return plain(ctx, f"no model trusted in stretch {ctx.stretch}")
        m = t.index[0]
        return mix(ctx, {m: 1.0}, f"most trusted in stretch {ctx.stretch}: IC {t.at[m, 'ic']:+.2f}, trust {t.at[m, 'trust']:.1f}")
