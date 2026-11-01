# paper-fund-v2/05_selection/top_n_blend.py
"""A mix of the top N trusted models for today's situation (settings.TOP_N), weighted by their score.
No model trusted yet = base plan."""
SETTINGS = {"enabled": True}

import settings
from core.selection import Selector, mix, plain, trusted


class TopNBlend(Selector):
    about = "mix of the top N trusted models for today's situation"

    def choose(self, ctx):
        t = trusted(ctx).head(settings.TOP_N)
        if t.empty:
            return plain(ctx, f"no model trusted yet in {ctx.situation}")
        return mix(ctx, {m: float(t.at[m, "ic"]) for m in t.index}, f"mix of {len(t)} trusted in {ctx.situation}")