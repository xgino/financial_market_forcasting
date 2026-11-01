# financial_market_forcasting1/05_selection/best_in_similar_days.py
"""Follows the model that did best on the past days most like today (core/similar.py), instead of on
today's situation box. Only days whose result was known today count. Needs: trust >= settings.MIN_TRUST
over >= settings.MIN_PERIODS independent periods, and luck (results shuffled between periods) beating it
less than settings.PASS["luck"] of the time. Fewer similar days = fewer periods = a higher bar.
No model passes = base plan. Run next to best_in_situation: the battle in step 5 shows which works better."""
SETTINGS = {"enabled": True, "days": 150}

import settings
from core.selection import Selector, allowed, mix, plain


class BestInSimilarDays(Selector):
    about = "the most trusted model on the past days most like today, else base plan"
    needs_similar = True

    def choose(self, ctx):
        if ctx.similar is None:
            return plain(ctx, "no similar days (step 5 ran without the timeline)")
        t = ctx.similar(self.s["days"])
        t = t[t.index.isin(allowed(ctx))]
        ok = t[(t["periods"] >= settings.MIN_PERIODS) & (t["trust"] >= settings.MIN_TRUST)
               & (t["luck"] < settings.PASS["luck"])]
        if ok.empty:
            best = f" (best: {t.index[0]} trust {t['trust'].iloc[0]:.1f})" if len(t) and t["trust"].notna().any() else ""
            return plain(ctx, f"no model trusted on the {self.s['days']} most similar days{best}")
        m = ok.index[0]
        r = ok.loc[m]
        return mix(ctx, {m: 1.0}, f"most trusted on the {int(r['days'])} most similar past days ({int(r['periods'])} periods): "
                                  f"IC {r['ic']:+.2f}, trust {r['trust']:.1f}, luck {r['luck']:.0%}")
