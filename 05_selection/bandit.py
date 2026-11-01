# paper-fund-v2/05_selection/bandit.py
"""Reward and punish, with a memory that fades. Every month it grades each model on the results that just
finished (the IC of that batch: did its higher guesses come before the better results? Saying "up" in a
rising market earns nothing). Good months raise a model's score, bad months lower it, old months count less
(half_life). It only follows models whose recent score is clearly above luck. Nobody = base plan."""
SETTINGS = {"enabled": True, "half_life": 24, "min_months": 12}

import numpy as np

import settings
from core.selection import Selector, allowed, mix, plain


def _ic(p, a):
    if len(p) < 5 or np.unique(p).size < 2 or np.unique(a).size < 2:
        return np.nan
    return float(np.corrcoef(p.argsort().argsort(), a.argsort().argsort())[0, 1])


class Bandit(Selector):
    about = "reward and punish: recent good months earn trust, bad months lose it"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.alpha = 1 - 0.5 ** (1 / self.s["half_life"])
        self.state, self.last = {}, None   # model -> [mean grade, variance, months counted]

    def _learn(self, ctx):
        if self.last is not None and ctx.date <= self.last:
            return
        self.last = ctx.date
        for m, g in ctx.finished.groupby("model"):
            grade = _ic(g["p50"].to_numpy(), g["actual"].to_numpy())
            if np.isnan(grade):
                continue
            if m not in self.state:
                self.state[m] = [grade, 0.0, 1.0]
                continue
            mean, var, w = self.state[m]
            d = grade - mean
            self.state[m] = [mean + self.alpha * d, (1 - self.alpha) * (var + self.alpha * d * d), (1 - self.alpha) * w + 1]

    def choose(self, ctx):
        self._learn(ctx)
        scores = {}
        for m in allowed(ctx):
            if m in self.state:
                mean, var, w = self.state[m]
                if w >= self.s["min_months"] and mean > 0:
                    scores[m] = (mean, mean / (max(np.sqrt(var), 1e-6) / np.sqrt(w)))
        passing = sorted(((m, v) for m, v in scores.items() if v[1] >= settings.MIN_TRUST), key=lambda x: -x[1][0])
        if not passing:
            return plain(ctx, "no model earned enough lately")
        top = passing[:settings.TOP_N]
        return mix(ctx, {m: v[0] for m, v in top}, "earned it lately: " + ", ".join(f"{m} {v[0]:+.2f}" for m, v in top))