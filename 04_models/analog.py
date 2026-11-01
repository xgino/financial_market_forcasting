# paper-fund-v2/04_models/analog.py
"""Pattern matching (analog forecasting): finds the past moments whose last 60 days looked most like
now (all assets if pool is on), and uses what happened in the 20 days after them.
Only past moments whose 20 days had already ENDED by today are used."""
SETTINGS = {"enabled": True, "window": 60, "neighbours": 50, "pool": True, "min_library": 500}
CARD = {  # shown on the dashboard page Models
    "type": 'statistics (pattern matching)',
    "trained": 'No training. Searches the past of all 6 traded assets.',
    "strengths": 'Finds moments that looked like now and shows what happened after: easy to understand.',
    "weaknesses": "Similar shapes don't mean similar causes.",
    "limits": 'Only the shape of the last 60 days, nothing else.',
    "source": 'Analog forecasting / nearest neighbours: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm',
}

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from core.contracts import Model


class Analog(Model):
    """Pattern matching (analog forecasting).
    1. Take the shape of the last `window` days, scaled so a calm and a wild asset are comparable.
    2. Find the `neighbours` past moments with the most similar shape (all assets if pool is on).
    3. Look at what happened in the 20 days after each of them. Their 10% / 50% / 90% = the forecast.
    Only past moments whose 20 days had already ENDED by today are used."""

    kind = "stats"
    about = "Pattern matching: finds past moments whose last 60 days looked most like now, uses what came after."

    def __init__(self, horizon, cfg):
        super().__init__(horizon, cfg)
        s = self.s
        self.window, self.k, self.pool, self.min_library = s["window"], s["neighbours"], s["pool"], s["min_library"]
        self.uses_all_assets = self.pool  # forecasts depend on other assets too (matters for the cache)

    def _shape(self, lc):
        """lc: log closes of one window -> scale-free shape vector and daily swing size."""
        vol = np.diff(lc).std(ddof=1)
        return (lc - lc[-1]) / (vol * np.sqrt(self.window)), vol

    def _library(self, view, asset):
        """Every past window, its shape, what came after (scaled), and the date that became known."""
        W, H = self.window, self.horizon
        shapes, outcomes, known = [], [], []
        for a in (view.assets if self.pool else [asset]):
            c = view.closes(a)
            if len(c) < W + H + 1:
                continue
            lc = np.log(c.to_numpy())
            wins = sliding_window_view(lc, W)                       # row j = days j .. j+W-1
            vol = sliding_window_view(np.diff(lc), W - 1).std(axis=1, ddof=1)
            ends = np.arange(W - 1, len(lc))
            done = ends + H < len(lc)                               # 20 days after it are in the data
            ends, wins, vol = ends[done], wins[done], vol[done]
            ok = vol > 0
            ends, wins, vol = ends[ok], wins[ok], vol[ok]
            shapes.append((wins - wins[:, -1:]) / (vol[:, None] * np.sqrt(W)))
            outcomes.append((lc[ends + H] - lc[ends]) / (vol * np.sqrt(H)))
            known.append(c.index[ends + H].to_numpy())
        if not shapes:
            return None
        known = np.concatenate(known)
        order = np.argsort(known, kind="stable")                    # sorted by the day the result was known
        return (np.concatenate(shapes)[order].astype(np.float32),
                np.concatenate(outcomes)[order], known[order])

    def predict(self, view, asset):
        return self.predict_batch([view], asset)[0]

    def predict_batch(self, views, asset):
        out = np.full((len(views), 3), np.nan)
        latest = max(views, key=lambda v: v.date)
        lib = self._library(latest, asset)
        if lib is None:
            return out
        shapes, outcomes, known = lib
        for i, v in enumerate(views):
            n_ok = int(np.searchsorted(known, np.datetime64(v.date), side="right"))  # results known on v.date
            if n_ok < max(self.min_library, self.k):
                continue
            c = v.closes(asset, self.window).to_numpy()
            if len(c) < self.window:
                continue
            q, vol_now = self._shape(np.log(c))
            if not vol_now > 0:
                continue
            dist = ((shapes[:n_ok] - q.astype(np.float32)) ** 2).sum(axis=1)
            nearest = np.argpartition(dist, self.k - 1)[:self.k]
            z = np.quantile(outcomes[:n_ok][nearest], [0.1, 0.5, 0.9])
            out[i] = np.exp(z * vol_now * np.sqrt(self.horizon)) - 1
        return out

    def explain(self, view, asset):
        lib = self._library(view, asset)
        c = view.closes(asset, self.window).to_numpy()
        if lib is None or len(c) < self.window:
            return ["Not enough history yet."]
        shapes, outcomes, known = lib
        n_ok = int(np.searchsorted(known, np.datetime64(view.date), side="right"))
        q, vol = self._shape(np.log(c))
        dist = ((shapes[:n_ok] - q.astype(np.float32)) ** 2).sum(axis=1)
        near = np.argpartition(dist, self.k - 1)[:self.k]
        res = np.expm1(outcomes[:n_ok][near] * vol * np.sqrt(self.horizon))
        when = sorted(pd.DatetimeIndex(known[:n_ok][near[np.argsort(dist[near])[:3]]]).strftime("%Y-%m"))
        return [f"Looked for the {self.k} past moments (all assets) whose last {self.window} days looked most like now.",
                f"After them, {(res > 0).mean():.0%} went up, the middle result was {np.median(res):+.1%}.",
                f"The 3 most similar ones were known around {', '.join(when)}."]