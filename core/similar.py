# financial_market_forcasting1/core/similar.py
"""
Similar days: instead of 6 fixed situation boxes, find the past days that looked most like today and score
every model only on those. "The pattern comes back: trust the model that did well the last times."

Each day gets 7 numbers, all from its own past only (rolling windows look back, never ahead):
  trend       move over ~6 months in units of its normal swings (like the situations' trend strength)
  swings      last 20 days' swings vs its usual level of the last 2 years (log, 0 = normal)
  shape_20d   the last 20 days' move, in units of its swings
  shape_60d   the last 60 days' move, in units of its swings
  from_high   how far below its 1 year high, in units of its swings
  real_rate   60 day change of the US real interest rate (macro, only values already published)
  dollar      60 day move of the dollar (macro)

On a decision day D, for one asset:
  pool        its past days whose 20 days had ENDED by D (so their result was known on D)
  scale       each number divided by its spread over the pool: also only past data
  neighbours  the `days` pool days closest to D (plain distance on the scaled numbers)
  scores      per model on those days: IC, periods (independent 20 day blocks among them), trust = IC x
              sqrt(periods), luck = how often IC is at least as high with the blocks' results shuffled

  sd = SimilarDays(journal, tl)
  sd.scores("gold", "2015-03-02", days=150)   # index = model
"""
import numpy as np
import pandas as pd

import settings
from core.metrics import _ic

FEATURES = ["trend", "swings", "shape_20d", "shape_60d", "from_high", "real_rate", "dollar"]


def describe(closes, extras=None):
    """One row of FEATURES per day. closes: one asset's closes; extras: {name: Series by date it became known}."""
    c = closes.astype(float)
    lr = np.log(c).diff()
    sd20, sd60 = lr.rolling(20).std(), lr.rolling(60).std()
    f = pd.DataFrame(index=c.index)
    f["trend"] = np.log(c / c.shift(120)) / (sd60 * np.sqrt(120))
    f["swings"] = np.log(sd20 / sd20.rolling(504, min_periods=252).median())
    f["shape_20d"] = np.log(c / c.shift(20)) / (sd60 * np.sqrt(20))
    f["shape_60d"] = np.log(c / c.shift(60)) / (sd60 * np.sqrt(60))
    f["from_high"] = np.log(c / c.rolling(252, min_periods=50).max()) / (sd60 * np.sqrt(252))
    for name, how in (("real_rate", "change"), ("dollar", "move")):
        s = (extras or {}).get(name)
        if s is None or s.dropna().empty:
            f[name] = np.nan
            continue
        s = s.dropna().sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = s - s.shift(60) if how == "change" else np.log(s / s.shift(60))
        f[name] = s.reindex(s.index.union(f.index)).ffill().reindex(f.index)   # last value known on each day
    return f[FEATURES]


class SimilarDays:
    def __init__(self, journal, tl, horizon=None):
        self.h = horizon or settings.HORIZON_DAYS
        extras = tl.extras()
        last = tl.view(tl.calendar()[-1])
        j = journal.dropna(subset=["p50"])
        self._a = {}
        for asset, g in j.groupby("asset"):
            closes = last.closes(asset)
            desc = describe(closes, extras)
            first = g.drop_duplicates("date").set_index("date").sort_index()
            days = first.index
            p50 = g.pivot_table(index="date", columns="model", values="p50").reindex(days)
            seen = g.pivot_table(index="date", columns="model", values="seen", aggfunc="max").reindex(days)
            p50 = p50.where(~seen.fillna(True).astype(bool))       # never score a model on prices it may have seen
            pos = closes.index.get_indexer(days)
            self._a[asset] = {"days": days, "X": desc.reindex(days).to_numpy(float),
                              "end": first["end_date"].to_numpy("datetime64[ns]"), "actual": first["actual"].to_numpy(float),
                              "block": np.where(pos >= 0, pos // self.h, -1), "p50": p50}

    def neighbours(self, asset, date, days=150):
        """Positions (into this asset's journal days) of the `days` known past days most like `date`."""
        a = self._a.get(asset)
        if a is None:
            return np.array([], dtype=int)
        d = np.datetime64(pd.Timestamp(date))
        i = int(np.searchsorted(a["days"].values, d, side="left"))
        if i >= len(a["days"]) or a["days"][i] != pd.Timestamp(date):
            return np.array([], dtype=int)
        pool = np.flatnonzero((a["end"] <= d) & ~np.isnan(a["actual"]))   # result known on `date`
        if len(pool) < days:
            return np.array([], dtype=int)
        X, q = a["X"][pool], a["X"][i]
        spread = np.nanstd(X, axis=0)
        usable = np.isfinite(q) & (spread > 0)
        if usable.sum() < 3:
            return np.array([], dtype=int)
        z = (X[:, usable] - np.nanmean(X[:, usable], axis=0)) / spread[usable]
        zq = (q[usable] - np.nanmean(X[:, usable], axis=0)) / spread[usable]
        dist = np.sqrt(np.nansum((z - zq) ** 2, axis=1) + np.isnan(z).sum(axis=1) * 1.0)   # a missing number counts as 1 apart
        return pool[np.argsort(dist, kind="stable")[:days]]

    def scores(self, asset, date, days=150, luck_draws=200, seed=0):
        """Per model, on the similar days: days, periods, ic, trust, luck. Empty if too little history."""
        idx = self.neighbours(asset, date, days)
        if not len(idx):
            return pd.DataFrame(columns=["days", "periods", "ic", "trust", "luck"])
        a = self._a[asset]
        act, blk = a["actual"][idx], a["block"][idx]
        rng = np.random.default_rng(seed)
        out = {}
        for model in a["p50"].columns:
            p = a["p50"][model].to_numpy(float)[idx]
            ok = ~np.isnan(p)
            if ok.sum() < 10 or np.unique(p[ok]).size < 2:
                continue
            ic, periods = _ic(p[ok], act[ok]), len(np.unique(blk[ok]))
            out[model] = {"days": int(ok.sum()), "periods": periods, "ic": ic,
                          "trust": ic * np.sqrt(periods) if pd.notna(ic) else np.nan,
                          "luck": _block_luck(p[ok], act[ok], blk[ok], ic, luck_draws, rng)}
        return pd.DataFrame(out).T.sort_values("trust", ascending=False) if out else \
            pd.DataFrame(columns=["days", "periods", "ic", "trust", "luck"])


def _block_luck(p, a, blk, real, n, rng):
    """Shuffle the results between independent 20 day blocks (one point per block: overlapping days aren't
    independent) and count how often that luck reaches the real IC. Only worth it for a positive IC."""
    if not (pd.notna(real) and real > 0):
        return np.nan
    b = pd.DataFrame({"p": p, "a": a, "b": blk}).groupby("b").mean()
    if len(b) < 5:
        return np.nan
    bp, ba = b["p"].to_numpy(), b["a"].to_numpy()
    real_b = _ic(bp, ba)
    if not pd.notna(real_b):
        return np.nan
    hits = sum(_ic(bp, rng.permutation(ba)) >= real_b for _ in range(n))
    return (1 + hits) / (n + 1)
