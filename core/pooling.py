# financial_market_forcasting1/core/pooling.py
"""
Whose results count for a model's trust on an asset. The skill book scores every model per asset (and per
asset and situation); this combines those into one record for the asset you decide about:

  own     the asset itself              weight settings.POOLS[pool]["own"]
  group   assets in the same group      weight ...["group"]   (settings.GROUPS: metals, stocks, bonds, crypto)
  other   every other asset             weight ...["other"]

Averages (IC, direction, hit rates, ...) are weighted by weight x periods, so a big record counts more than a
small one. Periods add up weighted too, so pooled evidence raises trust less than the asset's own results:
  periods = sum(w x periods)    ic = sum(w x periods x ic) / periods    trust = ic x sqrt(periods)
The warning record (warn_*, rise_*) is pooled the same way and its trust recomputed from the pooled numbers.

  pooled(book_rows, "gold")       one row per model: the record for gold under settings.EVIDENCE
  check(journal)                  does the pool's trust at the start of a year predict the next year? (step 4)
"""
import numpy as np
import pandas as pd

import settings
from core.metrics import _ic

BY_PERIODS = ["ic", "direction", "always_up", "edge", "skill", "range_hit", "warn_base", "rise_base"]
BY_CALLS = {"warn": ["warn_hit", "warn_move"], "rise": ["rise_hit", "rise_move"]}
EMPTY = ["periods", "ic", "trust", "warn_periods", "warn_trust", "rise_periods", "rise_trust"]


def group_of(asset):
    return next((g for g, members in settings.GROUPS.items() if asset in members), None)


def weights(asset, assets, pool=None):
    """{asset: weight} for every asset in `assets`, seen from `asset`."""
    w = settings.POOLS[pool or settings.EVIDENCE]
    g = group_of(asset)
    return {a: w["own"] if a == asset else w["group"] if g is not None and group_of(a) == g else w["other"]
            for a in assets}


def pooled(rows, asset, pool=None):
    """rows: skill book rows of ONE snapshot and ONE situation, scope = asset name. -> index model."""
    rows = rows[rows["scope"] != "all"]
    if rows.empty:
        return pd.DataFrame(columns=EMPTY, dtype=float)
    w = rows["scope"].map(weights(asset, rows["scope"].unique(), pool)).astype(float)
    rows = rows[w > 0].assign(_w=w[w > 0])
    if rows.empty:
        return pd.DataFrame(columns=EMPTY, dtype=float)
    out = {}
    for model, g in rows.groupby("model"):
        r = {"periods": _wsum(g["_w"], g["periods"])}
        for c in BY_PERIODS:
            if c in g:
                r[c] = _wavg(g[c], g["_w"] * g["periods"])
        r["trust"] = r["ic"] * np.sqrt(r["periods"]) if pd.notna(r.get("ic")) else np.nan
        for k, cols in BY_CALLS.items():
            if f"{k}_periods" not in g:
                continue
            calls = g["_w"] * g[f"{k}_periods"].fillna(0)
            r[f"{k}_periods"] = float(calls.sum())
            for c in cols:
                r[c] = _wavg(g[c], calls)
            base, n = r.get(f"{k}_base"), r[f"{k}_periods"]
            r[f"{k}_edge"] = r[f"{k}_hit"] - base if pd.notna(r[f"{k}_hit"]) and pd.notna(base) else np.nan
            luck = np.sqrt(base * (1 - base) / n) if n > 0 and pd.notna(base) and 0 < base < 1 else np.nan
            r[f"{k}_trust"] = r[f"{k}_edge"] / luck if pd.notna(luck) and luck > 0 else np.nan
        out[model] = r
    return pd.DataFrame(out).T


def _wsum(w, x):
    ok = x.notna()
    return float((w[ok] * x[ok]).sum())


def _wavg(x, w):
    ok = x.notna() & w.notna() & (w > 0)
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def check(journal, horizon=None, first_year=None, last_year=None):
    """For every year Y, every pool: each model's trust on each asset from results known on 1 January of Y
    (all situations), against the IC it really had on that asset during Y. A pool is better when its trust
    predicts next year better ("predicts", rank correlation over model x asset) and it is steadier (trust
    on 1 Jan of Y vs 1 Jan of Y+1, "steady"). Development years only, unseen results only."""
    h = horizon or settings.HORIZON_DAYS
    first_year = first_year or int(settings.WARMUP_UNTIL[:4]) + 1
    last_year = last_year or int(settings.HOLDOUT_FROM[:4]) - 1
    j = journal.dropna(subset=["p50", "actual", "end_date"])
    j = j[~j["seen"].astype(bool)]
    assets = sorted(j["asset"].unique())
    rows = []
    for pool in settings.POOLS:
        prev = None
        for y in range(first_year, last_year + 1):
            start, end = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
            past = j[j["end_date"] < start]
            year = j[(j["date"] >= start) & (j["date"] < end) & (j["end_date"] < end)]
            rec = _records(past, h)                                     # (model, asset) -> ic, periods
            if rec.empty or year.empty:
                continue
            trust = {}
            for a in assets:
                w = weights(a, assets, pool)
                r = rec.assign(_w=rec["asset"].map(w)).query("_w > 0")
                for m, g in r.groupby("model"):
                    p = float((g["_w"] * g["periods"]).sum())
                    ok = g["ic"].notna()
                    if p > 0 and ok.any():
                        ic = float((g.loc[ok, "_w"] * g.loc[ok, "periods"] * g.loc[ok, "ic"]).sum()
                                   / (g.loc[ok, "_w"] * g.loc[ok, "periods"]).sum())
                        trust[m, a] = ic * np.sqrt(p)
            real = {(m, a): _ic(g["p50"].to_numpy(), g["actual"].to_numpy())
                    for (m, a), g in year.groupby(["model", "asset"]) if g["p50"].nunique() > 1}
            both = [k for k in trust if k in real and pd.notna(real[k])]
            if len(both) >= 5:
                rows.append({"pool": pool, "year": y, "pairs": len(both),
                             "predicts": _ic(np.array([trust[k] for k in both]), np.array([real[k] for k in both])),
                             "steady": _steady(prev, trust)})
            prev = trust
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.groupby("pool").agg(years=("year", "size"), pairs=("pairs", "mean"), predicts=("predicts", "mean"),
                                 predicts_low=("predicts", "min"), steady=("steady", "mean")).reindex(list(settings.POOLS))


def _records(rows, h):
    """IC and periods per (model, asset) on these rows."""
    out = []
    for (m, a), g in rows.groupby(["model", "asset"]):
        if g["p50"].nunique() > 1:
            out.append({"model": m, "asset": a, "periods": len(g) / h,
                        "ic": _ic(g["p50"].to_numpy(), g["actual"].to_numpy())})
    return pd.DataFrame(out)


def _steady(prev, now):
    if not prev:
        return np.nan
    both = [k for k in now if k in prev]
    return _ic(np.array([prev[k] for k in both]), np.array([now[k] for k in both])) if len(both) >= 5 else np.nan
