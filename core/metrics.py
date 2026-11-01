# financial_market_forcasting1/core/metrics.py
"""
How forecasts get scored. Calculated fresh on every run from what's in the data, nothing fixed.

  score(rows)   per group of forecasts:
    direction   share of right up/down calls          always_up  same rows, if you always said up
    edge        direction - always_up                 skill      1 - error / no-change error (above 0 = better)
    range_hit   share inside bad..good case (~80%)    spread     avg result after "up" calls minus after "down"
    ic          rank correlation guess vs result      trust      ic x sqrt(periods), above 2 = probably not luck
    periods     forecasts / horizon = roughly how many independent results the score is based on
  A model that says "no change" (middle 0%) scores exactly like naive: not punished, not rewarded.

  calls(rows)   the warning record, part of score(): how good are a model's "fall" and "rise" calls?
    warn_*      after a fall call (middle guess below FALL_BELOW): how often it really fell
    rise_*      after a rise call (middle guess above RISE_ABOVE, so it beats the costs): how often it rose
    _periods    calls / horizon       _hit    share that came true      _base   same share on ALL days
    _edge       hit - base            _move   avg result after the call minus avg result on all days
    _trust      edge in units of its luck range (binomial, per independent period), above 2 = probably not luck

  ic_by_year / bootstrap / luck_test / call_luck_test   is a score real or luck (see each function)
"""
import numpy as np
import pandas as pd

import settings

# the same lines the monthly verdict (core/verdict.py) uses for a "fall" and a "rise" vote
RISE_ABOVE = 2 * (settings.FEE_RATE + settings.SPREAD) + settings.MIN_EDGE   # a rise that beats buy + sell costs
FALL_BELOW = -settings.MIN_EDGE


def spearman(a, b):
    both = pd.concat([pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)], axis=1).dropna()
    if len(both) < 3 or both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
        return np.nan
    r = both.rank()
    return float(r.iloc[:, 0].corr(r.iloc[:, 1]))


def score(g, horizon):
    a, p50 = g["actual"].to_numpy(), g["p50"].to_numpy()
    n = len(a)
    if n == 0:
        return {}
    said = p50 != 0
    up_calls, down_calls = a[p50 > 0], a[p50 < 0]
    ic = _ic(p50, a) if np.unique(p50).size > 1 else np.nan
    periods = n / horizon
    direction = float((np.sign(p50[said]) == np.sign(a[said])).mean()) if said.any() else np.nan
    always_up = float((a[said] > 0).mean()) if said.any() else float((a > 0).mean())
    err0 = np.abs(a).mean()
    return {
        "forecasts": n, "periods": periods, "direction": direction, "always_up": always_up,
        "edge": direction - always_up if not np.isnan(direction) else np.nan,
        "skill": float(1 - np.abs(p50 - a).mean() / err0) if err0 > 0 else np.nan,
        "range_hit": float(((a >= g["p10"].to_numpy()) & (a <= g["p90"].to_numpy())).mean()),
        "spread": float(up_calls.mean() - down_calls.mean()) if len(up_calls) and len(down_calls) else np.nan,
        "ic": ic, "trust": ic * np.sqrt(periods) if not np.isnan(ic) else np.nan,
        **calls(g, horizon),
    }


def calls(g, horizon):
    """The warning record: when a model said "fall" (or "rise"), how often did that happen, compared to any day?
    A model that warns on most days just gets the base rate: no edge. Only calls that pick the bad days count."""
    a, p50 = g["actual"].to_numpy(), g["p50"].to_numpy()
    out = {}
    for name, said, came in (("warn", p50 < FALL_BELOW, a < 0), ("rise", p50 > RISE_ABOVE, a > 0)):
        n, base = int(said.sum()), float(came.mean()) if len(a) else np.nan
        hit = float(came[said].mean()) if n else np.nan
        periods = n / horizon
        luck = np.sqrt(base * (1 - base) / periods) if periods > 0 and 0 < base < 1 else np.nan
        out.update({f"{name}_periods": periods, f"{name}_hit": hit, f"{name}_base": base,
                    f"{name}_edge": hit - base, f"{name}_trust": (hit - base) / luck if luck > 0 else np.nan,
                    f"{name}_move": float(a[said].mean() - a.mean()) if n else np.nan})
    return out


def score_table(rows, by, horizon):
    out = {}
    for key, g in rows.groupby(list(by), sort=True):
        out[key[0] if len(by) == 1 else key] = score(g, horizon)
    t = pd.DataFrame(out).T
    if len(by) > 1 and len(t):
        t.index = pd.MultiIndex.from_tuples(t.index, names=list(by))
    elif len(t):
        t.index.name = by[0]
    return t.astype(float)


def _ic(x, y):
    """Rank correlation without scipy. x, y numpy arrays."""
    if len(x) < 3:
        return np.nan
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    return float(np.corrcoef(rx, ry)[0, 1])


def _usable_runs(rows):
    spread = rows.groupby("run")["p50"].nunique()
    return [r for r in spread.index if spread[r] > 1]  # naive always says 0%, nothing to rank


def ic_by_year(rows):
    out = {}
    for run in _usable_runs(rows):
        g = rows[rows["run"] == run]
        out[run] = {y: _ic(gy["p50"].to_numpy(), gy["actual"].to_numpy())
                    for y, gy in g.groupby(g["date"].dt.year)}
    t = pd.DataFrame(out).T
    t["positive"] = [f"{(r > 0).sum()} of {r.notna().sum()}" for _, r in t.iterrows()]
    return t


def bootstrap(rows, n=300, seed=0):
    rng = np.random.default_rng(seed)
    blocks = rows["date"].dt.to_period("Q")
    names = blocks.unique()
    out = {}
    for run in _usable_runs(rows):
        g = rows["run"] == run
        p50, act = rows.loc[g, "p50"].to_numpy(), rows.loc[g, "actual"].to_numpy()
        b = blocks[g].to_numpy()
        members = [np.flatnonzero(b == q) for q in names]
        ics = []
        for _ in range(n):
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[k] for k in pick])
            ics.append(_ic(p50[idx], act[idx]))
        ics = np.array(ics)
        out[run] = {"ic": _ic(p50, act), "low_5": np.nanpercentile(ics, 5), "high_95": np.nanpercentile(ics, 95),
                    "above_0": np.nanmean(ics > 0)}
    return pd.DataFrame(out).T


def luck_test(rows, n=200, min_shift=60, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for run in _usable_runs(rows):
        g = rows[rows["run"] == run].sort_values(["asset", "date"])
        per_asset = [(gg["p50"].to_numpy(), gg["actual"].to_numpy()) for _, gg in g.groupby("asset")]
        real = _ic(g["p50"].to_numpy(), g["actual"].to_numpy())
        p50_all = np.concatenate([p for p, _ in per_asset])
        luck = []
        for _ in range(n):
            shifted = []
            for p, a in per_asset:
                k = rng.integers(min_shift, max(min_shift + 1, len(a) - min_shift)) if len(a) > 2 * min_shift else 0
                shifted.append(np.roll(a, k))
            luck.append(_ic(p50_all, np.concatenate(shifted)))
        luck = np.array(luck)
        out[run] = {"ic": real, "luck_95": np.nanpercentile(luck, 95),
                    "luck_chance": (1 + np.sum(luck >= real)) / (n + 1)}
    return pd.DataFrame(out).T


def similarity(rows):
    wide = rows.pivot_table(index=["date", "asset"], columns="run", values="p50")
    wide = wide[[r for r in wide.columns if wide[r].nunique() > 1]].dropna()
    return wide.rank().corr()

def call_luck_test(rows, kind="warn", n=200, min_shift=60, seed=0):
    """Like luck_test, for the warning record: shift each asset's results by a random number of days (so calls
    and results no longer belong together) and see how often luck gets an edge as big as the real one."""
    rng = np.random.default_rng(seed)
    said_fn = (lambda p: p < FALL_BELOW) if kind == "warn" else (lambda p: p > RISE_ABOVE)
    came_fn = (lambda a: a < 0) if kind == "warn" else (lambda a: a > 0)
    out = {}
    for run, g in rows.groupby("run"):
        g = g.sort_values(["asset", "date"])
        per_asset = [(gg["p50"].to_numpy(), gg["actual"].to_numpy()) for _, gg in g.groupby("asset")]
        said = np.concatenate([said_fn(p) for p, _ in per_asset])
        if said.sum() < 5:
            continue
        edge = lambda a: came_fn(a)[said].mean() - came_fn(a).mean()
        real = edge(np.concatenate([a for _, a in per_asset]))
        luck = []
        for _ in range(n):
            shifted = [np.roll(a, rng.integers(min_shift, max(min_shift + 1, len(a) - min_shift)))
                       if len(a) > 2 * min_shift else a for _, a in per_asset]
            luck.append(edge(np.concatenate(shifted)))
        out[run] = {"edge": real, "luck_chance": (1 + np.sum(np.array(luck) >= real)) / (n + 1)}
    return pd.DataFrame(out).T
