# financial_market_forcasting1/core/selection.py
"""
Who to trust. Once a month (first trading day), per asset, a selector decides which forecast to follow:
one model, a mix of models, or the base plan (no opinion on direction). It only gets what was known that
day: today's forecasts, today's situation, the skill book as of today and results that just finished.

Every choice also lists the top N models for today's situation (as the skill book ranked them that day)
with their forecasts, trusted or not, so you always see what the best ones expect.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import settings
from core.contracts import Player
from core.pooling import pooled


@dataclass
class Choice:
    p10: float
    p50: float
    p90: float
    used: dict = field(default_factory=dict)   # model -> weight. empty = base plan
    reason: str = ""


@dataclass
class Context:
    date: pd.Timestamp
    asset: str
    situation: str
    forecasts: pd.DataFrame      # index = model: p10 p50 p90, all made today
    trust: pd.DataFrame          # skill book as of today, this situation, pooled for this asset (core/pooling.py)
    finished: pd.DataFrame       # journal rows whose 20 days ended since the previous decision day
    trust_all: pd.DataFrame = None  # same, ALL situations together (more results per model)
    similar: object = None          # similar(days) -> scores on the past days most like today (core/similar.py)
    kinds: dict = None              # model -> kind (rule, stats, ml, pretrained, stack)
    stretch: str = None             # today's stretch (core/stretches.py), e.g. "down-sharp"
    by_label: object = None         # by_label("stretch:down-sharp") -> skill book for that label, pooled for this asset


class Selector(Player):
    """05_selection players. choose(ctx) -> Choice."""

    def choose(self, ctx):
        raise NotImplementedError


def allowed(ctx):
    bad = set(settings.EXCLUDE) | set(settings.QUARANTINE)
    return [m for m in ctx.forecasts.dropna().index if m not in bad and stack_ok(m, ctx.trust_all, ctx.kinds)]


def stack_ok(m, record, kinds):
    """A judge (kind "stack") may only be followed while its record so far (all situations, pooled for this
    asset) has a higher IC than the best single model's record so far. Any other model: always True."""
    kinds = kinds or {}
    if kinds.get(m) != "stack":
        return True
    if record is None or m not in record.index or "ic" not in record or pd.isna(record.at[m, "ic"]):
        return False
    bad = set(settings.EXCLUDE) | set(settings.QUARANTINE)
    singles = record.loc[[x for x in record.index if kinds.get(x) != "stack" and x not in bad], "ic"].dropna()
    return len(singles) > 0 and record.at[m, "ic"] > singles.max()


def trusted(ctx, table=None):
    t = ctx.trust if table is None else table
    ok = [m for m in allowed(ctx) if m in t.index and pd.notna(t.at[m, "trust"])
          and t.at[m, "periods"] >= settings.MIN_PERIODS and t.at[m, "trust"] >= settings.MIN_TRUST]
    return t.loc[ok].sort_values("ic", ascending=False)


def plain(ctx, reason):
    """The base plan: no opinion on direction (middle 0%), range from the range model."""
    f = ctx.forecasts
    for m in (settings.RANGE_MODEL, "naive"):
        if m in f.index and f.loc[m].notna().all():
            return Choice(float(f.at[m, "p10"]), 0.0, float(f.at[m, "p90"]), {}, reason)
    return Choice(np.nan, 0.0, np.nan, {}, reason)


def mix(ctx, weights, reason):
    """Weighted mix of forecasts. weights {model: w}, the key None = base plan (middle 0%)."""
    base = plain(ctx, reason)
    total = sum(weights.values())
    p = np.zeros(3)
    for m, w in weights.items():
        row = (base.p10, 0.0, base.p90) if m is None else ctx.forecasts.loc[m, ["p10", "p50", "p90"]]
        p += (w / total) * np.asarray(row, dtype=float)
    return Choice(float(p[0]), float(p[1]), float(p[2]), {m: w / total for m, w in weights.items() if m}, reason)


def decision_days(journal):
    out = []
    for asset, g in journal.groupby("asset"):
        d = pd.DatetimeIndex(g["date"].unique()).sort_values()
        out += [(day, asset) for day in pd.Series(d, index=d).groupby([d.year, d.month]).min()]
    return sorted(out)


def run(journal, book, selectors, tl=None, progress=True):
    """tl (the timeline) is only needed for selectors that look at similar days."""
    sd = None
    if tl is not None and any(getattr(s, "needs_similar", False) for s in selectors.values()):
        from core.similar import SimilarDays
        sd = SimilarDays(journal, tl)
    events = decision_days(journal)
    days = {d for d, _ in events}
    today = journal[journal["date"].isin(days)]
    fc = {k: g.set_index("model") for k, g in today.groupby(["asset", "date"])}
    book = book[book["scope"] != "all"]   # per asset rows: pooled per decision (settings.EVIDENCE)
    by_snap = {s: g for s, g in book.groupby("as_of")}
    snaps = pd.DatetimeIndex(sorted(by_snap))
    kinds = journal.drop_duplicates("model").set_index("model")["kind"].to_dict() if "kind" in journal else {}
    known = journal.dropna(subset=["actual", "end_date", "p50"])
    known = known[~known["seen"].astype(bool)].sort_values("end_date")  # no learning on trained-on years
    ends = known["end_date"].to_numpy()

    rows, prev = [], None
    for i, (d, asset) in enumerate(events):
        f = fc.get((asset, d))
        if f is None:
            continue
        k = int(snaps.searchsorted(d, side="right"))     # latest skill book snapshot on or before today
        snap = by_snap[snaps[k - 1]] if k else book.iloc[0:0]
        situation = f["situation"].dropna().iloc[0] if f["situation"].notna().any() else None
        lo = 0 if prev is None else int(np.searchsorted(ends, np.datetime64(prev), side="right"))
        hi = int(np.searchsorted(ends, np.datetime64(d), side="right"))
        trust = pooled(snap[snap["situation"] == situation], asset)
        ctx = Context(d, asset, situation, f[["p10", "p50", "p90"]], trust, known.iloc[lo:hi],
                      trust_all=pooled(snap[snap["situation"] == "all"], asset),
                      similar=(lambda days, a=asset, day=d: sd.scores(a, day, days)) if sd is not None else None,
                      kinds=kinds,
                      stretch=f["stretch"].dropna().iloc[0] if "stretch" in f and f["stretch"].notna().any() else None,
                      by_label=lambda label, sn=snap, a=asset: pooled(sn[sn["situation"] == label], a))
        prev = d
        ranked = trust.reindex([m for m in allowed(ctx) if m in trust.index])
        ranked = ranked.sort_values("ic", ascending=False) if "ic" in ranked else ranked
        top = ", ".join(f"{m} {f.at[m, 'p50']:+.1%} (IC {ranked.at[m, 'ic']:+.2f})"
                        for m in ranked.index[:settings.TOP_N] if pd.notna(ranked.at[m, "ic"]))
        first = f.iloc[0]
        for name, sel in selectors.items():
            c = sel.choose(ctx)
            main = max(c.used, key=c.used.get) if c.used else "base plan"
            rows.append({"selector": name, "asset": asset, "date": d, "end_date": first["end_date"],
                         "situation": situation, "price": first["price"], "p10": c.p10, "p50": c.p50, "p90": c.p90,
                         "main_model": main, "used": ", ".join(f"{m} {w:.0%}" for m, w in sorted(c.used.items(), key=lambda x: -x[1])),
                         "reason": c.reason, "top_models": top, "actual": first["actual"]})
        if progress and i % 100 == 0:
            print(f"\r  choosing: {i + 1}/{len(events)} decision days", end="", flush=True)
    if progress:
        print(f"\r  choosing: {len(events)}/{len(events)} decision days")
    return pd.DataFrame(rows)


def _ic(x, y):
    if len(x) < 3 or np.unique(x).size < 2:
        return np.nan
    return float(np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1])


def evaluate(choices, by=("selector",)):
    out = {}
    for key, g in choices.dropna(subset=["actual"]).groupby(list(by)):
        act, up, a = g[g["p50"] != 0], g[g["p50"] > 0], g["actual"].to_numpy()
        out[key if len(by) > 1 else key[0]] = {
            "months": len(g), "has_opinion": len(act) / len(g),
            "direction": (np.sign(act["p50"]) == np.sign(act["actual"])).mean() if len(act) else np.nan,
            "always_up": (act["actual"] > 0).mean() if len(act) else np.nan,
            "ic": _ic(act["p50"].to_numpy(), act["actual"].to_numpy()) if len(act) else np.nan,
            "move_when_up": up["actual"].mean() if len(up) else np.nan, "move_all": a.mean(),
            "skill": 1 - np.abs(g["p50"] - g["actual"]).mean() / np.abs(a).mean(),
            "range_hit": ((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).mean()}
    return pd.DataFrame(out).T