# financial_market_forcasting1/core/verdict.py
"""
The monthly call per asset (BUY / WAIT / KEEP YOUR PLAN), shared by the dashboard and the emailer.

Every model votes rise / fall / no clear move (a move smaller than costs + margin is "no clear move").
Only results a model couldn't have seen count for its record, pooled for the asset (core/pooling.py:
the asset itself, its group a bit less, settings.EVIDENCE). Two records, two questions:

  BUY   is it a good time?   votes weighted by trust (IC: do its higher guesses come before better results)
  WAIT  is it a bad time?    votes weighted by the WARNING record (warn_trust: when it said "fall", did it
                             fall more often than on any day?). A model can be bad at ranking months and
                             still good at spotting drops, or the other way round.

BUY or WAIT only when settings.VOTE_AGREE of the weighted votes agree, the weighted middle guess is past the
line, and (settings.VOTE_NEED_TRUSTED) at least one model trusted for that question agrees.
Models worse than random on both records don't vote. Otherwise: keep your plan.
"""
import numpy as np
import pandas as pd

import settings
from core.metrics import FALL_BELOW, RISE_ABOVE
from core.pooling import pooled
from core.selection import stack_ok

COST = 2 * (settings.FEE_RATE + settings.SPREAD)


def firsts(j, asset):
    d = pd.DatetimeIndex(j.loc[j["asset"] == asset, "date"].unique()).sort_values()
    return list(pd.Series(d, index=d).groupby([d.year, d.month]).min())


def _snapshot(book, situation, d, asset):
    """The record of every model for this asset and situation, as of d, pooled (settings.EVIDENCE)."""
    s = book[(book["scope"] != "all") & (book["situation"] == situation) & (book["as_of"] <= d)]
    return pooled(s[s["as_of"] == s["as_of"].max()], asset) if len(s) else pd.DataFrame()


def models_on(j, book, asset, d):
    """Every model's forecast for one asset on one day + its trust in that situation and overall."""
    rows = j[(j["asset"] == asset) & (j["date"] == d)].set_index("model")
    if rows.empty:
        return rows, None
    situation = rows["situation"].dropna().iloc[0] if rows["situation"].notna().any() else None
    here, overall = _snapshot(book, situation, d, asset), _snapshot(book, "all", d, asset)
    out = rows[["price", "p10", "p50", "p90", "actual", "end_date", "seen"]].copy()
    kinds = rows["kind"].to_dict() if "kind" in rows else {}
    out["blocked"] = [not stack_ok(m, overall if len(overall) else None, kinds) for m in out.index]   # a judge that doesn't beat the best
    for col, snap, name in (("ic", here, "ic"), ("trust", here, "trust"), ("periods", here, "periods"),
                            ("trust_all", overall, "trust"), ("periods_all", overall, "periods"), ("ic_all", overall, "ic"),
                            ("warn_trust", here, "warn_trust"), ("warn_periods", here, "warn_periods"),
                            ("warn_trust_all", overall, "warn_trust"), ("warn_periods_all", overall, "warn_periods"),
                            ("warn_hit_all", overall, "warn_hit"), ("warn_base_all", overall, "warn_base")):
        out[col] = snap[name].reindex(out.index) if len(snap) and name in snap else np.nan
    bad = set(settings.EXCLUDE) | set(settings.QUARANTINE)
    ok = lambda t, p: pd.notna(t) and t >= settings.MIN_TRUST and p >= settings.MIN_PERIODS
    seen = out["seen"].astype(bool)
    out["trusted"] = [(m not in bad) and not s and (ok(t, p) or ok(ta, pa)) for m, s, t, p, ta, pa in
                      zip(out.index, seen, out["trust"], out["periods"], out["trust_all"], out["periods_all"])]
    out["warner"] = [(m not in bad) and not s and (ok(t, p) or ok(ta, pa)) for m, s, t, p, ta, pa in
                     zip(out.index, seen, out["warn_trust"], out["warn_periods"], out["warn_trust_all"], out["warn_periods_all"])]
    out.loc[out["blocked"], ["trusted", "warner"]] = False
    return out.sort_values("trust_all", ascending=False, na_position="last"), situation


def chance_up(p10, p50, p90):
    if not (p10 < p50 < p90):
        return np.nan
    return 1 - float(np.interp(0.0, [p10, p50, p90], [0.1, 0.5, 0.9], left=0.05, right=0.95))


def verdict(mt, horizon=None):
    """-> dict: call, confidence, text, votes (table), and the trust-weighted bad / middle / good forecast.
    horizon: only for the text (the forecasts in mt already are for one horizon)."""
    v = mt[~mt.index.isin(settings.EXCLUDE) & ~mt["seen"].astype(bool)].dropna(subset=["p50"])
    if "blocked" in v:
        v = v[~v["blocked"].astype(bool)]                          # a judge only votes while it beats the best model
    for col in ("warn_trust", "warn_trust_all", "warner", "warn_hit_all", "warn_base_all"):   # an older skill book
        if col not in v:
            v = v.assign(**{col: False if col == "warner" else np.nan})
    best_trust = v[["trust", "trust_all"]].max(axis=1) if len(v) else pd.Series(dtype=float)
    best_warn = v[["warn_trust", "warn_trust_all"]].max(axis=1) if len(v) else pd.Series(dtype=float)
    v = v[~((best_trust < 0) & ~(best_warn > 0))]               # worse than random on both records: no vote
    if v.empty:
        return {"call": "KEEP YOUR PLAN", "confidence": 0, "text": "no models with a usable track record",
                "votes": pd.DataFrame(), "p10": np.nan, "p50": 0.0, "p90": np.nan, "who": []}
    side = np.where(v["p50"] > RISE_ABOVE, 1, np.where(v["p50"] < FALL_BELOW, -1, 0))

    def weights(cols):   # untrusted models still count a little (+0.25)
        w = v[cols].max(axis=1).clip(lower=0).fillna(0).to_numpy() + 0.25
        return w / w.sum()

    wn, ww = weights(["trust", "trust_all"]), weights(["warn_trust", "warn_trust_all"])
    up, down = wn[side == 1].sum(), ww[side == -1].sum()          # rise votes by trust, fall votes by warning record
    p = {q: float((wn * v[q]).sum()) for q in ("p10", "p50", "p90")}
    warn_p50 = float((ww * v["p50"]).sum())
    trusted, warners = v["trusted"].to_numpy(), v["warner"].astype(bool).to_numpy()
    t_up, t_down = v.index[(side == 1) & trusted].tolist(), v.index[(side == -1) & warners].tolist()
    need = settings.VOTE_NEED_TRUSTED
    if up >= settings.VOTE_AGREE and p["p50"] > RISE_ABOVE and (t_up or not need):
        call, share, who, pool = "BUY", up, t_up, trusted
    elif down >= settings.VOTE_AGREE and warn_p50 < FALL_BELOW and (t_down or not need):
        call, share, who, pool = "WAIT", down, t_down, warners
    else:
        call, share, who, pool = "KEEP YOUR PLAN", max(up, down), [], trusted
    t_share = len(who) / max(1, int(pool.sum())) if who else 0.0
    conf = int(round(100 * share * (0.4 + 0.6 * t_share))) if who else int(round(40 * share))
    n_up, n_down = int((side == 1).sum()), int((side == -1).sum())
    text = (f"{n_up} of {len(v)} models expect a rise, {n_down} a fall (trust-weighted middle guess {p['p50']:+.1%} "
            f"in {horizon or settings.HORIZON_DAYS} trading days). ")
    if call == "WAIT":
        text += f"Good at warning and saying 'fall' now: {', '.join(who)}."
    elif who:
        text += f"Trusted and agreeing: {', '.join(who)}."
    else:
        text += (f"Trusted this month: {', '.join(v.index[trusted]) or 'none'}. "
                 f"Good at warning: {', '.join(v.index[warners]) or 'none'}.")
    record = [f"{h:.0%} vs {b:.0%}" if pd.notna(h) and pd.notna(b) else "" for h, b in zip(v["warn_hit_all"], v["warn_base_all"])]
    votes = pd.DataFrame({"expects": v["p50"].map(lambda x: f"{x:+.1%}"),
                          "vote": np.where(side == 1, "rise", np.where(side == -1, "fall", "no clear move")),
                          "weight": np.where(side == -1, ww, wn).round(3), "trusted": np.where(trusted, "★", ""),
                          "trust": v[["trust", "trust_all"]].max(axis=1).clip(lower=0).fillna(0).map(lambda x: f"{x:.1f}"),
                          "warns well": np.where(warners, "★", ""),
                          "warn trust": best_warn.reindex(v.index).map(lambda x: "" if pd.isna(x) else f"{x:.1f}"),
                          "fell after its warnings (any day)": record}).sort_values("weight", ascending=False)
    return {"call": call, "confidence": conf, "text": text, "votes": votes, "who": who, **p}


def track_record(j, asset, model, d):
    """How a model did on one asset, only results known on d. -> (stats dict, last 12 monthly calls)."""
    g = j[(j["asset"] == asset) & (j["model"] == model) & ~j["seen"].astype(bool)].dropna(subset=["actual"])
    g = g[g["end_date"] <= d]
    if g.empty:
        return None, None
    said = g[g["p50"] != 0]
    stats = {"forecasts": len(g), "since": g["date"].min(),
             "right": (np.sign(said["p50"]) == np.sign(said["actual"])).mean() if len(said) else np.nan,
             "always_up": (said["actual"] > 0).mean() if len(said) else np.nan,
             "in_range": ((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).mean()}
    m = g.assign(ym=g["date"].dt.to_period("M")).groupby("ym").first().tail(12)
    table = pd.DataFrame({"month": m.index.astype(str), "expected": m["p50"].map(lambda x: f"{x:+.1%}"),
                          "really": m["actual"].map(lambda x: f"{x:+.1%}"),
                          "right?": ["no opinion" if p == 0 else ("✅" if np.sign(p) == np.sign(a) else "❌")
                                     for p, a in zip(m["p50"], m["actual"])]})
    return stats, table


def history(j, book, assets=None):
    """The same rule every past month (only what was known then), holdout hidden unless SHOW_HOLDOUT."""
    out = []
    for a in assets or sorted(j["asset"].unique()):
        ja = j[j["asset"] == a]                                   # once per asset, not once per month (much faster)
        for d in firsts(ja, a):
            if not settings.SHOW_HOLDOUT and d >= pd.Timestamp(settings.HOLDOUT_FROM):
                continue
            mt, _ = models_on(ja, book, a, d)
            if mt.empty or mt["actual"].isna().all():
                continue
            v = verdict(mt)
            out.append({"asset": a, "date": d, "call": v["call"], "confidence": v["confidence"],
                        "real": float(mt["actual"].dropna().iloc[0])})
    return pd.DataFrame(out)


def history_summary(h):
    if h.empty:
        return pd.DataFrame()
    return h.groupby("call").agg(times=("real", "size"), avg_move_after=("real", "mean"),
                                 went_up=("real", lambda r: (r > 0).mean()), avg_confidence=("confidence", "mean"))