# paper-fund-v2/core/verdict.py
"""
The monthly call per asset (BUY / WAIT / KEEP YOUR PLAN), shared by the dashboard and the emailer.

Every model votes rise / fall / no clear move (a move smaller than costs + margin is "no clear move").
A vote counts more when the model earned more trust (its track record, only results it couldn't have seen).
Models with a track record worse than random (trust below 0) don't vote at all.
BUY or WAIT only when settings.VOTE_AGREE of the weighted votes agree, the weighted average beats the costs,
and (settings.VOTE_NEED_TRUSTED) at least one trusted model agrees. Otherwise: keep your plan.
"""
import numpy as np
import pandas as pd

import settings

COST = 2 * (settings.FEE_RATE + settings.SPREAD)


def firsts(j, asset):
    d = pd.DatetimeIndex(j.loc[j["asset"] == asset, "date"].unique()).sort_values()
    return list(pd.Series(d, index=d).groupby([d.year, d.month]).min())


def _snapshot(book, situation, d):
    s = book[(book["scope"] == "all") & (book["situation"] == situation) & (book["as_of"] <= d)]
    return s[s["as_of"] == s["as_of"].max()].set_index("model") if len(s) else pd.DataFrame()


def models_on(j, book, asset, d):
    """Every model's forecast for one asset on one day + its trust in that situation and overall."""
    rows = j[(j["asset"] == asset) & (j["date"] == d)].set_index("model")
    if rows.empty:
        return rows, None
    situation = rows["situation"].dropna().iloc[0] if rows["situation"].notna().any() else None
    here, overall = _snapshot(book, situation, d), _snapshot(book, "all", d)
    out = rows[["price", "p10", "p50", "p90", "actual", "end_date", "seen"]].copy()
    for col, snap, name in (("ic", here, "ic"), ("trust", here, "trust"), ("periods", here, "periods"),
                            ("trust_all", overall, "trust"), ("periods_all", overall, "periods"), ("ic_all", overall, "ic")):
        out[col] = snap[name].reindex(out.index) if len(snap) else np.nan
    bad = set(settings.EXCLUDE) | set(settings.QUARANTINE)
    ok = lambda t, p: pd.notna(t) and t >= settings.MIN_TRUST and p >= settings.MIN_PERIODS
    out["trusted"] = [(m not in bad) and not s and (ok(t, p) or ok(ta, pa)) for m, s, t, p, ta, pa in
                      zip(out.index, out["seen"].astype(bool), out["trust"], out["periods"], out["trust_all"], out["periods_all"])]
    return out.sort_values("trust_all", ascending=False, na_position="last"), situation


def chance_up(p10, p50, p90):
    if not (p10 < p50 < p90):
        return np.nan
    return 1 - float(np.interp(0.0, [p10, p50, p90], [0.1, 0.5, 0.9], left=0.05, right=0.95))


def verdict(mt):
    """-> dict: call, confidence, text, votes (table), and the trust-weighted bad / middle / good forecast."""
    v = mt[~mt.index.isin(settings.EXCLUDE) & ~mt["seen"].astype(bool)].dropna(subset=["p50"])
    best_trust = v[["trust", "trust_all"]].max(axis=1) if len(v) else pd.Series(dtype=float)
    v = v[~(best_trust < 0)]                                     # worse than random: no vote
    if v.empty:
        return {"call": "KEEP YOUR PLAN", "confidence": 0, "text": "no models with a usable track record",
                "votes": pd.DataFrame(), "p10": np.nan, "p50": 0.0, "p90": np.nan, "who": []}
    bt = v[["trust", "trust_all"]].max(axis=1).clip(lower=0).fillna(0).to_numpy()
    side = np.where(v["p50"] > COST + settings.MIN_EDGE, 1, np.where(v["p50"] < -settings.MIN_EDGE, -1, 0))
    w = bt + 0.25                                                # untrusted models still count a little
    wn = w / w.sum()
    up, down = wn[side == 1].sum(), wn[side == -1].sum()
    p = {q: float((wn * v[q]).sum()) for q in ("p10", "p50", "p90")}
    trusted = v["trusted"].to_numpy()
    t_up, t_down = v.index[(side == 1) & trusted].tolist(), v.index[(side == -1) & trusted].tolist()
    need = settings.VOTE_NEED_TRUSTED
    if up >= settings.VOTE_AGREE and p["p50"] > COST + settings.MIN_EDGE and (t_up or not need):
        call, share, who = "BUY", up, t_up
    elif down >= settings.VOTE_AGREE and p["p50"] < -settings.MIN_EDGE and (t_down or not need):
        call, share, who = "WAIT", down, t_down
    else:
        call, share, who = "KEEP YOUR PLAN", max(up, down), []
    t_share = len(who) / max(1, int(trusted.sum())) if who else 0.0
    conf = int(round(100 * share * (0.4 + 0.6 * t_share))) if who else int(round(40 * share))
    n_up, n_down = int((side == 1).sum()), int((side == -1).sum())
    text = (f"{n_up} of {len(v)} models expect a rise, {n_down} a fall (trust-weighted middle guess {p['p50']:+.1%} "
            f"in 20 trading days). " + (f"Trusted and agreeing: {', '.join(who)}." if who else
                                        f"Trusted this month: {', '.join(v.index[trusted]) or 'none'}."))
    votes = pd.DataFrame({"expects": v["p50"].map(lambda x: f"{x:+.1%}"),
                          "vote": np.where(side == 1, "rise", np.where(side == -1, "fall", "no clear move")),
                          "weight": wn.round(3), "trusted": np.where(trusted, "★", ""),
                          "trust": pd.Series(bt, index=v.index).map(lambda x: f"{x:.1f}")}).sort_values("weight", ascending=False)
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
        for d in firsts(j, a):
            if not settings.SHOW_HOLDOUT and d >= pd.Timestamp(settings.HOLDOUT_FROM):
                continue
            mt, _ = models_on(j, book, a, d)
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