# financial_market_forcasting1/core/battle.py
"""
The battle: every model forecasts every trading day as if live, and gets scored once the 20 days are over.

  forecast_history  every forecast of one model on one asset, cached in data/forecasts/<asset>/<model>.csv
  build_journal     every forecast of every model + situation + what really happened
  skillbook         month by month: how good each model was per situation, using only what was known then
  leaderboard       the ranking (development period only, unless settings.SHOW_HOLDOUT)
  winners           per asset per month: whose price guess came closest
  check_no_future   proves a model can't see the future

Rules that keep it honest:
- a forecast for day D gets a view of D only (core/timeline.py)
- a score "as of" a date only uses forecasts whose 20 days had ended by then
- pretrained models never earn trust on prices they may have trained on (SETTINGS seen_until)
"""
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import settings
from core import metrics, store
from core.situations import regime_table
from core.stretches import stretch_table
from core.timeline import Timeline

COLS = ["p10", "p50", "p90"]


# ---------- forecasts ----------

def horizon_name(name, horizon):
    """journal / skillbook / leaderboard file names: the main horizon keeps the plain name, others get _h<n>."""
    return name if horizon in (None, settings.HORIZON_DAYS) else f"{name}_h{horizon}"


SAVE_EVERY = 60   # seconds: forecasts are saved at least this often, so a stopped run loses at most ~1 minute


def assets_used(model, tl, asset):
    return list(tl.assets) if getattr(model, "uses_all_assets", False) else [asset]


def data_hash(tl, assets, until, macro=False):
    v = tl.view(until)
    parts = [v.closes(a).round(6) for a in sorted(assets)]
    if macro:
        parts += [v.extra(n).round(6) for n in sorted(tl.extra_names)]
    return store.fingerprint(*parts)


def month_hashes(tl, assets, until, macro=False):
    """{"2014-03": fingerprint of that month's data}. When old prices change, only forecasts from the
    first changed month on are made again: a forecast never depends on later data.
    Prices count as daily moves, not levels: Yahoo rescales an ETF's whole past at every dividend, which
    changes no move except the dividend day's (and every model here reads moves, not euro levels)."""
    v = tl.view(until)
    parts = [np.log(v.closes(a)).diff().round(6) for a in sorted(assets)]
    if macro:
        parts += [v.extra(n).round(6) for n in sorted(tl.extra_names)]
    months = sorted({str(m) for p in parts for m in p.index.to_period("M")})
    by = [p.groupby(p.index.to_period("M").astype(str)) for p in parts]
    return {m: store.fingerprint(*[g.get_group(m) for g in by if m in g.groups]) for m in months}


def first_changed_day(old, new):
    """First day of the first month whose data differs (None = nothing changed up to old's last month)."""
    for m in sorted(old):
        if old[m] != new.get(m):
            return pd.Timestamp(m + "-01")
    return None


def eligible_days(tl, asset, model, use_first_day=True):
    days = tl.view(tl.calendar()[-1]).closes(asset).index
    days = days[days >= tl.first_date(asset, model.min_history)]
    if use_first_day and model.s.get("first_day"):
        days = days[days >= pd.Timestamp(model.s["first_day"])]
    return days


def forecast_history(model, tl, asset, fresh=False, progress=True):
    """Forecast for every eligible day. Only new days are computed (a stopped run continues where it was);
    changed settings mean everything again, changed old prices only from the first changed month.
    fresh=True ignores what's saved and replaces it."""
    days = eligible_days(tl, asset, model)
    p = store.path("forecasts", asset, f"{model.cache_name}.csv")
    used = assets_used(model, tl, asset)
    settings_key = store.fingerprint(type(model).__name__, model.horizon, model.cache_settings())
    macro = getattr(model, "uses_macro", False)
    cached = pd.DataFrame(columns=COLS, dtype=float)
    if p.exists() and not fresh:
        m = store.read_meta(p)
        if m.get("settings") == settings_key and m.get("assets") == sorted(used):
            if m.get("data") == data_hash(tl, used, m["until"], macro):
                cached = store.read_csv(p, index_col="date")
            elif m.get("months"):   # old prices changed: keep everything before the first changed month
                cut = first_changed_day(m["months"], month_hashes(tl, used, m["until"], macro))
                cached = store.read_csv(p, index_col="date")
                if cut is not None:
                    cached = cached[cached.index < cut]
                if progress:
                    print(f"  {asset} {model.run_name}: " + (f"old prices changed from {cut.date()}, redoing from there"
                          if cut is not None else "old prices only rescaled (dividend), nothing to redo"))
            elif progress:
                print(f"  {asset} {model.run_name}: old prices changed, computing again")
        elif progress:
            print(f"  {asset} {model.run_name}: settings changed, computing again")

    todo = days.difference(cached.index)
    if len(todo):
        step = getattr(model, "chunk_days", 256)   # slow models use smaller chunks, so a stop loses less
        parts, saved = [], time.monotonic()
        bar = tqdm(total=len(todo), desc=f"  {asset} {model.run_name}", unit="day", disable=not progress,
                   dynamic_ncols=True, mininterval=1.0)
        for s in range(0, len(todo), step):
            chunk = todo[s:s + step]
            parts.append(pd.DataFrame(model.predict_batch([tl.view(d) for d in chunk], asset),
                                      index=chunk, columns=COLS))
            bar.update(len(chunk))
            last = s + step >= len(todo)
            if last or time.monotonic() - saved > SAVE_EVERY:  # save progress as it goes (resume)
                done = pd.concat([cached] + parts) if len(cached) else pd.concat(parts)
                done = done[~done.index.duplicated(keep="last")].sort_index()
                done.index.name = "date"
                store.write_csv(done, p, {"settings": settings_key, "assets": sorted(used),
                                          "until": str(done.index.max().date()),
                                          "data": data_hash(tl, used, done.index.max(), macro),
                                          "months": month_hashes(tl, used, done.index.max(), macro)})
                saved = time.monotonic()
        bar.close()
        cached = done
    return cached.reindex(days)


# ---------- journal ----------

def forecast_all(models, tl, fresh=False, progress=True):
    """Every model on every asset, ONE MODEL AT A TIME: a model forecasts all assets, then its network is
    freed (model.unload) before the next one loads, so only one big model is ever in memory.
    Returns {(run, asset): forecasts}."""
    out = {}
    models = dict(sorted(models.items(), key=lambda x: x[1].kind == "stack"))   # judges last: they read the others
    for i, (run, model) in enumerate(models.items(), 1):
        if progress:
            print(f"[{i}/{len(models)}] {run}")
        try:
            for asset in tl.assets:
                out[run, asset] = forecast_history(model, tl, asset, fresh=fresh, progress=progress)
        finally:
            model.unload()
    return out


def build_journal(models, tl, fresh=False, progress=True, horizon=None):
    h = horizon or settings.HORIZON_DAYS
    forecasts = forecast_all(models, tl, fresh=fresh, progress=progress)
    parts = []
    for asset in tl.assets:
        closes = tl.view(tl.calendar()[-1]).closes(asset)
        days = closes.index
        end_date = pd.Series(days, index=days).shift(-h)     # the day its 20 days are over
        actual = closes.shift(-h) / closes - 1                # what really happened (scoring only)
        situation = regime_table(closes)                      # past only
        stretch = stretch_table(closes)["stretch"]            # past only: the stretch as known that day
        for run, model in models.items():
            fc = forecasts[run, asset].dropna(subset=["p50"])
            d = fc.index
            cutoff = pd.Timestamp(str(model.seen_until)) if model.seen_until else None
            parts.append(pd.DataFrame({
                "date": d, "end_date": end_date.reindex(d).to_numpy(), "asset": asset, "model": run,
                "kind": model.kind, "situation": situation.reindex(d).to_numpy(), "stretch": stretch.reindex(d).to_numpy(),
                "price": closes.reindex(d).to_numpy(),
                "p10": fc["p10"].to_numpy(), "p50": fc["p50"].to_numpy(), "p90": fc["p90"].to_numpy(),
                "actual": actual.reindex(d).to_numpy(),
                "seen": (d <= cutoff) if cutoff is not None else False}))
    return pd.concat(parts, ignore_index=True)


def known(journal, as_of=None, unseen_only=True):
    """Forecasts whose result was known on as_of (their 20 days had ended)."""
    j = journal[journal["actual"].notna() & journal["end_date"].notna()]
    if unseen_only:
        j = j[~j["seen"].astype(bool)]
    if as_of is not None:
        j = j[j["end_date"] <= pd.Timestamp(as_of)]
    return j


def dev_window(journal):
    """Development period: after the warm-up, finished before the holdout (unless SHOW_HOLDOUT)."""
    j = journal[journal["date"] > pd.Timestamp(settings.WARMUP_UNTIL)]
    if not settings.SHOW_HOLDOUT:
        j = j[j["end_date"] < pd.Timestamp(settings.HOLDOUT_FROM)]
    return j


# ---------- skill book: month by month, only what was known then ----------

BOOK_VERSION = 3   # 2: rows per asset and situation (core/pooling.py), 3: stretches. Change it = rebuilt
BOOK_INPUTS = ["date", "asset", "model", "p10", "p50", "p90", "actual", "situation"]
SKILL = ["periods", "ic", "trust", "direction", "always_up", "edge", "skill", "range_hit",
         "warn_periods", "warn_hit", "warn_base", "warn_edge", "warn_trust", "warn_move",
         "rise_periods", "rise_hit", "rise_base", "rise_edge", "rise_trust", "rise_move"]


def skillbook(journal, tl, progress=True, horizon=None):
    """horizon None = the main one (skillbook.csv); else skillbook_h<horizon>.csv."""
    h = horizon or settings.HORIZON_DAYS
    rows = known(journal).sort_values(["end_date", "asset", "model", "date"]).reset_index(drop=True)
    p = store.path("battle", horizon_name("skillbook", h) + ".csv")
    months = tl.month_starts(rows["end_date"].min(), tl.calendar()[-1])
    old, start_after = None, None
    if p.exists():
        m = store.read_meta(p)
        if m.get("until") and m.get("columns") == SKILL and m.get("version") == BOOK_VERSION and m.get("inputs") == store.fingerprint(
                rows[rows["end_date"] <= pd.Timestamp(m["until"])][BOOK_INPUTS + (["stretch"] if "stretch" in rows else [])]):
            old, start_after = store.read_csv(p, dates=("as_of",)), pd.Timestamp(m["until"])
        elif progress:
            print("  skill book: inputs changed, building it again")
    todo = months if start_after is None else months[months > start_after]
    new = _snapshots(rows, todo, h, progress) if len(todo) else pd.DataFrame()
    book = pd.concat([old, new], ignore_index=True) if old is not None else new
    if len(todo):
        until = todo[-1]
        store.write_csv(book, p, {"until": str(until.date()), "columns": SKILL, "version": BOOK_VERSION, "inputs": store.fingerprint(
            rows[rows["end_date"] <= until][BOOK_INPUTS + (["stretch"] if "stretch" in rows else [])])},
            index=False)
    return book


def _snapshots(rows, months, h, progress):
    end = rows["end_date"].to_numpy()
    arr = {c: rows[c].to_numpy() for c in ("p10", "p50", "p90", "actual")}
    df = pd.DataFrame({"model": rows["model"].to_numpy(), "situation": rows["situation"].fillna("unknown").to_numpy(),
                       "asset": rows["asset"].to_numpy()})
    if "stretch" in rows:   # stretches go in the same column, as "stretch:<kind>" (core/stretches.py)
        st = ("stretch:" + rows["stretch"].astype(str)).where(rows["stretch"].notna(), "unknown").to_numpy()
        df = pd.concat([df, df.assign(situation=st)], keys=[0, 1]).reset_index(level=0, drop=True)
    first = df.iloc[:len(rows)]   # every forecast once (the second copy only relabels it by stretch)
    groups = [(r, "all", "all", np.sort(i)) for r, i in first.groupby("model").indices.items()]
    n_rows = len(rows)   # positions in the stretch copy map back to the same forecast (each label is in one copy)
    groups += [(r, g, "all", np.sort(i % n_rows)) for (r, g), i in df.groupby(["model", "situation"]).indices.items() if g != "unknown"]
    groups += [(r, "all", a, np.sort(i)) for (r, a), i in first.groupby(["model", "asset"]).indices.items()]
    groups += [(r, g, a, np.sort(i % n_rows)) for (r, g, a), i in df.groupby(["model", "situation", "asset"]).indices.items()
               if g != "unknown"]   # per asset AND situation: core/pooling.py combines these per asset
    out = []
    for k, d in enumerate(months):
        n = int(np.searchsorted(end, np.datetime64(d), side="right"))  # results known on d
        for model, situation, scope, pos in groups:
            m = int(np.searchsorted(pos, n))
            if m:
                g = pd.DataFrame({c: arr[c][pos[:m]] for c in arr})
                s = metrics.score(g, h)
                out.append({"as_of": d, "model": model, "situation": situation, "scope": scope,
                            **{c: s[c] for c in SKILL}})
        if progress:
            print(f"\r  skill book: {k + 1}/{len(months)} months", end="", flush=True)
    if progress:
        print()
    return pd.DataFrame(out)


# ---------- leaderboard and winners ----------

def verdict(r):
    if pd.isna(r["periods"]) or r["periods"] < settings.MIN_PERIODS:
        return "not enough unseen results yet"
    if pd.isna(r["ic"]) or r["ic"] <= 0:
        v = "no edge"
    elif r["ic"] > 0.25 and r["trust"] >= settings.MIN_TRUST:
        v = "too good to be true? check for memory"
    elif r["trust"] >= settings.MIN_TRUST and r["luck_beats_it"] < 0.05 and r["skill"] > 0:
        v = "edge, confirm in the holdout"
    elif r["trust"] >= settings.MIN_TRUST:
        v = "maybe, could still be luck"
    else:
        v = "no proven edge"
    if r.get("kind") == "stack" and pd.notna(r.get("best_single_ic")):
        v += (f" | beats the best single model (IC {r['ic']:+.3f} vs {r['best_single_ic']:+.3f})" if r["ic"] > r["best_single_ic"]
              else f" | does NOT beat the best single model (IC {r['ic']:+.3f} vs {r['best_single_ic']:+.3f})")
    if r["range_hit"] < 0.7:
        v += " | range too narrow"
    elif r["range_hit"] > 0.9:
        v += " | range too wide"
    return v


def leaderboard(journal, models, horizon=None):
    h = horizon or settings.HORIZON_DAYS
    rows = known(dev_window(journal))
    if rows.empty:
        return pd.DataFrame()
    t = metrics.score_table(rows, ["model"], h)
    years = metrics.ic_by_year(rows.rename(columns={"model": "run"}))
    boot = metrics.bootstrap(rows.rename(columns={"model": "run"}))
    shift = max(60, 2 * h)   # luck shifts results further than a forecast overlaps with itself
    luck = metrics.luck_test(rows.rename(columns={"model": "run"}), min_shift=shift)
    t["years_positive"] = years["positive"].reindex(t.index)
    t["chunks_low"] = boot["low_5"].reindex(t.index)
    t["luck_beats_it"] = luck["luck_chance"].reindex(t.index)
    for kind in ("warn", "rise"):   # the warning record gets its own luck test
        cl = metrics.call_luck_test(rows.rename(columns={"model": "run"}), kind, min_shift=shift)
        t[f"{kind}_luck"] = cl["luck_chance"].reindex(t.index) if len(cl) else np.nan
    t["kind"] = [models[m].kind if m in models else "" for m in t.index]
    singles = t[(t["kind"] != "stack") & ~t.index.isin(settings.EXCLUDE)]["ic"]
    t["best_single_ic"] = singles.max() if singles.notna().any() else np.nan
    t["verdict"] = [verdict(r) for _, r in t.iterrows()]
    t = t.sort_values(["ic"], ascending=False, na_position="last")
    t.insert(0, "rank", range(1, len(t) + 1))
    t.index.name = "model"
    return t


def winners(journal):
    """Per asset, first trading day of each month: whose middle guess came closest to the real price.
    Pretrained models only compete after their seen_until date, like everywhere else."""
    j = journal[~journal["seen"].astype(bool)].dropna(subset=["actual", "p50"])
    out = []
    for asset, g in j.groupby("asset"):
        d = pd.DatetimeIndex(g["date"].unique()).sort_values()
        firsts = pd.Series(d, index=d).groupby([d.year, d.month]).min()
        sub = g[g["date"].isin(firsts.to_numpy())].copy()
        sub["miss"] = (sub["p50"] - sub["actual"]).abs()
        for day, gg in sub.groupby("date"):
            w = gg.loc[gg["miss"].idxmin()]
            out.append({"date": day, "asset": asset, "end_date": w["end_date"], "winner": w["model"],
                        "winner_guess": w["p50"], "actual": w["actual"], "winner_miss": w["miss"],
                        "players": len(gg)})
    return pd.DataFrame(out)


# ---------- no-future check ----------

def scrambled(tl, cutoff, seed=0):
    """Copy of the timeline where every price after cutoff is replaced by garbage."""
    rng = np.random.default_rng(seed)
    last = tl.view(tl.calendar()[-1])
    fake = {}
    for a in tl.assets:
        b = last.bars(a)
        later = b.index > cutoff
        for c in ("open_eur", "close_eur"):
            b.loc[later, c] = b.loc[later, c] * rng.uniform(0.5, 1.5, later.sum())
        fake[a] = b
    extras = {}
    for n, ser in tl.extras().items():
        later = ser.index > cutoff
        ser[later] = ser[later] * rng.uniform(0.5, 1.5, later.sum())
        extras[n] = ser
    return Timeline(fake, extras)


def check_no_future(model, tl, asset, samples=3):
    """Forecasts on a date must be identical when everything after it is garbage. The real one is made
    in a batch together with later days, like the walk-forward does it, so batches can't leak either."""
    days = eligible_days(tl, asset, model, use_first_day=False)   # first_day only saves time, check every model
    picks = days[np.linspace(len(days) // 4, len(days) - 30, samples).astype(int)]
    out = []
    for d in picks:
        fake = scrambled(tl, d)
        later = days[days > d][[4, 19]] if (days > d).sum() > 20 else days[days > d][:0]
        a = model.predict_batch([tl.view(x) for x in [d, *later]], asset)[0]
        b = model.predict(fake.view(d), asset)
        out.append((d, bool(np.allclose(a, b, atol=1e-5, equal_nan=True))))
    return out