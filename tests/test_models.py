# financial_market_forcasting1/tests/test_models.py
"""Every model and every trainer must be unable to see the future. Run with: python main.py test"""
from pathlib import Path

import numpy as np
import pandas as pd

import settings
from core import battle
from core.contracts import Model
from core.plugins import load_file, load_players
from core.timeline import Timeline

synthetic = load_file(Path(__file__).with_name("synthetic.py"))


def usable(m):
    """Skip models whose library isn't installed (pretrained) or that have nothing trained yet (gbm)."""
    try:
        if m.kind == "pretrained":
            __import__(next(lib for key, lib in (("chronos", "chronos"), ("timesfm", "timesfm"), ("tirex", "tirex"),
                                                 ("kronos", "torch"), ("moirai", "uni2ts"), ("timemoe", "torch"))
                       if key in m.run_name))
        if m.run_name == "gbm" and m.index.empty:
            return False
        if m.run_name in ("ets", "theta"):
            __import__("statsforecast")
        return True
    except ImportError:
        return False


def test_no_model_sees_the_future():
    tl = synthetic.timeline()
    models = load_players(settings.ROOT / "04_models", Model, settings.HORIZON_DAYS)
    checked = []
    for name, m in models.items():
        if not usable(m):
            continue
        for d, same in battle.check_no_future(m, tl, "asset0", samples=2):
            assert same, f"{name} changed its forecast for {d.date()} when the future changed"
        checked.append(name)
    assert checked, "no model could be checked"
    print("    checked:", ", ".join(checked))


def test_training_cannot_see_the_future():
    """gbm and linear: trained on a cutoff, with prices AND macro after it replaced by garbage, the
    trained model must be exactly the same."""
    from core.training import training_rows

    tl = synthetic.timeline()
    cutoff = tl.calendar()[1800]
    fake = battle.scrambled(tl, cutoff)
    real_rows = training_rows(tl.view(cutoff), settings.HORIZON_DAYS)
    fake_rows = training_rows(fake.view(cutoff), settings.HORIZON_DAYS)
    assert real_rows.index.max() <= cutoff and "vix" in real_rows.columns, "macro features should be in training"
    noop = lambda **kw: None
    for name, small in (("gbm", {"grid": {"max_depth": [2], "learning_rate": [0.1], "max_iter": [30]}, "cv_folds": 2}),
                        ("linear", {"alphas": [1, 100], "cv_folds": 2})):
        trainer = load_file(settings.ROOT / "03_training" / f"{name}.py")
        cfg = {**trainer.SETTINGS, **small}
        a, _ = trainer.fit(real_rows, cfg, settings.HORIZON_DAYS, noop)
        b, _ = trainer.fit(fake_rows, cfg, settings.HORIZON_DAYS, noop)
        X = real_rows[a["columns"]].tail(50)
        if name == "gbm":
            same = all(np.allclose(m1.predict(X), m2.predict(X)) for m1, m2 in zip(a["models"], b["models"]))
        else:
            same = np.allclose(a["model"].predict(X), b["model"].predict(X))
        assert same, f"{name}: training changed when only the future changed"


def test_macro_is_hidden_until_known():
    tl = synthetic.timeline()
    d = tl.calendar()[2000]
    v = tl.view(d)
    assert all(v.extra(n).index.max() <= d for n in tl.extra_names)
    fake = battle.scrambled(tl, d)
    assert all(v.extra(n).equals(fake.view(d).extra(n)) for n in tl.extra_names)


def test_choices_cannot_see_the_future():
    """Garbage in every forecast after a date, every result not known yet and every later situation:
    the skill book and every selector's choices on or before that date must not change."""
    import pandas as pd

    from core import selection

    rng = np.random.default_rng(1)
    days = pd.bdate_range("2006-01-02", "2014-12-31")
    rows = []
    for asset in ("a", "b"):
        for model in ("m1", "m2", "m3", "naive"):
            p50 = np.zeros(len(days)) if model == "naive" else rng.normal(0, 0.03, len(days))
            rows.append(pd.DataFrame({"date": days, "end_date": pd.Series(days).shift(-20).to_numpy(), "asset": asset,
                                      "model": model, "situation": rng.choice(["up-calm", "down-wild"], len(days)),
                                      "stretch": rng.choice(["up-fast", "sideways-calm"], len(days)),
                                      "price": 100.0, "p10": p50 - 0.05, "p50": p50, "p90": p50 + 0.05,
                                      "actual": rng.normal(0, 0.04, len(days)), "seen": False}))
    j = pd.concat(rows, ignore_index=True)
    j.loc[j["end_date"].isna(), "actual"] = np.nan
    cutoff = pd.Timestamp("2011-06-30")
    fake = j.copy()
    later = fake["date"] > cutoff
    fake.loc[later, ["p10", "p50", "p90"]] = rng.normal(0, 0.05, (later.sum(), 3))
    fake.loc[later, "situation"] = "up-calm"
    fake.loc[later, "stretch"] = "down-sharp"
    unknown = ~(fake["end_date"] <= cutoff)
    fake.loc[unknown, "actual"] = rng.normal(0, 0.05, unknown.sum())

    def book(jj):
        k = battle.known(jj).sort_values(["end_date", "asset", "model", "date"]).reset_index(drop=True)
        months = pd.DatetimeIndex(pd.Series(days, index=days).groupby([days.year, days.month]).min())
        return battle._snapshots(k, months[months >= k["end_date"].min()], 20, False)

    tl = Timeline({"a": synthetic.prices(seed=0), "b": synthetic.prices(seed=1)}, synthetic.macro())   # similar days
    selectors = load_players(settings.ROOT / "05_selection", selection.Selector)
    a = selection.run(j, book(j), selectors, tl, progress=False)
    selectors = load_players(settings.ROOT / "05_selection", selection.Selector)
    b = selection.run(fake, book(fake), selectors, battle.scrambled(tl, cutoff), progress=False)
    cols = ["selector", "asset", "date", "p10", "p50", "p90", "main_model", "reason", "top_models"]
    a, b = a[a["date"] <= cutoff][cols].reset_index(drop=True), b[b["date"] <= cutoff][cols].reset_index(drop=True)
    assert len(a) and a.equals(b), "a choice changed when only the future changed"
    assert a[a["selector"] == "best_in_stretch"]["reason"].str.contains("stretch").all()
    sim = a[a["selector"] == "best_in_similar_days"]
    assert len(sim) and sim["reason"].str.contains("similar days").all(), "best_in_similar_days didn't get its similar days"

def test_cache_redoes_only_from_the_first_changed_month():
    """Old prices changed after a date: cached forecasts before that month are kept, from it on redone."""
    tl = synthetic.timeline()
    cal = tl.calendar()
    cutoff, end = cal[1500], cal[-1]
    fake = battle.scrambled(tl, cutoff)
    old = battle.month_hashes(tl, ["asset0"], end, macro=True)
    cut = battle.first_changed_day(old, battle.month_hashes(fake, ["asset0"], end, macro=True))
    first_changed = cal[cal > cutoff][0]
    assert cut == pd.Timestamp(first_changed.strftime("%Y-%m-01")), f"redo from {cut}, expected {first_changed}"
    assert battle.first_changed_day(old, battle.month_hashes(tl, ["asset0"], end, macro=True)) is None
    # a dividend: Yahoo multiplies every price before it by the same factor, the moves stay the same
    b = tl.view(end).bars("asset0")
    div = b.index[2000]
    b.loc[b.index < div, ["open_eur", "close_eur"]] *= 0.99
    rescaled = Timeline({**{a: tl.view(end).bars(a) for a in tl.assets}, "asset0": b}, tl.extras())
    cut = battle.first_changed_day(old, battle.month_hashes(rescaled, ["asset0"], end, macro=True))
    assert cut == pd.Timestamp(div.strftime("%Y-%m-01")), f"a dividend in {div.date()} should only redo its month, got {cut}"


def test_similar_days_only_use_known_results():
    """Similar days for date D: only days whose 20 days had ended by D, and garbage after D (prices, macro,
    forecasts, results) changes nothing about which days are picked or how models score on them."""
    from core.similar import SimilarDays

    tl = synthetic.timeline()
    days = tl.calendar()
    rng = np.random.default_rng(0)
    rows = []
    for a in tl.assets:
        c = tl.view(days[-1]).closes(a)
        actual = (c.shift(-20) / c - 1)
        end = pd.Series(c.index, index=c.index).shift(-20)
        for m in ("m1", "m2"):
            rows.append(pd.DataFrame({"date": c.index, "end_date": end.to_numpy(), "asset": a, "model": m,
                                      "p50": actual.to_numpy() * (0.5 if m == "m1" else 0) + rng.normal(0, 0.03, len(c)),
                                      "actual": actual.to_numpy(), "seen": False}))
    j = pd.concat(rows, ignore_index=True)
    d = days[1800]
    sd = SimilarDays(j, tl)
    idx = sd.neighbours("asset0", d, 150)
    a = sd._a["asset0"]
    assert len(idx) == 150 and (a["end"][idx] <= np.datetime64(d)).all(), "a similar day whose result wasn't known yet"
    real = sd.scores("asset0", d, 150)

    fake_j = j.copy()
    later = fake_j["date"] > d
    fake_j.loc[later, ["p50", "actual"]] = rng.normal(0, 1, (later.sum(), 2))
    fake_j.loc[fake_j["end_date"] > d, "actual"] = rng.normal(0, 1, (fake_j["end_date"] > d).sum())
    fake = SimilarDays(fake_j, battle.scrambled(tl, d))
    assert (fake.neighbours("asset0", d, 150) == idx).all(), "garbage after D changed which days are similar"
    pd.testing.assert_frame_equal(real, fake.scores("asset0", d, 150))


def _judge(source, **cfg):
    from core.stacking import Judge

    j = Judge(settings.HORIZON_DAYS, {"min_rows": 500, "max_iter": 40, **cfg})
    j.source, j.run_name = source, "judge"
    return j


def test_judge_cannot_see_the_future():
    """Garbage after D in the prices, the macro AND the other models' forecasts: the judge's forecast for D
    is the same (it only trains on results that ended before D's month)."""
    tl = synthetic.timeline()
    rng = np.random.default_rng(3)
    src = {}
    for a in tl.assets:
        c = tl.view(tl.calendar()[-1]).closes(a)
        future = np.log(c.shift(-20) / c).fillna(0)
        src[a] = pd.DataFrame({"m1": future * 0.3 + rng.normal(0, 0.02, len(c)), "m2": rng.normal(0, 0.02, len(c))},
                              index=c.index)
    d = tl.calendar()[2000]
    real = _judge(src).predict_batch([tl.view(x) for x in [d, *tl.calendar()[2001:2003]]], "asset0")[0]
    fake_src = {a: t.mask(np.repeat((t.index > d)[:, None], t.shape[1], axis=1),
                          pd.DataFrame(rng.normal(0, 1, t.shape), index=t.index, columns=t.columns)) for a, t in src.items()}
    fake = _judge(fake_src).predict(battle.scrambled(tl, d).view(d), "asset0")
    assert np.isfinite(real).all() and np.allclose(real, fake), f"{real} vs {fake}"


def test_judge_ignores_memory_years():
    """A pretrained model's guesses up to its seen_until count as 'no opinion' (0)."""
    j = _judge({"gold": pd.DataFrame({"m": [0.05, 0.05, 0.05]}, index=pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]))})
    j._inputs = {"m": "2021-01-01"}
    assert j._base("gold")["m"].tolist() == [0.0, 0.0, 0.05]


def test_judge_followed_only_while_it_beats_the_best_model():
    from core.selection import stack_ok

    kinds = {"judge": "stack", "a": "stats", "b": "ml"}
    rec = pd.DataFrame({"ic": [0.05, 0.08, 0.02]}, index=["judge", "a", "b"])
    assert not stack_ok("judge", rec, kinds), "IC 0.05 doesn't beat a's 0.08"
    assert stack_ok("judge", rec.assign(ic=[0.10, 0.08, 0.02]), kinds)
    assert stack_ok("a", rec, kinds), "only judges are gated"
    assert not stack_ok("judge", None, kinds), "no record yet: not followed"


def test_other_horizons_join_honestly():
    """settings.HORIZONS: trained (20 day) and slow models stay on the main horizon; the rest can't see the
    future on 5 or 60 days either, and a 5 day journal scores 5 day results."""
    tl = synthetic.timeline()
    for h in [x for x in settings.HORIZONS if x != settings.HORIZON_DAYS]:
        models = load_players(settings.ROOT / "04_models", Model, h)
        joined = {n for n, m in models.items() if m.supports(h)}
        assert not {"gbm", "linear", "kronos_small", "timemoe"} & joined, f"{h} days: {joined}"
        assert {"drift", "naive", "theta", "judge"} <= joined
        for name in ("drift", "analog", "theta", "garch"):
            m = models[name]
            assert m.cache_name == f"{name}_h{h}"
            for d, same in battle.check_no_future(m, tl, "asset0", samples=2):
                assert same, f"{name} on {h} days changed its forecast for {d.date()} when the future changed"
    assert battle.horizon_name("journal", 5) == "journal_h5" and battle.horizon_name("journal", settings.HORIZON_DAYS) == "journal"
    d = tl.calendar()[1500]
    width = {h: np.diff(load_players(settings.ROOT / "04_models", Model, h, only=["drift"])["drift"]
                        .predict(tl.view(d), "asset0")[[0, 2]])[0] for h in (5, 60)}
    assert 0 < width[5] < width[60], f"a 5 day range should be narrower than a 60 day one: {width}"


def test_stretches_only_use_the_past():
    """A day's stretch (label, start, length) is decided with data up to that day and never changes later."""
    from core.stretches import KINDS, stretch_table

    tl = synthetic.timeline()
    end = tl.calendar()[-1]
    real = stretch_table(tl.view(end).closes("asset0"))
    for d in tl.calendar()[[900, 1600, 2400]]:
        fake = stretch_table(battle.scrambled(tl, d).view(end).closes("asset0"))
        pd.testing.assert_frame_equal(real[real.index <= d], fake[fake.index <= d])
    assert set(real["stretch"].dropna()) <= set(KINDS) and real["stretch"].notna().mean() > 0.8
    assert real["stretch_start"].dropna().nunique() > 5, "a random walk should still get cut into several stretches"


def test_skillbook_scores_stretches_on_the_right_forecasts():
    from core import metrics

    rng = np.random.default_rng(5)
    days = pd.bdate_range("2010-01-01", periods=600)
    j = pd.DataFrame({"date": days, "end_date": days + pd.Timedelta(days=28), "asset": "a", "model": "m",
                      "situation": rng.choice(["up-calm", "down-wild"], 600), "stretch": rng.choice(["up-fast", "down-sharp"], 600),
                      "p10": -0.05, "p50": rng.normal(0, 0.03, 600), "p90": 0.05, "actual": rng.normal(0, 0.04, 600)})
    rows = j.sort_values(["end_date", "asset", "model", "date"]).reset_index(drop=True)
    book = battle._snapshots(rows, pd.DatetimeIndex([days[-1] + pd.Timedelta(days=60)]), 20, False)
    got = book[(book["situation"] == "stretch:up-fast") & (book["scope"] == "all")].iloc[0]
    want = metrics.score(rows[rows["stretch"] == "up-fast"], 20)
    assert abs(got["ic"] - want["ic"]) < 1e-12 and abs(got["periods"] - want["periods"]) < 1e-12
    allrow = book[(book["situation"] == "all") & (book["scope"] == "all")].iloc[0]
    assert abs(allrow["periods"] - 600 / 20) < 1e-12, "the stretch copy must not count forecasts twice"


def test_finetune_only_learns_from_before_its_cutoff():
    """Fine-tuning windows (context + 20 day target) all end before the checkpoint's cutoff, the held-back
    check windows come after the training ones with a gap, and a forecast uses the newest checkpoint on or
    before its day."""
    mod = load_file(settings.ROOT / "03_training" / "chronos_finetune.py")
    t = mod.ChronosFinetune({**mod.SETTINGS, "context_days": 100})
    tl = synthetic.timeline()
    cutoff = pd.Timestamp("2012-01-01")
    train, check = t._windows(tl, cutoff)
    h, c = settings.HORIZON_DAYS, 100

    def last_day(w):
        x, i = w
        a = next(a for a in tl.assets if len(tl.view(cutoff).closes(a)) >= len(x))
        return tl.view(cutoff - pd.Timedelta(days=1)).closes(a).index[i + c + h - 1]

    pick = lambda ws: [ws[k] for k in np.linspace(0, len(ws) - 1, 20).astype(int)]
    assert max(last_day(w) for w in pick(train)) < min(last_day(w) for w in pick(check)) < cutoff
    assert max(last_day(w) for w in check) < cutoff

    from core.pretrained import ChronosFinetuned

    m = ChronosFinetuned(settings.HORIZON_DAYS, {"trainer": "none"})
    m.index = pd.DataFrame({"cutoff": pd.to_datetime(["2025-01-01", "2026-01-01"])})
    assert m._cutoff(pd.Timestamp("2024-12-31")) is None
    assert m._cutoff(pd.Timestamp("2025-06-30")) == pd.Timestamp("2025-01-01")
    assert m._cutoff(pd.Timestamp("2026-01-01")) == pd.Timestamp("2026-01-01")
