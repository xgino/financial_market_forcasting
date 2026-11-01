# paper-fund-v2/tests/test_models.py
"""Every model and every trainer must be unable to see the future. Run with: python main.py test"""
from pathlib import Path

import numpy as np

import settings
from core import battle
from core.contracts import Model
from core.plugins import load_file, load_players

synthetic = load_file(Path(__file__).with_name("synthetic.py"))


def usable(m):
    """Skip models whose library isn't installed (pretrained) or that have nothing trained yet (gbm)."""
    try:
        if m.kind == "pretrained":
            __import__(next(lib for key, lib in (("chronos", "chronos"), ("timesfm", "timesfm"), ("tirex", "tirex"),
                                                 ("kronos", "torch")) if key in m.run_name))
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
                                      "price": 100.0, "p10": p50 - 0.05, "p50": p50, "p90": p50 + 0.05,
                                      "actual": rng.normal(0, 0.04, len(days)), "seen": False}))
    j = pd.concat(rows, ignore_index=True)
    j.loc[j["end_date"].isna(), "actual"] = np.nan
    cutoff = pd.Timestamp("2011-06-30")
    fake = j.copy()
    later = fake["date"] > cutoff
    fake.loc[later, ["p10", "p50", "p90"]] = rng.normal(0, 0.05, (later.sum(), 3))
    fake.loc[later, "situation"] = "up-calm"
    unknown = ~(fake["end_date"] <= cutoff)
    fake.loc[unknown, "actual"] = rng.normal(0, 0.05, unknown.sum())

    def book(jj):
        k = battle.known(jj).sort_values(["end_date", "asset", "model", "date"]).reset_index(drop=True)
        months = pd.DatetimeIndex(pd.Series(days, index=days).groupby([days.year, days.month]).min())
        return battle._snapshots(k, months[months >= k["end_date"].min()], 20, False)

    selectors = load_players(settings.ROOT / "05_selection", selection.Selector)
    a = selection.run(j, book(j), selectors, progress=False)
    selectors = load_players(settings.ROOT / "05_selection", selection.Selector)
    b = selection.run(fake, book(fake), selectors, progress=False)
    cols = ["selector", "asset", "date", "p10", "p50", "p90", "main_model", "reason", "top_models"]
    a, b = a[a["date"] <= cutoff][cols].reset_index(drop=True), b[b["date"] <= cutoff][cols].reset_index(drop=True)
    assert len(a) and a.equals(b), "a choice changed when only the future changed"