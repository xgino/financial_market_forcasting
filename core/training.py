# financial_market_forcasting1/core/training.py
"""
Shared training loop for your own models (03_training). Walk-forward: for every cutoff (1 January of each
year) a fresh checkpoint is trained with ONLY results that had finished by that date. Tuning happens
inside that training data only, on time-ordered folds.

A trainer file only has to provide fit(rows, cfg, horizon, log) -> (artifact, info). This loop does the
rest: which cutoffs are missing, saving checkpoints + index.csv, the live log for the dashboard, and
training everything again when the settings, the features, the assets or the macro data change.

  data/checkpoints/<name>/<cutoff>.pkl + index.csv
  data/training/<name>/log.csv   (one row per try or per epoch, the dashboard reads it live)
"""
import pickle
import time

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import settings
from core import features, store
from core.features import full_table


def training_rows(view, horizon, min_days=300):
    """Features (price + macro) and the 20 day target for every asset the view has. The view stops at
    the cutoff, so the last 20 days have no target yet and drop out by themselves."""
    parts = []
    for asset in view.assets:
        closes = view.closes(asset)
        if len(closes) < min_days:
            continue
        X = full_table(view, asset)
        X["target"] = closes.shift(-horizon) / closes - 1
        X["asset"] = asset
        parts.append(X)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).dropna(subset=["target", "ret_60d"]).sort_index()


class KeepInformative(BaseEstimator, TransformerMixin):
    """Drops columns that are empty or constant in the data it's fitted on. Very old years have no macro
    data at all (no VIX before 1990), and some model libraries crash on a column with no values."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_ = [c for c in X.columns if X[c].nunique(dropna=True) >= 2]
        return self

    def transform(self, X):
        return pd.DataFrame(X).reindex(columns=self.keep_)


class Logger:
    """Appends rows to data/training/<name>/log.csv right away, so the dashboard can follow live."""

    def __init__(self, name, fresh):
        self.path = settings.DATA / "training" / name / "log.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = pd.read_csv(self.path).to_dict("records") if self.path.exists() and not fresh else []
        self.cutoff = None

    def __call__(self, **row):
        self.rows.append({"cutoff": self.cutoff, **row})
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


def walk_forward(name, cfg, tl, fit, progress=True, fresh=False):
    h = settings.HORIZON_DAYS
    folder = settings.DATA / "checkpoints" / name
    folder.mkdir(parents=True, exist_ok=True)
    key = store.fingerprint(cfg, h, features.VERSION, sorted(tl.assets), sorted(tl.extra_names))
    index_path = folder / "index.csv"
    index = pd.read_csv(index_path) if index_path.exists() and not fresh else pd.DataFrame()
    if len(index) and (index["settings"] != key).any():
        print("  settings, features, assets or macro data changed: training every checkpoint again")
        index, fresh = pd.DataFrame(), True
    log = Logger(name, fresh)
    done = set(index["cutoff"]) if len(index) else set()
    cutoffs = pd.date_range(cfg["first_cutoff"], tl.calendar()[-1], freq=f"{12 * cfg.get('every_years', 1)}MS")
    new = 0
    for c in cutoffs:
        if str(c.date()) in done:
            continue
        t0 = time.time()
        rows = training_rows(tl.view(c), h)
        if len(rows) < cfg.get("min_rows", 1000):
            if progress:
                print(f"  {c.date()}: only {len(rows)} rows, skipped")
            continue
        log.cutoff = str(c.date())
        artifact, info = fit(rows, cfg, h, log)
        with open(folder / f"{c.date()}.pkl", "wb") as f:
            pickle.dump({**artifact, "cutoff": str(c.date())}, f)
        index = pd.concat([index, pd.DataFrame([{"cutoff": str(c.date()), "rows": len(rows),
                                                 "trained_until": str(rows.index.max().date()), **info,
                                                 "seconds": round(time.time() - t0, 1), "settings": key}])],
                          ignore_index=True)
        index.to_csv(index_path, index=False)
        new += 1
        if progress:
            print(f"  {c.date()}: {len(rows)} rows, {info} ({time.time() - t0:.0f}s)")
    return {"checkpoints": len(index), "new": new}