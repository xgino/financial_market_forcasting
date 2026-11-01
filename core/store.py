# paper-fund-v2/core/store.py
"""
Where everything is saved: plain CSV files in data/, so you can open any of them in Excel or VS Code.

A CSV that depends on other data gets a small .meta.json next to it with a fingerprint of what it was
made from. If those inputs change (repaired prices, a new setting), the fingerprint no longer matches
and the step makes the file again. That's how a CSV can't silently go out of date.

  data/prices/<asset>.csv              01_data
  data/macro/<name>.csv                01_data (FRED)
  data/features/<asset>.csv            02_features
  data/situations/<asset>.csv          02_features
  data/checkpoints/<model>/            03_training (trained models, one per cutoff date)
  data/training/<model>/log.csv        03_training (live progress, the dashboard reads it)
  data/forecasts/<asset>/<model>.csv   04_models (every forecast, every day)
  data/battle/*.csv                    04_models (journal, scores, leaderboard, winners)
"""
import hashlib
import json

import numpy as np
import pandas as pd

import settings

DATA = settings.DATA


def path(*parts):
    p = DATA.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ----- prices -----

def read_prices(asset):
    p = DATA / "prices" / f"{asset}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing, run step 1 first: python main.py 1")
    return pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()


# ----- fingerprints -----

def fingerprint(*things):
    """Short hash of anything: DataFrames, Series, dicts, strings."""
    h = hashlib.sha256()
    for t in things:
        if isinstance(t, (pd.DataFrame, pd.Series)):
            h.update(pd.util.hash_pandas_object(t, index=True).to_numpy().tobytes())
        elif isinstance(t, np.ndarray):
            h.update(np.ascontiguousarray(t).tobytes())
        else:
            h.update(json.dumps(t, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def meta_path(p):
    return p.with_name(p.stem + ".meta.json")


def read_meta(p):
    m = meta_path(p)
    return json.loads(m.read_text()) if m.exists() else {}


def write_csv(df, p, meta=None, index=True):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index, date_format="%Y-%m-%d")
    if meta is not None:
        meta_path(p).write_text(json.dumps(meta, indent=1, default=str))


def read_csv(p, dates=("date",), index_col=None):
    df = pd.read_csv(p, index_col=index_col)
    for c in dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    if index_col is not None:
        df.index = pd.to_datetime(df.index)
    return df


def read_macro():
    """{name: Series} from data/macro/*.csv, each indexed by the day the value became usable:
    its own date + settings.MACRO_LAG_DAYS trading days (values are published with a delay)."""
    out = {}
    folder = DATA / "macro"
    if not folder.exists():
        return out
    for f in sorted(folder.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["date"]).dropna()
        if df.empty:
            continue
        known = df["date"] + pd.offsets.BDay(settings.MACRO_LAG_DAYS)
        out[f.stem] = pd.Series(df["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(known))
    return out


def tag(name):
    """File name for results: '<name>_holdout' when the holdout is open, so normal results never mix with it."""
    return f"{name}_holdout" if settings.SHOW_HOLDOUT else name