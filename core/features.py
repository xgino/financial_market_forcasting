# financial_market_forcasting1/core/features.py
"""
Core, stage 1: features = simple numbers that describe the price line on a day.
Every feature on day D only uses closes up to and including D (rolling windows look back, never ahead).
Used by trained models now, and by the scoreboard later to label the situation (trend up, wild swings, ...).

  from core.features import feature_table, features_at
  table = feature_table(view.closes("gold"))   # one row per past day
  today = features_at(view, "gold")            # just today's row
"""
import numpy as np
import pandas as pd

NEEDS_DAYS = 260  # longest look back below is 252 days
VERSION = 3       # change this when the features change: your own models retrain automatically


def rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def feature_table(closes):
    """closes: Series of past closes. Returns one row of features per day."""
    lr = np.log(closes).diff()
    f = pd.DataFrame(index=closes.index)
    for n in (5, 20, 60, 120, 252):
        f[f"ret_{n}d"] = closes / closes.shift(n) - 1            # how much it moved over the last n days
    f["swings_20d"] = lr.rolling(20).std()                       # how wild the last month was
    f["swings_60d"] = lr.rolling(60).std()
    f["swings_ratio"] = f["swings_20d"] / f["swings_60d"]        # > 1 = getting wilder
    # the same moves in units of the asset's own swings, so a calm and a wild asset compare fairly
    for n in (20, 60, 252):
        f[f"ret_{n}d_z"] = np.log1p(f[f"ret_{n}d"]) / (f["swings_60d"] * np.sqrt(n))
    f["vs_avg_50d"] = closes / closes.rolling(50).mean() - 1     # above or below the 50 day average
    f["vs_avg_200d"] = closes / closes.rolling(200).mean() - 1
    f["drop_from_1y_high"] = closes / closes.rolling(252, min_periods=50).max() - 1
    f["rsi_14"] = rsi(closes)                                    # momentum 0-100
    return f


def features_at(view, asset):
    """Today's features for one asset."""
    return feature_table(view.closes(asset, NEEDS_DAYS + 10)).iloc[-1]


def macro_table(view, n=400):
    """Macro features per day from the FRED series (settings.MACRO_SERIES), only values known by then.
    Same for every asset on a day. Empty when no macro data was downloaded."""
    cols = {}
    names = set(view.extra_names)
    if "real_rate" in names:
        r = view.extra("real_rate", n)
        cols["real_rate"], cols["real_rate_chg_60d"] = r, r - r.shift(60)
    if "rate10" in names:
        r = view.extra("rate10", n)
        cols["rate10_chg_20d"] = r - r.shift(20)
    if "curve" in names:
        r = view.extra("curve", n)
        cols["curve"], cols["curve_chg_60d"] = r, r - r.shift(60)
    if "dollar" in names:
        r = view.extra("dollar", n)
        cols["dollar_ret_20d"], cols["dollar_ret_60d"] = r / r.shift(20) - 1, r / r.shift(60) - 1
    if "vix" in names:
        r = view.extra("vix", n)
        cols["vix"], cols["vix_vs_1y"] = r, r / r.rolling(252, min_periods=60).mean() - 1
    if not cols:
        return None
    return pd.DataFrame(cols).sort_index()


def full_table(view, asset, n=None):
    """Price features + macro features (the last known macro value on or before each day)."""
    f = feature_table(view.closes(asset, n))
    m = macro_table(view, None if n is None else n + 300)
    if m is not None and len(m):
        f = f.join(m.reindex(f.index, method="ffill"))
    return f


# plain words for every feature, and how to show its value ("%" or "num"), for the explanations
DESCRIBE = {
    "ret_5d": ("move over the last week", "%"), "ret_20d": ("move over the last month", "%"),
    "ret_60d": ("move over the last 3 months", "%"), "ret_120d": ("move over the last 6 months", "%"),
    "ret_252d": ("move over the last year", "%"), "swings_20d": ("daily swings last month", "%"),
    "swings_60d": ("daily swings last 3 months", "%"), "swings_ratio": ("swings now vs 3 months (1 = normal)", "num"),
    "vs_avg_50d": ("distance from the 50 day average", "%"), "vs_avg_200d": ("distance from the 200 day average", "%"),
    "drop_from_1y_high": ("drop from the 1 year high", "%"), "rsi_14": ("RSI momentum (0-100)", "num"),
    "ret_20d_z": ("last month's move vs its usual swings", "num"), "ret_60d_z": ("3 month move vs its usual swings", "num"),
    "ret_252d_z": ("1 year move vs its usual swings", "num"), "real_rate": ("US real interest rate", "num"),
    "real_rate_chg_60d": ("change in the real rate, 3 months", "num"), "rate10_chg_20d": ("change in the 10y rate, 1 month", "num"),
    "curve": ("yield curve (10y minus short rate)", "num"), "curve_chg_60d": ("change in the yield curve, 3 months", "num"),
    "dollar_ret_20d": ("dollar move last month", "%"), "dollar_ret_60d": ("dollar move last 3 months", "%"),
    "vix": ("VIX fear index", "num"), "vix_vs_1y": ("VIX vs its last year", "%"),
}


def describe(name, value):
    words, kind = DESCRIBE.get(name, (name, "num"))
    return words, (f"{value:+.1%}" if kind == "%" else f"{value:.2f}")