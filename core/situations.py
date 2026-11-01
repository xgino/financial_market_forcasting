# paper-fund-v2/core/situations.py
"""
Core, stage 1: labels the situation an asset is in on each day, using only the past.

  trend  : up / flat / down   how strong the last ~6 months move was, compared to its normal swings
  swings : calm / wild        are the last 20 days wilder than usual for THIS asset (its own last 2 years)

Together 6 situations: up-calm, up-wild, flat-calm, flat-wild, down-calm, down-wild.
Kept to 6 on purpose: more boxes = fewer examples per box = more luck in the scores.
Everything is relative to the asset itself, so gold, silver and bonds are labeled the same way.

  from core.situations import regime_table, regime_at
  labels = regime_table(view.closes("gold"))   # one label per day
  today = regime_at(view, "gold")              # e.g. "up-calm"
"""
import numpy as np
import pandas as pd

import settings

DEFAULTS = settings.SITUATIONS
REGIMES = ["up-calm", "up-wild", "flat-calm", "flat-wild", "down-calm", "down-wild"]


def regime_table(closes, settings=None):
    s = {**DEFAULTS, **(settings or {})}
    lr = np.log(closes).diff()
    td = s["trend_days"]
    # trend strength: the move over trend_days, in units of how much it normally moves in that time
    strength = np.log(closes / closes.shift(td)) / (lr.rolling(60).std() * np.sqrt(td))
    trend = pd.Series(np.select([strength > s["trend_cutoff"], strength < -s["trend_cutoff"]],
                                ["up", "down"], "flat"), index=closes.index)
    swings_now = lr.rolling(s["swing_days"]).std()
    typical = swings_now.rolling(s["swing_lookback"], min_periods=252).median()
    swings = pd.Series(np.where(swings_now > typical, "wild", "calm"), index=closes.index)
    label = trend + "-" + swings
    return label.where(strength.notna() & typical.notna())  # unknown at the start


def regime_at(view, asset, settings=None):
    s = {**DEFAULTS, **(settings or {})}
    need = max(s["trend_days"] + 60, s["swing_lookback"] + s["swing_days"]) + 5
    return regime_table(view.closes(asset, need), settings).iloc[-1]