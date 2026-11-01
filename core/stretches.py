# financial_market_forcasting1/core/stretches.py
"""
Pattern chunks: cut each price line into stretches where it behaved the same, instead of fixed windows.
The situations (core/situations.py) always look back 120 / 20 days; a stretch is as long as the behaviour lasts.

Walking forward day by day (change-point detection, online, so only data known that day):
  - the current stretch runs from its start to today
  - a new stretch starts (20 days ago) when the last `window` days clearly differ from the stretch before them:
      average move   z > settings.STRETCHES["z"]           (a different direction or speed)
      swings         ratio > settings.STRETCHES["swings"]  (much wilder or calmer)
  - the label on day D describes the stretch as it was known on D, and is never changed afterwards

The label: shape x size, measured against the asset's own normal swings (median of its last 2 years):
  up-steady / up-fast        moved up more than `move` normal swings over its length; fast = more than
  down-steady / down-sharp   `fast` normal swings per day (a sharp drop)
  sideways-calm / -wild      no clear move; wild = swings above normal
Kept to 6 on purpose, like the situations: more kinds = fewer examples per kind = more luck in the scores.

  stretch_table(closes)   per day: stretch (label), stretch_start, stretch_days
"""
import numpy as np
import pandas as pd

import settings

KINDS = ["up-steady", "up-fast", "down-steady", "down-sharp", "sideways-calm", "sideways-wild"]


def stretch_table(closes, cfg=None):
    s = {**settings.STRETCHES, **(cfg or {})}
    w = s["window"]
    c = closes.astype(float)
    r = np.log(c).diff().to_numpy()
    normal = (pd.Series(r, index=c.index).rolling(20).std()
              .rolling(504, min_periods=252).median().to_numpy())   # the asset's usual daily swings, past only
    n = len(r)
    label, start_at, length = np.full(n, None, dtype=object), np.full(n, -1), np.zeros(n, dtype=int)
    cs = np.concatenate([[0.0], np.nancumsum(np.nan_to_num(r))])
    cs2 = np.concatenate([[0.0], np.nancumsum(np.nan_to_num(r) ** 2)])
    seg = lambda a, b: (cs[b] - cs[a], cs2[b] - cs2[a], b - a)        # sum, sum of squares, count of r[a:b]
    start = 1                                                          # r[0] is NaN
    for t in range(1, n):
        end = t + 1
        if end - start >= 2 * w:                                       # enough for "before" and "recent"
            s1, q1, n1 = seg(start, end - w)
            s2, q2, n2 = seg(end - w, end)
            m1, m2 = s1 / n1, s2 / n2
            v1, v2 = max(q1 / n1 - m1 ** 2, 1e-12), max(q2 / n2 - m2 ** 2, 1e-12)
            sd = np.sqrt((q1 + q2) / (n1 + n2) - ((s1 + s2) / (n1 + n2)) ** 2)
            z = (m2 - m1) / (sd * np.sqrt(1 / n1 + 1 / n2)) if sd > 0 else 0.0
            if abs(z) > s["z"] or max(v2 / v1, v1 / v2) > s["swings"] ** 2:
                start = end - w                                        # the change began ~window days ago
        if not np.isfinite(normal[t]) or normal[t] <= 0:
            continue
        s1, q1, k = seg(start, end)
        m = s1 / k
        sd = np.sqrt(max(q1 / k - m ** 2, 0.0))
        moved = s1 / (normal[t] * np.sqrt(k))                          # its whole move in normal swings
        fast = abs(m) / normal[t] > s["fast"]
        if moved > s["move"]:
            label[t] = "up-fast" if fast else "up-steady"
        elif moved < -s["move"]:
            label[t] = "down-sharp" if fast else "down-steady"
        else:
            label[t] = "sideways-wild" if sd > normal[t] else "sideways-calm"
        start_at[t], length[t] = start, k
    starts = [c.index[i] if i >= 0 else pd.NaT for i in start_at]
    return pd.DataFrame({"stretch": label, "stretch_start": starts, "stretch_days": length}, index=c.index)
