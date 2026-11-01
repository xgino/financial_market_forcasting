# financial_market_forcasting1/tests/synthetic.py
"""Fake random prices for the tests, so they run anywhere without downloading anything.
On random prices no model can have real skill: a good test result here is 'nothing found'."""
import numpy as np
import pandas as pd

from core.timeline import Timeline


def prices(seed=0, start="2004-01-01", end="2014-12-31", vol=0.18):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    c = 100 * np.exp(np.cumsum(rng.normal(0.06 / 252, vol / np.sqrt(252), len(idx))))
    o = c * np.exp(rng.normal(0, 0.003, len(idx)))
    return pd.DataFrame({"open_eur": o, "close_eur": c}, index=pd.Index(idx, name="date"))


def macro(seed=10, start="2003-01-01", end="2014-12-31"):
    """Fake FRED-like series, indexed by the day each value became known."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    walk = lambda lvl, sd: pd.Series(lvl + np.cumsum(rng.normal(0, sd, len(idx))), index=idx)
    return {"real_rate": walk(1.0, 0.03), "rate10": walk(3.0, 0.04), "curve": walk(1.0, 0.03),
            "dollar": walk(100, 0.3), "vix": walk(18, 0.5).clip(lower=9)}


def timeline(n_assets=2, with_macro=True):
    return Timeline({f"asset{i}": prices(seed=i, vol=0.12 + 0.06 * i) for i in range(n_assets)},
                    macro() if with_macro else None)