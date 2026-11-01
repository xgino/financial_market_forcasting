# paper-fund-v2/04_models/garch.py
"""GARCH(1,1): the classic model for how wild the next weeks will be. Calm stretches and wild stretches
come in clusters; GARCH follows that. It has no opinion on direction (middle 0%), it's for the RANGE:
better sizing, not timing. Its settings are re-estimated once a year on the years before (past only).
Set settings.RANGE_MODEL = "garch" to let the base plan use its range."""
SETTINGS = {"enabled": True, "fit_years": 6, "window": 500}
CARD = {  # shown on the dashboard page Models
    "type": 'statistics (risk)',
    "trained": "Settings re-estimated each year on each asset's own past 6 years.",
    "strengths": 'The standard model for how wild markets get: wild and calm periods come in clusters.',
    "weaknesses": 'No direction at all.',
    "limits": 'Only for the range (sizing), never for buy or wait.',
    "source": 'GARCH: https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity',
}

import numpy as np
import pandas as pd

from core.contracts import NAN3, Model, bell_range


def fit_garch(r):
    """Maximum likelihood GARCH(1,1) on demeaned daily log returns -> (omega, alpha, beta)."""
    from scipy.optimize import minimize

    r = r - r.mean()
    var0 = r.var()

    def nll(x):
        w, a, b = x
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        s2 = np.empty_like(r)
        s2[0] = var0
        for t in range(1, len(r)):
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
        return 0.5 * np.sum(np.log(s2) + r ** 2 / s2)

    res = minimize(nll, [var0 * 0.05, 0.08, 0.9], method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-8})
    w, a, b = res.x
    if not res.success or a + b >= 0.999 or w <= 0:
        return var0 * 0.05, 0.05, 0.9   # safe fallback, still a valid GARCH
    return w, a, b


class Garch(Model):
    kind = "stats"
    about = "GARCH(1,1): how wild the next 20 days will be. Range only, no opinion on direction."

    def __init__(self, horizon, cfg):
        super().__init__(horizon, cfg)
        self._params = {}

    def _year_params(self, closes, year):
        key = year
        if key not in self._params:
            past = closes[closes.index < pd.Timestamp(f"{year}-01-01")]   # only the years before
            past = past.iloc[-int(self.s["fit_years"] * 252):]
            if len(past) < 500:
                return None
            self._params[key] = fit_garch(np.diff(np.log(past.to_numpy())))
        return self._params[key]

    def predict(self, view, asset):
        closes = view.closes(asset)
        if not hasattr(self, "_asset") or self._asset != asset:
            self._asset, self._params = asset, {}
        p = self._year_params(closes, view.date.year)
        if p is None:
            return NAN3
        w, a, b = p
        r = np.diff(np.log(closes.to_numpy()[-self.s["window"]:]))
        r = r - r.mean()
        s2 = r.var()
        for x in r:                       # run the variance forward up to today
            s2 = w + a * x * x + b * s2
        long = w / (1 - a - b)
        total = sum(long + (a + b) ** k * (s2 - long) for k in range(self.horizon))  # next 20 days
        return bell_range(0.0, np.sqrt(total))

    def explain(self, view, asset):
        closes = view.closes(asset)
        p = self._year_params(closes, view.date.year)
        if p is None:
            return ["Not enough history yet."]
        w, a, b = p
        r = np.diff(np.log(closes.to_numpy()[-self.s["window"]:]))
        s2 = r.var()
        for x in r - r.mean():
            s2 = w + a * x * x + b * s2
        return ["No opinion on direction (middle 0%). It only forecasts how wild the next 20 days will be.",
                f"Swings right now: {np.sqrt(s2):.2%} a day, long-run normal for this asset: {np.sqrt(w / (1 - a - b)):.2%} a day.",
                "Wild periods calm down slowly and calm ones get wilder slowly, so the range moves between the two."]