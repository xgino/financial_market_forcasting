# financial_market_forcasting1/04_models/theta.py
"""Classic statistics from the statsforecast library (the Theta method), fitted fresh every day on the last
year of log prices. Strong in forecasting competitions; on market prices often close to naive.
Install: pip install statsforecast"""
SETTINGS = {"enabled": True, "window": 252}
CARD = {  # shown on the dashboard page Models
    "type": 'statistics (classic)',
    "trained": "Fitted fresh every day on each asset's own last year.",
    "strengths": 'Won the M3 forecasting competition; simple and robust.',
    "weaknesses": "Same as ets: markets aren't sales data.",
    "limits": 'Price line only.',
    "source": 'statsforecast AutoTheta: https://nixtlaverse.nixtla.io/statsforecast/',
}

import numpy as np

from core.contracts import NAN3, Model


class Theta(Model):
    kind = "stats"
    about = "Classic statistics: the Theta method, refitted daily."

    def _model(self):
        if not hasattr(self, "_m"):
            from statsforecast.models import AutoTheta
            self._m = AutoTheta(season_length=1)
        return self._m

    def predict(self, view, asset):
        c = view.closes(asset, self.s["window"]).to_numpy()
        if len(c) < 60:
            return NAN3
        y = np.log(c)
        try:
            r = self._model().forecast(y=y, h=self.horizon, level=[80])
        except Exception:
            return NAN3
        x = np.array([r["lo-80"][-1], r["mean"][-1], r["hi-80"][-1]], dtype=float)  # 80% band = 10%..90%
        return np.sort(np.exp(x - y[-1]) - 1)

    def explain(self, view, asset):
        p = self.predict(view, asset)
        return ["The Theta method: a long-term trend line mixed with the recent level of the last year's prices.",
                f"Together they give {p[1]:+.2%} over 20 trading days."]