# financial_market_forcasting1/04_models/ets.py
"""Classic statistics from the statsforecast library (exponential smoothing (ETS)), fitted fresh every day on the last
year of log prices. Strong in forecasting competitions; on market prices often close to naive.
Install: pip install statsforecast"""
SETTINGS = {"enabled": True, "window": 252}
CARD = {  # shown on the dashboard page Models
    "type": 'statistics (classic)',
    "trained": "Fitted fresh every day on each asset's own last year.",
    "strengths": 'Classic, robust, well understood. Strong in forecasting competitions on sales and demand data.',
    "weaknesses": 'Market prices have no stable trend or season, so it usually ends up close to naive.',
    "limits": 'Price line only.',
    "source": 'statsforecast AutoETS: https://nixtlaverse.nixtla.io/statsforecast/',
}

import numpy as np

from core.contracts import NAN3, Model


class ETS(Model):
    kind = "stats"
    about = "Classic statistics: exponential smoothing (ETS), refitted daily."

    def _model(self):
        if not hasattr(self, "_m"):
            from statsforecast.models import AutoETS
            self._m = AutoETS(season_length=1, model="ZZN")
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
        return ["Exponential smoothing of the last year's prices: recent days count more than old ones.",
                f"The smoothed trend gives {p[1]:+.2%} over 20 trading days."]