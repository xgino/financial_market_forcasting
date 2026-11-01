# financial_market_forcasting1/core/trained.py
"""
Base for the 04_models files of your own trained models. It picks, for every day, the newest checkpoint
whose cutoff is on or before that day (so it only knows what was known then), builds the same features
as training (price + macro, from the view) and asks the file's predict_rows for the 3 numbers.
Retraining (new checkpoints) automatically makes all its forecasts again.
"""
import pickle

import numpy as np
import pandas as pd

import settings
from core import store
from core.contracts import NAN3, Model
from core.features import NEEDS_DAYS, describe, full_table


class TrainedModel(Model):
    kind = "ml"
    uses_macro = True
    horizons = (settings.HORIZON_DAYS,)   # trained in 03_training on the main horizon's results only
    checkpoints = None          # folder name in data/checkpoints/

    def __init__(self, horizon, cfg):
        super().__init__(horizon, cfg)
        folder = settings.DATA / "checkpoints" / (self.checkpoints or "")
        idx = folder / "index.csv"
        self.folder = folder
        self.index = pd.read_csv(idx, parse_dates=["cutoff"]).sort_values("cutoff") if idx.exists() else pd.DataFrame()
        self._loaded = {}

    def cache_settings(self):
        return {**super().cache_settings(), "checkpoints": store.fingerprint(self.index) if len(self.index) else None}

    def _checkpoint(self, date):
        if self.index.empty:
            return None
        ok = self.index[self.index["cutoff"] <= date]
        if ok.empty:
            return None
        c = str(ok["cutoff"].iloc[-1].date())
        if c not in self._loaded:
            with open(self.folder / f"{c}.pkl", "rb") as f:
                self._loaded[c] = pickle.load(f)
        return self._loaded[c]

    def predict_rows(self, artifact, X):
        """X: one row of features per forecast -> array (n, 3) of p10, p50, p90 returns."""
        raise NotImplementedError

    def predict(self, view, asset):
        return self.predict_batch([view], asset)[0]

    def predict_batch(self, views, asset):
        out = np.tile(NAN3, (len(views), 1))
        groups = {}
        for i, v in enumerate(views):
            ck = self._checkpoint(v.date)
            if ck is None:
                continue
            x = full_table(v, asset, NEEDS_DAYS + 10).iloc[[-1]].reindex(columns=ck["columns"])
            if x["ret_60d"].isna().any():
                continue
            pos, rows = groups.setdefault(ck["cutoff"], ([], []))
            pos.append(i)
            rows.append(x)
        for cutoff, (pos, rows) in groups.items():
            out[pos] = np.sort(self.predict_rows(self._loaded[cutoff], pd.concat(rows)), axis=1)
        return out

    def explain(self, view, asset, top=5):
        """Which features push this forecast up or down: each one is set back to its normal (training median)
        value in turn, and the change in the middle guess is its effect."""
        ck = self._checkpoint(view.date)
        if ck is None:
            return ["No checkpoint was trained before this date yet."]
        x = full_table(view, asset, NEEDS_DAYS + 10).iloc[[-1]].reindex(columns=ck["columns"])
        base = float(self.predict_rows(ck, x)[0][1])
        lines = [f"{self.about}", f"Checkpoint trained up to {ck['cutoff']} on {ck.get('assets', '?')} assets. "
                                  f"Middle guess now: {base:+.2%}."]
        med = ck.get("medians")
        if not med:
            return lines + ["Retrain (python main.py 3) to see which features drive it."]
        effects = []
        for c in ck["columns"]:
            v = x[c].iloc[0]
            if pd.isna(v) or pd.isna(med.get(c, np.nan)):
                continue
            x2 = x.copy()
            x2[c] = med[c]
            effects.append((c, v, med[c], base - float(self.predict_rows(ck, x2)[0][1])))
        effects.sort(key=lambda e: -abs(e[3]))
        lines.append("What pushes it most (vs a normal value of that feature):")
        for c, v, m, eff in effects[:top]:
            words, now = describe(c, v)
            _, normal = describe(c, m)
            lines.append(f"• {words}: now {now}, normally {normal} → {eff:+.2%} on the middle guess")
        return lines