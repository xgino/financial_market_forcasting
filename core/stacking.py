# financial_market_forcasting1/core/stacking.py
"""
The judge: a model on top of the models (stacking). It learns "whom to believe when" from the battle itself.

  inputs   every other model's middle guess on day D (from data/forecasts/), their average and spread,
           and the 7 numbers that describe the day (core/similar.py: trend, swings, shape, ...)
  output   the log return over the next 20 trading days
  learner  a small gradient boosting model: it can learn "believe model X when the trend is down"

Honest, walk-forward:
  - refit at the start of every month, on every traded asset together, with only results that had ENDED
    before that month (like 03_training, but inside step 4: the inputs only exist once the others forecast)
  - a pretrained model's guesses up to its seen_until count as "no opinion" (0), in training AND when
    predicting: the judge must never learn to trust a model's memory
  - it runs after every other model (kind "stack"), so their forecasts for the day are already saved
Following it is guarded too (core/selection.py: stack_ok): only while its record so far beats the best
single model's record so far.
"""
import numpy as np
import pandas as pd

import settings
from core import store
from core.contracts import NAN3, NOT_IN_CACHE_KEY, Model, bell_range
from core.similar import describe


class Judge(Model):
    kind = "stack"
    about = "The judge: learned from the battle which model to believe in which situation (stacking)."
    uses_all_assets = True      # trains on every traded asset: its forecasts depend on all of them
    uses_macro = True

    def __init__(self, horizon, cfg):
        super().__init__(horizon, cfg)
        self.min_rows = self.s.get("min_rows", 2000)
        self.swing_days = self.s.get("swing_days", 60)
        self.source = None      # tests: {asset: DataFrame (date x model) of p50}. None = data/forecasts/
        self._inputs = None
        self._fits, self._p50, self._desc = {}, {}, {}

    # ----- which models it listens to -----

    def inputs(self):
        """{model: seen_until} for every enabled model in 04_models except judges and settings 'skip'."""
        if self._inputs is None and self.source is not None:   # tests: the given forecasts are the inputs
            self._inputs = {c: None for t in self.source.values() for c in t.columns}
        if self._inputs is None:
            from core.plugins import load_file, player_files

            out = {}
            for f in player_files(settings.ROOT / "04_models"):
                if f.stem in self.s.get("skip", []) or f.stem == self.run_name:
                    continue
                mod = load_file(f)
                cfg = dict(getattr(mod, "SETTINGS", {}))
                classes = [o for o in vars(mod).values() if isinstance(o, type) and issubclass(o, Model)
                           and o.__module__ == mod.__name__]
                if len(classes) != 1 or classes[0].kind == "stack" or not cfg.get("enabled", True):
                    continue
                if classes[0](self.horizon, cfg).supports(self.horizon):   # only models battled on this horizon
                    out[f.stem] = cfg.get("seen_until")
            self._inputs = out
        return self._inputs

    def cache_settings(self):
        """Its forecasts change when any input model's forecasts change (their own cache keys)."""
        keys = {}
        if self.source is None:
            for m in self.inputs():
                p = store.path("forecasts", "gold", f"{self._file(m)}.csv")
                keys[m] = store.read_meta(p).get("settings") if p.exists() else None
        return {**{k: v for k, v in self.s.items() if k not in NOT_IN_CACHE_KEY}, "inputs": keys}

    def _file(self, model):
        """The input model's forecast file for this judge's horizon (same naming as Model.cache_name)."""
        return model if self.horizon == settings.HORIZON_DAYS else f"{model}_h{self.horizon}"

    def _base(self, asset):
        if asset not in self._p50:
            if self.source is not None:
                t = self.source.get(asset, pd.DataFrame())
            else:
                cols = {}
                for m in self.inputs():
                    p = store.path("forecasts", asset, f"{self._file(m)}.csv")
                    if p.exists():
                        cols[m] = store.read_csv(p, index_col="date")["p50"]
                t = pd.DataFrame(cols)
            t = t.reindex(columns=list(self.inputs())).sort_index()
            t.index = pd.DatetimeIndex(t.index)   # also when nothing is saved for this asset yet
            for m, seen in self.inputs().items():   # memory years: no opinion
                if seen:
                    t.loc[t.index <= pd.Timestamp(str(seen)), m] = 0.0
            self._p50[asset] = t.fillna(0.0)
        return self._p50[asset]

    # ----- the table it learns from -----

    def _table(self, view, asset):
        """Features for every day of `asset` up to the view date (rolling, past only), plus the target where known."""
        closes = view.closes(asset)
        key = (asset, closes.index[-1])
        if key not in self._desc:
            extras = {n: view.extra(n) for n in ("real_rate", "dollar") if n in view.extra_names}
            d = describe(closes, extras)
            b = self._base(asset).reindex(closes.index).fillna(0.0)
            x = pd.concat([b, d], axis=1)
            x["avg"], x["spread"] = b.mean(axis=1), b.std(axis=1)
            lc = np.log(closes.to_numpy(float))
            target = np.full(len(lc), np.nan)
            target[:-self.horizon] = lc[self.horizon:] - lc[:-self.horizon]
            end = pd.Series(closes.index, index=closes.index).shift(-self.horizon)   # the day its 20 days are over
            self._desc = {k: v for k, v in self._desc.items() if k[0] != asset}       # keep one per asset
            self._desc[key] = (x, pd.Series(target, index=closes.index), end)
        return self._desc[key]

    def _fit(self, view, month):
        """The judge as of `month` (first day of a month): trained on results that ended before it."""
        if month not in self._fits:
            from sklearn.ensemble import HistGradientBoostingRegressor

            X, y = [], []
            for a in view.assets:
                if view.n_bars(a) < 300:
                    continue
                x, target, end = self._table(view, a)
                ok = (end < month).to_numpy() & np.isfinite(target.to_numpy())
                X.append(x[ok]), y.append(target[ok])
            X = pd.concat(X).replace([np.inf, -np.inf], np.nan) if X else pd.DataFrame()
            y = pd.concat(y) if y else pd.Series(dtype=float)
            X = X[[c for c in X.columns if X[c].nunique(dropna=True) >= 2]]   # early years: no macro yet, sklearn crashes on an empty column
            if len(X) < self.min_rows or X.empty:
                self._fits[month] = None
            else:
                m = HistGradientBoostingRegressor(max_depth=self.s.get("max_depth", 3), learning_rate=self.s.get("learning_rate", 0.05),
                                                  max_iter=self.s.get("max_iter", 200), min_samples_leaf=self.s.get("min_leaf", 200),
                                                  l2_regularization=1.0, random_state=0)
                self._fits[month] = (m.fit(X.to_numpy(float), y.to_numpy(float)), list(X.columns))
        return self._fits[month]

    # ----- forecasts -----

    def predict(self, view, asset):
        return self.predict_batch([view], asset)[0]

    def predict_batch(self, views, asset):
        out = np.tile(NAN3, (len(views), 1))
        for i, v in enumerate(views):
            if v.n_bars(asset) < self.min_history or not v.has_bar(asset):
                continue
            month = v.date.to_period("M").start_time
            fit = self._fit(v, month)
            if fit is None:
                continue
            m, cols = fit
            x, _, _ = self._table(v, asset)
            row = x.reindex(columns=cols).iloc[[-1]].to_numpy(float)
            mid = float(m.predict(row)[0])
            r = np.diff(np.log(v.closes(asset, self.swing_days + 1).to_numpy(float)))
            out[i] = bell_range(mid, r.std(ddof=1) * np.sqrt(self.horizon))
        return out

    def explain(self, view, asset):
        return [f"A judge that learned from the battle which model to believe when: it reads all {len(self.inputs())} "
                "models' guesses for today and today's situation numbers.",
                "Refit every month on results that had ended by then; followed only while its record beats the best single model."]
