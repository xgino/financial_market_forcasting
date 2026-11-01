# paper-fund-v2/03_training/linear.py
# %% [markdown]
# # Training: linear, your own simple model
# Ridge regression: one straight-line formula over the same features as gbm (price + macro, all assets).
# Simple models overfit less, and with a weak signal they're often as good as complicated ones.
#
# It predicts the 20 day move in units of the asset's own swings (so calm and wild assets learn together
# fairly), then turns it back into a % for that asset. The bad / good case come from how far its guesses
# were off in training (10% and 90% of the misses).
# Tuning: GridSearchCV over the ridge strength (alpha), time-ordered folds of the training data only.

SETTINGS = {
    "enabled": True,
    "first_cutoff": "2006-01-01",
    "every_years": 1,
    # bigger = simpler formula. Every checkpoint picked the largest of the old grid (10000), so it goes further
    "alphas": [100, 1000, 10000, 100000, 1000000],
    "cv_folds": 3,
    "min_rows": 1000,
}

# %%
import sys
from pathlib import Path

if "settings" not in sys.modules:
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(next(p for p in [here, *here.parents] if (p / "settings.py").exists())))

import numpy as np

import settings
from core.contracts import Trainer
from core.training import walk_forward


def to_z(target, swings, horizon):
    return np.log1p(target) / (swings * np.sqrt(horizon))


# %%
def fit(rows, cfg, horizon, log):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = rows[rows["swings_60d"] > 0]
    X = rows.drop(columns=["target", "asset"])
    z = to_z(rows["target"], rows["swings_60d"], horizon)
    keep = np.isfinite(z)
    X, z = X[keep], z[keep]
    pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge())
    folds = TimeSeriesSplit(n_splits=cfg["cv_folds"], gap=horizon * rows["asset"].nunique())
    search = GridSearchCV(pipe, {"ridge__alpha": cfg["alphas"]}, cv=folds, scoring="neg_mean_absolute_error").fit(X, z)
    for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
        log(alpha=params["ridge__alpha"], cv_loss=-score, rows=len(X), best=params == search.best_params_)
    model = search.best_estimator_
    miss = z - model.predict(X)
    return ({"model": model, "columns": list(X.columns), "miss_10": float(np.quantile(miss, 0.1)),
             "miss_90": float(np.quantile(miss, 0.9)), "medians": X.median(numeric_only=True).to_dict(),
             "assets": rows["asset"].nunique()},
            {"cv_loss": round(float(-search.best_score_), 6), "alpha": search.best_params_["ridge__alpha"]})


class LinearTrainer(Trainer):
    about = "ridge regression on all assets + macro, one checkpoint per year"

    def train(self, tl, progress=True, fresh=False):
        return walk_forward("linear", self.s, tl, fit, progress, fresh)


# %%
if __name__ == "__main__":
    from core.timeline import Timeline

    print(LinearTrainer(SETTINGS).train(Timeline.load([a for a, s in settings.ALL_ASSETS.items() if s.get("train", True)])))