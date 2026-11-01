# paper-fund-v2/03_training/gbm.py
# %% [markdown]
# # Training: gbm, your own model
# Gradient boosting (same family as LightGBM, via scikit-learn) on price features + macro features (FRED),
# learned from ALL assets together, including settings.TRAIN_ONLY_ASSETS. Three small models: one learns
# the bad case (10%), one the middle (50%), one the good case (90%).
#
# Walk-forward (core/training.py): one checkpoint per year, each trained only on results that had finished
# by its date. GridSearchCV tries every combination in SETTINGS["grid"] on time-ordered folds of the
# training data only (TimeSeriesSplit with a 20 day gap). The model in 04_models/gbm.py uses the newest
# checkpoint from before the day it forecasts.
#
# Run: python main.py 3  (or cell by cell in VS Code). Watch live: streamlit run dashboard.py, page Training.

SETTINGS = {
    "enabled": True,
    "first_cutoff": "2006-01-01",
    "every_years": 1,
    # every checkpoint picked the simplest corner of the old grid (depth 2, slow learning, few trees):
    # a sign the signal is weak, so the grid now reaches further toward simple instead of complex
    "grid": {"max_depth": [1, 2], "learning_rate": [0.01, 0.03], "max_iter": [100, 300], "l2_regularization": [1.0, 10.0]},
    "min_samples_leaf": 200,
    "cv_folds": 3,
    "min_rows": 1000,
}

# %%
import sys
from pathlib import Path

if "settings" not in sys.modules:  # running on its own (VS Code cells): find the repo root
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(next(p for p in [here, *here.parents] if (p / "settings.py").exists())))

import settings
from core.contracts import Trainer
from core.training import training_rows, walk_forward  # noqa: F401  (training_rows: handy in the cells)


# %%
def fit(rows, cfg, horizon, log):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import make_scorer, mean_pinball_loss
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline

    from core.training import KeepInformative

    X, y = rows.drop(columns=["target", "asset"]), rows["target"]

    def model(q, **params):
        return Pipeline([("keep", KeepInformative()),
                         ("gbm", HistGradientBoostingRegressor(loss="quantile", quantile=q, early_stopping=False,
                                                               random_state=0, min_samples_leaf=cfg["min_samples_leaf"], **params))])

    folds = TimeSeriesSplit(n_splits=cfg["cv_folds"], gap=horizon * rows["asset"].nunique())
    grid = {f"gbm__{k}": v for k, v in cfg["grid"].items()}
    search = GridSearchCV(model(0.5), grid, cv=folds, n_jobs=-1,
                          scoring=make_scorer(mean_pinball_loss, alpha=0.5, greater_is_better=False)).fit(X, y)
    best = {k.replace("gbm__", ""): v for k, v in search.best_params_.items()}
    for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
        clean = {k.replace("gbm__", ""): v for k, v in params.items()}
        log(**clean, cv_loss=-score, rows=len(rows), best=clean == best)
    models = [model(q, **best).fit(X, y) for q in (0.1, 0.5, 0.9)]   # the winner, trained on ALL rows
    medians = X.median(numeric_only=True).to_dict()                  # "normal" values, for explanations
    return ({"models": models, "columns": list(X.columns), "medians": medians, "assets": rows["asset"].nunique()},
            {"cv_loss": round(float(-search.best_score_), 6), **best})


class GBMTrainer(Trainer):
    about = "gradient boosting on all assets + macro, one checkpoint per year, tuned with GridSearchCV"

    def train(self, tl, progress=True, fresh=False):
        return walk_forward("gbm", self.s, tl, fit, progress, fresh)


# %% [markdown]
# ## Run it here, cell by cell, or with: python main.py 3

# %%
if __name__ == "__main__":
    from core.timeline import Timeline

    print(GBMTrainer(SETTINGS).train(Timeline.load([a for a, s in settings.ALL_ASSETS.items() if s.get("train", True)])))