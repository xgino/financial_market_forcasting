# financial_market_forcasting1/04_models/linear.py
"""Your own simple model: ridge regression trained in 03_training/linear.py. Predicts the move in units of
the asset's own swings, then turns it back into a % for this asset."""
SETTINGS = {"enabled": True}
CARD = {  # shown on the dashboard page Models
    "type": 'your own, trained (statistics)',
    "trained": 'DYNAMIC',
    "strengths": 'The standard baseline in finance (factor models are linear). Very hard to overfit, fully explainable.',
    "weaknesses": "Can't see combinations (for example: falling AND wild).",
    "limits": 'Same features as gbm.',
    "source": 'scikit-learn Ridge: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression',
}

import numpy as np

import settings
from core.trained import TrainedModel


class Linear(TrainedModel):
    about = "Your own: ridge regression on all assets + macro, trained walk-forward in 03_training."
    checkpoints = "linear"

    def predict_rows(self, artifact, X):
        z = artifact["model"].predict(X)
        scale = X["swings_60d"].to_numpy() * np.sqrt(settings.HORIZON_DAYS)
        return np.column_stack([np.expm1((z + artifact["miss_10"]) * scale), np.expm1(z * scale),
                                np.expm1((z + artifact["miss_90"]) * scale)])