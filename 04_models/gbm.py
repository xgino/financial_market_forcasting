# financial_market_forcasting1/04_models/gbm.py
"""Your own model: gradient boosting trained in 03_training/gbm.py (price + macro features, all assets).
Uses the newest checkpoint from before each day, so it only knows what was known then."""
SETTINGS = {"enabled": True}
CARD = {  # shown on the dashboard page Models
    "type": 'your own, trained (machine learning)',
    "trained": 'DYNAMIC',
    "strengths": 'Learns from 35 assets and macro data at once; tuned with GridSearchCV; can explain its drivers.',
    "weaknesses": 'Can learn patterns that were luck; the walk-forward checks and the holdout are there for that.',
    "limits": "Only what's in its features: price moves, swings, averages, macro. No news.",
    "source": 'scikit-learn HistGradientBoosting: https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting',
}

import numpy as np

from core.trained import TrainedModel


class GBM(TrainedModel):
    about = "Your own: gradient boosting on all assets + macro, trained walk-forward in 03_training."
    checkpoints = "gbm"

    def predict_rows(self, artifact, X):
        return np.column_stack([m.predict(X) for m in artifact["models"]])