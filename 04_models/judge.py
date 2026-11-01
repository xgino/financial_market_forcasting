# financial_market_forcasting1/04_models/judge.py
"""The judge (stacking): learns from the battle which model to believe when. Reads every other model's
guess for the day + the day's situation numbers, refit monthly on finished results only (core/stacking.py).
Followed only while its record beats the best single model's record (core/selection.py stack_ok)."""
SETTINGS = {
    "enabled": True,
    "skip": ["naive"],        # inputs it ignores (naive always says 0%)
    "min_rows": 2000,         # finished results needed before its first forecast
    "max_depth": 3, "learning_rate": 0.05, "max_iter": 200, "min_leaf": 200,
    "swing_days": 60,         # its bad..good range: the last 60 days' swings
}
CARD = {  # shown on the dashboard page Models
    "type": 'your own (stacking: a model on top of the models)',
    "trained": 'Refit every month in step 4, on every traded asset, with only results that had ended. Pretrained '
               "models' guesses from before their release count as 'no opinion'.",
    "strengths": 'Can learn who to believe in which situation, from all models at once.',
    "weaknesses": 'Easy to overfit; only as good as the models it reads. A range from recent swings, not its own.',
    "limits": 'Followed only while its record beats the best single model. Runs after all other models.',
    "source": 'Stacked generalization (Wolpert 1992): https://en.wikipedia.org/wiki/Ensemble_learning#Stacking',
}

from core.stacking import Judge


class StackJudge(Judge):
    pass
