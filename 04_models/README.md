# 04_models: the battle

Every file here is a model. Each one forecasts every trading day as if live, gets scored once the 20 days
are over, and lands on the leaderboard (`data/battle/leaderboard.csv`).

A model returns 3 numbers for 20 trading days ahead: bad case (10%), middle guess (50%), good case (90%),
as returns. Price it expects = today's price x (1 + middle guess).

| kind | files |
|---|---|
| your own, trained in 03 | `gbm.py` |
| rules | `naive.py` (the floor), `drift.py`, `mean_reversion.py`, `bounce.py` |
| statistics | `history_range.py`, `analog.py`, `ets.py`, `theta.py` |
| pretrained | `chronos_small.py`, `chronos_base.py`, `chronos_2.py`, `timesfm.py`, `tirex.py`, `kronos_small.py` (all with `seen_until`; Kronos is slow, so it only forecasts from its `first_day`) |

Add a model:
```python
# paper-fund-v2/04_models/my_model.py
"""What it does, in plain words."""
SETTINGS = {"enabled": True, "days": 60}

from core.contracts import Model

class MyModel(Model):
    kind = "rule"
    about = "one line"
    def predict(self, view, asset):
        c = view.closes(asset, self.s["days"])   # only the past, the view enforces it
        return [-0.05, 0.0, 0.05]                # bad, middle, good
```
Then `python main.py 4 --check --only my_model` and `python main.py 4`.