# Paper fund (v2)

A lab that tests whether any model can really forecast markets, with a virtual EUR 1000 and strict rules
against fooling yourself. Every folder with a number is a step. Steps run in order.

| folder | what happens | writes |
|---|---|---|
| `01_data/` | download and clean prices | `data/prices/*.csv` |
| `02_features/` | trend, swings, situations per day | `data/features/`, `data/situations/` |
| `03_training/` | train your own models, walk-forward, tuned | `data/checkpoints/`, `data/training/` |
| `04_models/` | the battle: every model forecasts, gets scored and ranked | `data/battle/*.csv` |
| `05_selection/` | who to trust: per month and situation, which model(s) to follow, or none | `data/selection/*.csv` |
| `06_strategies/` | what to do with EUR 1000: buy / sell / hold, how much, month by month with fees | `data/simulation/*.csv` |
| `07_...` | next: the judge, live paper trading | |

Not numbered:
- `core/` fixed building blocks every step uses (timeline that hides the future, wallet, scoring)
- `tests/` checks that everything stays honest
- `settings.py` all global settings in one place
- `main.py` runs the steps, `dashboard.py` shows everything

## Commands
```
pip install -r requirements.txt
python main.py              # every step
python main.py 4            # one step
python main.py 4 --check    # prove step 4's players can't see the future
python main.py list         # steps and players
python main.py test         # all tests
streamlit run dashboard.py  # the dashboard
```

## How a folder works
- `_step.py` runs the folder. Files starting with `_` are the folder's machinery.
- Every other `.py` file is a player (a data source, a model, a trainer, ...). Drop a new file in, it joins the next run.
- Every player has `SETTINGS` at the top. `"enabled": False` takes it out without deleting it.

## The honesty rules
1. A forecast for day D only sees data up to D (`core/timeline.py`). Asking for later data crashes.
2. A result only counts once its 20 days are over.
3. Pretrained models never earn trust on prices they may have trained on (`seen_until`).
4. Training only uses results that had finished by the checkpoint's date.
5. The holdout (from `settings.HOLDOUT_FROM`) stays hidden until the very end.
6. `python main.py test` and `--check` prove 1 to 4 on every run you like.