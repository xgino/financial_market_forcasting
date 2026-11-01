# financial_market_forcasting1

A lab that tests whether any model can really forecast markets, with a virtual EUR 1000 and strict rules
against fooling yourself. Gold first; silver, stocks, bonds, Bitcoin and Ethereum are in too.
Every folder with a number is a step. Steps run in order. What's done and what's next: [ROADMAP.md](ROADMAP.md).

## The flow

```
01_data        prices (Yahoo, in EUR) + macro (FRED)          ─┐
02_features    trend, swings, situation label per day           │  the data
03_training    your own models (gbm, linear), walk-forward      ─┘
04_models      the battle: every model forecasts every day,       one forecast = bad / middle / good case
               20 trading days ahead, scored once those days are over (also 5 and 60 days: settings.HORIZONS)
05_selection   who to trust this month, per situation (or nobody = base plan)
06_strategies  what to do with EUR 1000: buy / sell / hold, with fees
07_judge       plain-language report: what's good, what needs a look
08_notify      the monthly email, only when it matters
```

| folder | what happens | writes |
|---|---|---|
| `01_data/` | download and clean prices and macro data. Today's bar waits until it's final (tomorrow) | `data/prices/`, `data/macro/` |
| `02_features/` | trend, swings and one of 6 situations per day (up/flat/down × calm/wild) | `data/features/`, `data/situations/` |
| `03_training/` | train your own models walk-forward: each checkpoint only learns from results finished by its date | `data/checkpoints/`, `data/training/` |
| `04_models/` | the battle: every model forecasts, gets scored and ranked, month by month (skill book). Same for 5 and 60 days (information; decisions use 20) | `data/forecasts/`, `data/battle/` (`*_h5`, `*_h60`) |
| `05_selection/` | per month and situation: which model(s) to follow, or none. A model's record for an asset = that asset, plus its group (`settings.GROUPS`) at half weight | `data/selection/` |
| `06_strategies/` | EUR 1000 month by month, with fees, against buy & hold | `data/simulation/` |
| `07_judge/` | checks data, models, strategies and live results | `data/judge/` |
| `08_notify/` | the monthly email (needs `secrets.env`) | `data/notify/` |

Not numbered:
- `core/` fixed building blocks every step uses (timeline that hides the future, wallet, scoring, the
  pretrained model wrappers, copied model code in `core/kronos/` and `core/timemoe/`)
- `tests/` checks that everything stays honest
- `settings.py` all global settings in one place (assets, horizon, holdout, trust, fees)
- `main.py` runs the steps, `dashboard.py` shows everything

## The models (04_models)

| kind | models |
|---|---|
| rules | `naive`, `drift`, `mean_reversion`, `bounce`, `history_range` |
| statistics | `theta`, `ets`, `garch`, `analog` (pattern matching) |
| your own (trained in 03) | `gbm`, `linear` |
| judge (stacking, refit monthly in 04) | `judge`: reads every other model's guess + the day's situation numbers; followed only while it beats the best single model |
| pretrained, fine-tuned by you (03) | `chronos_small_ft`: Chronos-Bolt small fine-tuned walk-forward on your assets, battles `chronos_small` |
| pretrained, zero shot | `chronos_small`, `chronos_base`, `chronos_2` (Amazon), `timesfm` (Google), `tirex` (NX-AI), `kronos_small` (market candles), `moirai_2` (Salesforce), `timemoe` (Time-MoE 200M) |

Pretrained models download themselves from Hugging Face the first time (cached in `~/.cache/huggingface`).
They only earn trust on days after their release (`seen_until`): before that they may have seen the prices.

## Setup (once)

Only the `venv/` in this folder runs this project. Any other Python (Anaconda, the system one) is stopped
with a message (`core/env.py`), because other package versions break in ways that are hard to trace.

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps uni2ts==2.0.0     # for moirai_2: its own pins would downgrade torch and numpy
```
VS Code picks `venv/` by itself (`.vscode/settings.json`).

## Commands
Every time you open a terminal: `source venv/bin/activate` (or put `venv/bin/` in front of each command).
```
python main.py              # every step
python main.py 4            # one step
python main.py 2-4          # steps 2 to 4
python main.py 4 --only tirex,moirai_2   # only these players
python main.py 4 --check    # prove step 4's players can't see the future
python main.py list         # steps and players
python main.py test         # all tests
streamlit run dashboard.py  # the dashboard (or: venv/bin/streamlit run dashboard.py)
```

Every step only does what's missing, so stopping and starting again continues where it was (Ctrl+C is safe:
forecasts are saved at least every minute). `--fresh` is the only way to redo a step from scratch.
- The battle runs one model at a time over all assets, then frees its memory before the next model loads.
- Forecasts are cached per model and asset. When old prices change, only days from the first changed month
  are redone; a dividend rescale (ETFs) redoes nothing, because months are compared by their daily moves.

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
