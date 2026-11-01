# Roadmap

What's done, what's next. Tick a box (`[x]`) when it's done. The order inside "Next" is the order to do it in.

## Done

### The base (v2)
- [x] Steps 01 to 08: data → features → training → battle → selection → strategies → judge → email
- [x] Timeline that hides the future, honesty tests, `--check` per step
- [x] 6 situations per day (up/flat/down × calm/wild), skill book month by month
- [x] Selectors: best overall, best in situation, top N blend, bandit, base plan
- [x] Strategies: buy & hold, fixed mix, CPPI floor, risk balanced, follow best / overall / bandit
- [x] Dashboard (Streamlit) with Run, Battle, Leaderboard, Now, Results, Models, Email, Judge pages

### Assets and models (2026-11)
- [x] Bitcoin and Ethereum as traded assets, kept out of your own models' training (`"train": False`)
- [x] Month starts on weekdays only (crypto's weekend days don't move the monthly decisions)
- [x] Pretrained: TiRex (NX-AI), Kronos small (market candles), Moirai 2 (Salesforce), Time-MoE 200M
- [x] Kronos only forecasts from its `first_day` (too slow for the full history; earlier years can't earn trust anyway)
- [x] Same day = same forecast, whatever it's batched with (Kronos seeds per day, TiRex fills batches)
- [x] Time-MoE: code copied into `core/timemoe` instead of `trust_remote_code`; fixed its zeroed position buffers

### Running it
- [x] Today's unfinished bar waits until tomorrow (it made the cache redo old days on every run)
- [x] Changed old prices only redo forecasts from the first changed month, not the whole history
- [x] Only the project's `venv/` can run it (`core/env.py`), also from the dashboard
- [x] README with the flow, setup and models; `.gitignore` for venvs, secrets, editor files

## Now
- [ ] Finish the battle run with all 19 models (Kronos ~1-2 h, Time-MoE ~2.5 h), then steps 5-6
- [ ] Look at the leaderboard: do TiRex, Kronos, Moirai 2 or Time-MoE show anything on gold after their release?

## Next: better decisions from the same historical data

### 1. Score "is it a bad time to buy?" directly
- [ ] New score per model: when it said "high chance of falling", how often did it really fall (and how much)?
- [ ] Same for "high chance of rising": how often did it rise?
- [ ] Put both in the skill book, per situation, with the same luck test as the other scores
- [ ] WAIT on the Now page uses the models with the best warning record
- [ ] Dashboard: warning record per model

### 2. Select on similar days instead of 6 fixed boxes
- [ ] Describe each day with numbers: trend strength, swings now vs normal, last 20/60 day shape, real rate, dollar
- [ ] For today: find the 100-200 most similar past days whose results were already known
- [ ] Score every model only on those days, with the luck test (fewer days = higher bar)
- [ ] New selector `best_in_similar_days`, next to `best_in_situation`, so the battle shows which works better
- [ ] Test: similar days never include results that weren't known yet

### 3. More evidence: pool related assets
- [ ] Groups in settings: metals (gold, silver, platinum, gold miners), stocks, bonds, crypto
- [ ] A model's record on the other assets in its group counts as extra evidence (weighted lower than its own)
- [ ] Check: does pooling make the trust scores steadier from year to year, or just noisier?

### 4. A judge model on top of the models (stacking)
- [ ] Inputs: every model's forecast + today's situation numbers; output: what really happened
- [ ] Trained walk-forward like 03_training (only finished results), as a new player in 04_models
- [ ] It has to beat the best single model on the development years before it can be followed

### 5. More than one horizon
- [ ] Forecast 5, 20 and 60 trading days ahead (now only `settings.HORIZON_DAYS = 20`)
- [ ] Skill book per horizon: 5 days gives ~4x more results to score, 60 days fewer but bigger moves
- [ ] "When to buy this week" from 5 days, "the direction this quarter" from 60 days

### 6. Pattern chunks instead of calendar chunks
- [ ] Cut each price line into stretches where it behaved the same (change-point detection), only with data known then
- [ ] Label stretches by shape and size (steady up, sharp drop, sideways, ...) and score models per kind
- [ ] Compare with the 6 situations and with similar days (2): which explains model skill best?

### 7. Fine-tune the pretrained models (last: slow, small expected gain)
- [ ] Fine-tune Chronos-Bolt on your assets, walk-forward (checkpoint per year, only data up to its date)
- [ ] Battle it against the zero-shot version

## Housekeeping
- [ ] Dashboard Run page: run steps in the background with a log file (clicking around now stops the live output and can hang the step)
- [x] `data/` and `__pycache__/` stay in git on purpose
- [ ] `tools/` (monthly reminder, backup) and `secrets.example.env` are mentioned in 07_judge and 08_notify, but missing
- [ ] Moirai 2 install: `uni2ts` needs `--no-deps`; make that one command (setup script)

## Later
- [ ] The final test: `python main.py 4-6 --holdout`, once, when models and strategies are final
- [ ] Live paper trading month by month; real money only after `settings.REAL_MONEY_RULE_MONTHS` good live months
