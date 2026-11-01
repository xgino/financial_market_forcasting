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
- [x] Battle runs one model at a time on all assets, then frees its memory (no 8 big networks in RAM at once)
- [x] tqdm progress bar per model and asset; progress saved at least every minute (Kronos in small chunks)
- [x] Dividend rescales (SPY, QQQ, TLT) no longer redo whole histories: months are compared by daily moves
- [x] `python main.py 4 --fresh` really starts the battle from scratch (it was ignored before)

## Now
- [ ] Finish the battle run with all 19 models (Kronos ~1-2 h, Time-MoE ~2.5 h), then steps 5-6
- [ ] Look at the leaderboard: do TiRex, Kronos, Moirai 2 or Time-MoE show anything on gold after their release?

## Next: better decisions from the same historical data

### 1. Score "is it a bad time to buy?" directly
- [x] New score per model: when it said "fall" (middle guess below -0.5%), how often did it really fall, and how much lower than on any day? (`warn_*` in `core/metrics.py`)
- [x] Same for "rise" (middle guess beats buy + sell costs): how often did it rise? (`rise_*`)
- [x] Both in the skill book, per situation and per asset, month by month; trust = edge in units of luck (z ≥ 2); leaderboard adds a shift luck test
- [x] WAIT on the Now page weighs fall votes by the warning record and needs a proven warner (`core/verdict.py`); BUY still uses IC trust
- [x] Dashboard: warning record table on Leaderboard, a line per model on Models, the warner's record on Now when it says WAIT
- [ ] After the full battle run: `python main.py 4-6` once more (forecasts are cached, only the skill book and leaderboard are rebuilt with the new columns)
- [ ] Look at it: does any model warn better than luck on gold? Does WAIT avoid the bad months in the history replay ("past success of this call")?

### 2. Select on similar days instead of 6 fixed boxes
- [x] Describe each day with numbers: trend strength, swings now vs normal, last 20/60 day shape, distance from the 1 year high, real rate, dollar (`core/similar.py`)
- [x] For today: the 150 most similar past days of that asset whose results were already known (scaled with past days only)
- [x] Score every model only on those days: trust over independent 20 day periods (fewer/clustered days = higher bar) + a luck test that shuffles whole periods
- [x] New selector `best_in_similar_days` next to `best_in_situation`, and strategy `follow_similar` next to `follow_best`, so steps 5 and 6 compare them
- [x] Tests: similar days only have known results, garbage after the date changes nothing; the selection no-future test covers the new selector
- [ ] After the full battle run: compare `best_in_similar_days` vs `best_in_situation` (step 5 scores) and `follow_similar` vs `follow_best` (step 6)
- [ ] Try `days` 100 / 150 / 250 in its SETTINGS: does a smaller, closer set help or just add luck?

### 3. More evidence: pool related assets
Found: trust was already pooled, over ALL assets equally (gold's trust counted Bitcoin's results as much as gold's).
- [x] Groups in settings (`settings.GROUPS`): metals (gold, silver), stocks (sp500, nasdaq, aex), bonds, crypto
- [x] Own asset counts fully, its group at half weight, other assets not (`settings.POOLS`, `settings.EVIDENCE = "group"`); periods add up weighted, so borrowed evidence raises trust less (`core/pooling.py`)
- [x] Skill book has rows per asset and situation; selectors, the Now verdict and the email use the pooled record
- [x] Check in step 4 (`data/battle/pooling.csv`, Leaderboard page): own / group / all, does trust on 1 January predict that year's IC, and how steady is it
- [ ] After the full battle run: read the check and set `settings.EVIDENCE` to the pool that predicts best
- [ ] Platinum and gold miners as traded (battled) assets, so metals has 4 members (costs ~2 more assets of battle time)

### 4. A judge model on top of the models (stacking)
- [x] Inputs: every other model's middle guess, their average and spread, and the 7 similar-days numbers; output: the 20 day result (`core/stacking.py`, `04_models/judge.py`)
- [x] Walk-forward: refit every month inside step 4 (its inputs only exist there), on all traded assets, only results that ended before that month; pretrained guesses up to `seen_until` count as "no opinion"
- [x] Runs after every other model; its cache follows the other models' cache keys
- [x] Followed (selectors) and voting (Now verdict) only while its record so far beats the best single model's record so far (`stack_ok`); the leaderboard says whether it beats the best on the development years
- [x] Tests: no future (prices, macro and other models' forecasts after the date), memory years ignored, the beat-the-best gate
- [ ] After the full battle run: does the judge beat the best single model? If it doesn't, try fewer inputs (only models with some trust) or a plainer learner (ridge)

### 5. More than one horizon
- [x] `settings.HORIZONS = [5, 20, 60]`: step 4 battles every horizon; decisions and strategies stay on `HORIZON_DAYS = 20`
- [x] Own journal, skill book (with the warning record) and leaderboard per horizon (`*_h5.csv`, `*_h60.csv`); forecasts cached as `<model>_h5.csv`; luck test shifts further for longer horizons
- [x] Which models join: SETTINGS `horizons` (gbm and linear only 20: trained on 20 day results; Kronos and Time-MoE only 20: slow); the judge reads the same horizon's forecasts
- [x] Dashboard: Now page "This week (5 days)" and "This quarter (60 days)" calls; Leaderboard page has a horizon picker
- [x] Tests: who joins which horizon, no future on 5 and 60 days, a 5 day range is narrower than a 60 day one
- [ ] After the full battle run: which models have skill on 5 or 60 days that they don't have on 20?
- [ ] Train gbm and linear per horizon too (03_training on 5 and 60 day results)
- [ ] If 5 days shows real skill: a weekly strategy (more trades, more fees: does it still win after costs?)

### 6. Pattern chunks instead of calendar chunks
- [x] Cut each price line into stretches where it behaved the same: online change-point detection, a new stretch when the last 20 days' average move (z > 3) or swings (2x) differ from the stretch so far; a day's label is never revised (`core/stretches.py`, `settings.STRETCHES`)
- [x] Label by shape and size against the asset's normal swings: up-steady, up-fast, down-steady, down-sharp, sideways-calm, sideways-wild; in the journal (`stretch`) and the situations files
- [x] Skill book scores every model per stretch kind (`stretch:<kind>`, pooled per asset like the situations); selector `best_in_stretch`, strategy `follow_stretch`; Now page shows the stretch and since when
- [x] Step 5 prints which grouping finds the model to follow best: none / situations / stretches / similar days (`data/selection/groupings.csv`)
- [x] Tests: stretches never change when the future changes, stretch scores use the right forecasts and don't double count, selection no-future test covers `best_in_stretch`
- [ ] After the full battle run: read the groupings table. A grouping only helps if it beats "no grouping" on IC and on "move when up vs all"
- [ ] Tune `settings.STRETCHES` only on the development years (gold: 36 stretches since 2003, median ~100 days, sideways-wild 39%)

### 7. Fine-tune the pretrained models (last: slow, small expected gain)
- [x] Fine-tune Chronos-Bolt small on your training assets, walk-forward: a checkpoint per year from its release (2025-01-01, 2026-01-01), only windows whose 20 days ended before the cutoff, the last ~6 months held back for a check (`03_training/chronos_finetune.py`, ~2 min per checkpoint)
- [x] Held-back check (Training page): 2025 fine-tuned 5.857 vs zero-shot 5.873 (0.3% better), 2026 4.478 vs 4.411 (1.5% worse)
- [x] `chronos_small_ft` in the battle against `chronos_small` (same model, zero shot), from 2025 on; a year whose fine-tuning didn't help (known at its cutoff) uses the base model
- [x] Weights (~190 MB each) kept out of git; `index.csv` stays in
- [ ] After the battle re-run: does `chronos_small_ft` beat `chronos_small` on 2025-2026? (Few results yet: judge again in a year)
- [ ] If it helps: fine-tune Chronos-2 with its built-in `fit()` (LoRA), same walk-forward

## Housekeeping
- [ ] `python main.py 4 --only x` overwrites journal, leaderboard and skill book with only model x: merge into the full ones instead
- [ ] Dashboard Run page: run steps in the background with a log file (clicking around now stops the live output and can hang the step)
- [x] `data/` and `__pycache__/` stay in git on purpose
- [ ] `tools/` (monthly reminder, backup) and `secrets.example.env` are mentioned in 07_judge and 08_notify, but missing
- [ ] Moirai 2 install: `uni2ts` needs `--no-deps`; make that one command (setup script)

## Later
- [ ] The final test: `python main.py 4-6 --holdout`, once, when models and strategies are final
- [ ] Live paper trading month by month; real money only after `settings.REAL_MONEY_RULE_MONTHS` good live months
