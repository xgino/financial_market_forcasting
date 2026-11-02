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
- [x] Full battle run done (2026-11-02): 21 models × 8 assets × 5/20/60 days, then steps 5-7, 0 errors
- [x] Yahoo's rounding after a rescale (~0.000003 per price) no longer looks like changed old prices: months are compared with a 0.1% margin, not an exact fingerprint (it redid all history from 2003 on every run)
- [x] Judge: skips empty columns in its monthly fit (the early years have no macro yet; sklearn crashed)
- [x] One run at a time (`core/runs.py`, `data/runs/running.json`): a second `main.py` stops right away; two at once had broken `skillbook_h60.csv`
- [x] Every CSV is written to a temp file first and then swapped in: a stopped run never leaves half a file
- [x] Skill book pooling 5× faster (same numbers, checked on 1,800 snapshots); the monthly call on every past month is saved by step 5 (`data/selection/past_calls.csv`)

### Dashboard v3 (2026-11-02)
- [x] One file per page in `app/pages/`, shared code in `app/common.py`; `dashboard.py` only lists the pages
- [x] Overview page (KPIs): a signal card per asset (best guess vs the market's normal swing and the costs, instead of only a bad .. good range), the best strategy vs buy & hold, how many models proved an edge per horizon, the judge's verdict
- [x] Forecasts page: price + every model's forecast ahead (5 / 20 / 60 days), every past guess against what really happened (development years), right direction / average miss / range / IC per model, and the long-term view (held for 1 month to 10 years)
- [x] Runs in the background from the sidebar on every page: Update now, Keep updating every day (`python main.py 1-7 --every 24`), Stop, live status; the page reloads with the new results when a run ends
- [x] This month page: the old Now page with the signal in euros and how often this call was right before

## Results of the full run (development years 2010-2018, holdout hidden)
- Nothing beats the plan yet: `risk_balanced` (Sharpe 1.23, worst drop -8%) and `fixed_mix` (+12.1% per year) use no forecasts; every strategy that follows forecasts lands around +9% per year
- Proven edge: `theta` on **5 days** (trust 3.9, positive 9 of 9 years, luck 1%) and `gbm` on 20 days (trust 2.1, luck 0.5%). Nothing on 60 days
- The monthly rule said BUY 0 times and WAIT 2 times in 1,197 past asset-months: with these models it almost always says KEEP YOUR PLAN
- Pretrained models (Chronos, TimesFM, TiRex, Kronos, Moirai 2, Time-MoE) can't be judged yet: they only count after their release (2024-2025), which is inside the holdout

## Next: better decisions from the same historical data

### 1. Score "is it a bad time to buy?" directly
- [x] New score per model: when it said "fall" (middle guess below -0.5%), how often did it really fall, and how much lower than on any day? (`warn_*` in `core/metrics.py`)
- [x] Same for "rise" (middle guess beats buy + sell costs): how often did it rise? (`rise_*`)
- [x] Both in the skill book, per situation and per asset, month by month; trust = edge in units of luck (z ≥ 2); leaderboard adds a shift luck test
- [x] WAIT on the Now page weighs fall votes by the warning record and needs a proven warner (`core/verdict.py`); BUY still uses IC trust
- [x] Dashboard: warning record table on Leaderboard, a line per model on Models, the warner's record on Now when it says WAIT
- [x] After the full battle run: `python main.py 4-6` once more (done 2026-11-02, the skill book has the new columns)
- [x] Looked at it: **no model warns better than luck**. Best warn trust: history_range 1.5, theta 1.2 (needs 2); analog passes the luck test (1.5%) but trust is only 1.0. WAIT came 2 times in 1,197 asset-months
- [ ] Try a stricter fall line (middle guess below -2% instead of -0.5%): fewer calls, maybe a clearer record

### 2. Select on similar days instead of 6 fixed boxes
- [x] Describe each day with numbers: trend strength, swings now vs normal, last 20/60 day shape, distance from the 1 year high, real rate, dollar (`core/similar.py`)
- [x] For today: the 150 most similar past days of that asset whose results were already known (scaled with past days only)
- [x] Score every model only on those days: trust over independent 20 day periods (fewer/clustered days = higher bar) + a luck test that shuffles whole periods
- [x] New selector `best_in_similar_days` next to `best_in_situation`, and strategy `follow_similar` next to `follow_best`, so steps 5 and 6 compare them
- [x] Tests: similar days only have known results, garbage after the date changes nothing; the selection no-future test covers the new selector
- [x] Compared (2026-11-02): similar days is the only grouping with a positive IC (+0.046, move when up +0.26% vs all), but it has an opinion only 10% of the months; `follow_similar` +9.0% per year vs `follow_best` +8.9%, Sharpe 1.04 vs 1.05: no real difference
- [ ] Try `days` 100 / 150 / 250 in its SETTINGS: does a smaller, closer set help or just add luck?

### 3. More evidence: pool related assets
Found: trust was already pooled, over ALL assets equally (gold's trust counted Bitcoin's results as much as gold's).
- [x] Groups in settings (`settings.GROUPS`): metals (gold, silver), stocks (sp500, nasdaq, aex), bonds, crypto
- [x] Own asset counts fully, its group at half weight, other assets not (`settings.POOLS`, `settings.EVIDENCE = "group"`); periods add up weighted, so borrowed evidence raises trust less (`core/pooling.py`)
- [x] Skill book has rows per asset and situation; selectors, the Now verdict and the email use the pooled record
- [x] Check in step 4 (`data/battle/pooling.csv`, Leaderboard page): own / group / all, does trust on 1 January predict that year's IC, and how steady is it
- [x] Read the check (2026-11-02): `all` predicts next year's IC best (+0.26), `group` -0.09, `own` -0.04 (9 years)
- [ ] Set `settings.EVIDENCE = "all"`, then `python main.py 5-7` (only 9 years of evidence: check again after the holdout)
- [ ] Platinum and gold miners as traded (battled) assets, so metals has 4 members (costs ~2 more assets of battle time)

### 4. A judge model on top of the models (stacking)
- [x] Inputs: every other model's middle guess, their average and spread, and the 7 similar-days numbers; output: the 20 day result (`core/stacking.py`, `04_models/judge.py`)
- [x] Walk-forward: refit every month inside step 4 (its inputs only exist there), on all traded assets, only results that ended before that month; pretrained guesses up to `seen_until` count as "no opinion"
- [x] Runs after every other model; its cache follows the other models' cache keys
- [x] Followed (selectors) and voting (Now verdict) only while its record so far beats the best single model's record so far (`stack_ok`); the leaderboard says whether it beats the best on the development years
- [x] Tests: no future (prices, macro and other models' forecasts after the date), memory years ignored, the beat-the-best gate
- [x] After the full battle run: **the judge doesn't beat the best single model** on any horizon (20 days: IC +0.031 vs theta +0.097; 5 days -0.001; 60 days -0.011). The gate keeps it out of the decisions, as designed
- [ ] Try fewer inputs (only models with some trust) or a plainer learner (ridge)

### 5. More than one horizon
- [x] `settings.HORIZONS = [5, 20, 60]`: step 4 battles every horizon; decisions and strategies stay on `HORIZON_DAYS = 20`
- [x] Own journal, skill book (with the warning record) and leaderboard per horizon (`*_h5.csv`, `*_h60.csv`); forecasts cached as `<model>_h5.csv`; luck test shifts further for longer horizons
- [x] Which models join: SETTINGS `horizons` (gbm and linear only 20: trained on 20 day results; Kronos and Time-MoE only 20: slow); the judge reads the same horizon's forecasts
- [x] Dashboard: Now page "This week (5 days)" and "This quarter (60 days)" calls; Leaderboard page has a horizon picker
- [x] Tests: who joins which horizon, no future on 5 and 60 days, a 5 day range is narrower than a 60 day one
- [x] After the full battle run: **theta on 5 days** has the strongest record found so far (trust 3.9, luck 1%, positive 9 of 9 years); ets and analog on 5 days "maybe". Nothing proven on 60 days
- [ ] Let decisions use the 5 day horizon too: a weekly "is this a good week to buy?" from theta
- [ ] Train gbm and linear per horizon too (03_training on 5 and 60 day results)
- [ ] If 5 days shows real skill: a weekly strategy (more trades, more fees: does it still win after costs?)

### 6. Pattern chunks instead of calendar chunks
- [x] Cut each price line into stretches where it behaved the same: online change-point detection, a new stretch when the last 20 days' average move (z > 3) or swings (2x) differ from the stretch so far; a day's label is never revised (`core/stretches.py`, `settings.STRETCHES`)
- [x] Label by shape and size against the asset's normal swings: up-steady, up-fast, down-steady, down-sharp, sideways-calm, sideways-wild; in the journal (`stretch`) and the situations files
- [x] Skill book scores every model per stretch kind (`stretch:<kind>`, pooled per asset like the situations); selector `best_in_stretch`, strategy `follow_stretch`; Now page shows the stretch and since when
- [x] Step 5 prints which grouping finds the model to follow best: none / situations / stretches / similar days (`data/selection/groupings.csv`)
- [x] Tests: stretches never change when the future changes, stretch scores use the right forecasts and don't double count, selection no-future test covers `best_in_stretch`
- [x] Read the groupings table (2026-11-02): stretches IC -0.184 and move when up -0.04% vs all: **stretches don't help** as they are. No grouping -0.413, situations never found a trusted model
- [ ] Tune `settings.STRETCHES` only on the development years (gold: 36 stretches since 2003, median ~100 days, sideways-wild 39%)

### 7. Fine-tune the pretrained models (last: slow, small expected gain)
- [x] Fine-tune Chronos-Bolt small on your training assets, walk-forward: a checkpoint per year from its release (2025-01-01, 2026-01-01), only windows whose 20 days ended before the cutoff, the last ~6 months held back for a check (`03_training/chronos_finetune.py`, ~2 min per checkpoint)
- [x] Held-back check (Training page): 2025 fine-tuned 5.857 vs zero-shot 5.873 (0.3% better), 2026 4.478 vs 4.411 (1.5% worse)
- [x] `chronos_small_ft` in the battle against `chronos_small` (same model, zero shot), from 2025 on; a year whose fine-tuning didn't help (known at its cutoff) uses the base model
- [x] Weights (~190 MB each) kept out of git; `index.csv` stays in
- [ ] Does `chronos_small_ft` beat `chronos_small` on 2025-2026? Can't tell yet: both only count from 2025, which is inside the holdout (the battle ran, forecasts are cached). Judge it when the holdout is opened
- [ ] If it helps: fine-tune Chronos-2 with its built-in `fit()` (LoRA), same walk-forward

## Housekeeping
- [ ] `python main.py 4 --only x` overwrites journal, leaderboard and skill book with only model x: merge into the full ones instead
- [x] Dashboard Run page: runs in the background with a log file (`data/runs/live.log`), keeps going when you click around or close the dashboard
- [ ] 4 battle files are over GitHub's 100 MB limit (journal.csv, journal_h5.csv, journal_h60.csv, skillbook.csv): keep them out of git (they're rebuilt from the cached forecasts) or use Git LFS
- [x] `data/` and `__pycache__/` stay in git on purpose
- [ ] `tools/` (monthly reminder, backup) and `secrets.example.env` are mentioned in 07_judge and 08_notify, but missing
- [ ] Moirai 2 install: `uni2ts` needs `--no-deps`; make that one command (setup script)

## Later
- [ ] The final test: `python main.py 4-6 --holdout`, once, when models and strategies are final
- [ ] Live paper trading month by month; real money only after `settings.REAL_MONEY_RULE_MONTHS` good live months
