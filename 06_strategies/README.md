# 06_strategies: what to do with the money

Every player is a strategy. It gets EUR 1000, follows one selector (`SETTINGS["follow"]`) and once a month
says how much of the fund goes in each asset. Trades happen at the next day's open with fees; trades
smaller than `settings.NO_TRADE_BAND` are skipped.

- `buy_and_hold.py` equal parts on day one, never trade (the benchmark)
- `fixed_mix.py` always 80% invested, rebalanced (no forecasts)
- `risk_balanced.py` sized by the range: less in wild assets
- `follow_best.py` / `follow_bandit.py` base 60% plus or minus a 30% tilt when a trusted model expects a clear move
- `cppi_floor.py` your "house money" idea: protect 85% of the best level so far, risk 3x what is above it

KPIs and pass marks (`settings.PASS`) in `data/simulation/kpis.csv`, and what each strategy would do today
with a fresh EUR 1000 in `data/simulation/now.csv` (dashboard page Now).
Add a strategy: a class based on `Strategy` with `targets(fund, picks)` returning `({asset: share}, {asset: why})`.