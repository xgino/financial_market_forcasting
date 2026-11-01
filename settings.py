# paper-fund-v2/settings.py
"""
Global settings, in one place. Every file in a step folder (a model, a strategy, ...) also has its
own SETTINGS at the top for what only that file uses.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"                     # everything the steps write, as CSV (wallets as SQLite)

# ----- money -----
START_EUR = 1000
FEE_RATE = 0.0025                        # 0.25% on every buy and every sell

# ----- time -----
HORIZON_DAYS = 20                        # every forecast looks 20 trading days (~1 month) ahead
WARMUP_UNTIL = "2009-12-31"              # models predict and get scored, nothing is judged yet
HOLDOUT_FROM = "2019-01-01"              # never look at this period until the very end
SHOW_HOLDOUT = False                     # True only for the final test

# ----- assets: all converted to EUR. USD data starts end of 2003 (EUR/USD) -----
ASSETS = {
    "gold":   {"ticker": "GC=F", "currency": "USD"},   # gold futures, per ounce
    "silver": {"ticker": "SI=F", "currency": "USD"},
    "sp500":  {"ticker": "SPY",  "currency": "USD"},   # 500 biggest US companies
    "nasdaq": {"ticker": "QQQ",  "currency": "USD"},   # mostly US tech
    "bonds":  {"ticker": "TLT",  "currency": "USD"},   # 20+ year US government bonds
    "aex":    {"ticker": "^AEX", "currency": "EUR"},   # Dutch AEX index
    # Crypto trades 7 days a week, so its 20 days ahead are ~3 weeks, not a month. "train": False keeps it
    # out of your own models' training (03_training): 3 to 5x wilder than the rest, it would pull them its way.
    "bitcoin":  {"ticker": "BTC-USD", "currency": "USD", "train": False},   # from 2014
    "ethereum": {"ticker": "ETH-USD", "currency": "USD", "train": False},   # from end of 2017
}

# ----- training-only assets: more history for your own models (03_training), never traded or shown -----
# More different markets = more examples to learn from. Same rules: daily, converted to EUR.
# Crypto is left out on purpose (see "train": False above): 3 to 5x wilder than everything else and only
# ~11 years of data, it would pull your model toward crypto behaviour. Set "train": True to test that yourself.
TRAIN_ONLY_ASSETS = {
    "japan":     {"ticker": "EWJ",  "currency": "USD"},   # Japanese stocks
    "germany":   {"ticker": "EWG",  "currency": "USD"},
    "uk":        {"ticker": "EWU",  "currency": "USD"},
    "canada":    {"ticker": "EWC",  "currency": "USD"},
    "emerging":  {"ticker": "EEM",  "currency": "USD"},   # emerging markets
    "energy":    {"ticker": "XLE",  "currency": "USD"},   # US energy companies
    "finance":   {"ticker": "XLF",  "currency": "USD"},
    "tech":      {"ticker": "XLK",  "currency": "USD"},
    "bonds_mid": {"ticker": "IEF",  "currency": "USD"},   # 7-10 year US government bonds
    "platinum":  {"ticker": "PL=F", "currency": "USD"},
    "copper":    {"ticker": "HG=F", "currency": "USD"},
    # Long histories in their own currency ("LOCAL" = no conversion). For training only the moves matter,
    # not the currency, so these can go back decades: many more examples, also from very different times.
    "us_long":     {"ticker": "^GSPC",  "currency": "LOCAL", "until": "2003-11-30"},  # S&P 500 from 1927, until sp500 starts
    "dow":         {"ticker": "^DJI",   "currency": "LOCAL"},
    "nasdaq_comp": {"ticker": "^IXIC",  "currency": "LOCAL", "until": "2003-11-30"},  # from 1971, until nasdaq starts
    "russell":     {"ticker": "^RUT",   "currency": "LOCAL"},  # small US companies
    "dax":         {"ticker": "^GDAXI", "currency": "LOCAL"},  # Germany, from 1987
    "ftse":        {"ticker": "^FTSE",  "currency": "LOCAL"},  # UK
    "nikkei":      {"ticker": "^N225",  "currency": "LOCAL"},  # Japan, from 1965
    "hang_seng":   {"ticker": "^HSI",   "currency": "LOCAL"},  # Hong Kong
    "cac":         {"ticker": "^FCHI",  "currency": "LOCAL"},  # France
    "oil_co":      {"ticker": "XOM",    "currency": "LOCAL"},  # Exxon, one of the oldest listed oil companies
    "gold_miners": {"ticker": "GDX",    "currency": "LOCAL"},
    "silver_etf":  {"ticker": "SLV",    "currency": "LOCAL"},
    "reits":       {"ticker": "VNQ",    "currency": "LOCAL"},  # property
    "utilities":   {"ticker": "XLU",    "currency": "LOCAL"},
    "health":      {"ticker": "XLV",    "currency": "LOCAL"},
    "staples":     {"ticker": "XLP",    "currency": "LOCAL"},
    "corp_bonds":  {"ticker": "LQD",    "currency": "LOCAL"},
    "tips":        {"ticker": "TIP",    "currency": "LOCAL"},  # inflation-protected bonds
}
ALL_ASSETS = {**ASSETS, **TRAIN_ONLY_ASSETS}

# ----- macro data from FRED (01_data/fred_macro.py): information that isn't in the price line -----
MACRO_SERIES = {
    "real_rate": "DFII10",    # 10 year US real interest rate (after inflation), important for gold
    "rate10": "DGS10",        # 10 year US interest rate
    "curve": "T10Y2Y",        # 10 year minus 2 year rate: below 0 has often come before recessions
    "dollar": "DTWEXBGS",     # US dollar against other currencies (from 2006)
    "vix": "VIXCLS",          # the "fear index": expected swings of the S&P 500
}
MACRO_LAG_DAYS = 1            # a value for day D is only used from the next trading day (publication delay)

# ----- trust: when does a model count as better than luck -----
MIN_PERIODS = 20                         # independent results needed (forecasts / horizon)
MIN_TRUST = 2.0                          # IC x sqrt(periods), above 2 = probably not luck

# ----- situations (02_features/situations.py) -----
SITUATIONS = {
    "trend_days": 120,                   # trend over the last ~6 months
    "trend_cutoff": 0.5,                 # strength above +0.5 = up, below -0.5 = down
    "swing_days": 20,                    # last month's swings...
    "swing_lookback": 504,               # ...vs the typical level of the last 2 years
}

# ----- hardware -----
DEVICE = "auto"                          # auto = Mac GPU (mps) if there is one, else CPU

# ----- the monthly verdict on the Now page: do the models agree? -----
VOTE_AGREE = 0.6                         # share of (trust-weighted) votes that must point the same way
VOTE_NEED_TRUSTED = True                 # BUY / WAIT only when at least one trusted model agrees

# ----- 05_selection: who to trust -----
TOP_N = 3                                # how many models to show / blend per situation
EXCLUDE = ["naive"]                      # never followed for direction (naive has no opinion by design)
RANGE_MODEL = "naive"                    # whose bad..good range the base plan uses (naive held ~83% on 2025-26)
QUARANTINE = {}                          # manual override {model: reason}: never followed for direction

# ----- 06_strategies: what to do with the money -----
NO_TRADE_BAND = 0.02                     # skip a trade smaller than 2% of the fund (saves fees)
SPREAD = 0.0005                          # extra cost per trade on top of the fee: half the buy/sell price gap
MIN_EDGE = 0.005                         # a move has to beat all costs by this much to count as a reason
# Box 3 (NL): tax on wealth above the tax-free allowance. For a small fund this is usually EUR 0.
# Check belastingdienst.nl for your own situation and fill in the effective yearly % of the fund here.
BOX3_RATE = 0.0
CASH_RATE = 0.0                          # yearly interest on cash (0.02 = 2%), the base plan earns it too

# ----- pass marks: what counts as good enough -----
PASS = {
    "trust": 2.0,                        # model: bigger than luck
    "luck": 0.05,                        # model: luck beats it less than 5% of the time
    "years": 2 / 3,                      # model: positive in at least 2 of every 3 years
    "skill": 0.0,                        # model: beats "no change"
    "range": (0.70, 0.90),               # model: real result inside bad..good about 80% of the time
    "beat_hold": 0.0,                    # strategy: ends above buy & hold (risk adjusted, see Sharpe)
    "max_drop": -0.25,                   # strategy: worst drop not deeper than -25%
    "sharpe": 0.5,                       # strategy: return per unit of swings, above 0.5 is decent, above 2 suspicious
}

# ----- the rule for yourself -----
REAL_MONEY_RULE_MONTHS = 12              # real money only after this many live months that beat the base plan

# ----- 08_notify: the monthly email -----
NOTIFY = {
    "assets": ["gold"],          # which assets you want mail about
    "only_when": ["BUY"],        # mail only on these calls ("BUY", "WAIT", "KEEP YOUR PLAN")
    "min_confidence": 0,         # and only when the confidence is at least this (0 to 100)
    "amount": 1000,              # the example amount in the mail
    "dashboard_url": "http://localhost:8501",
}
# The mail account goes in secrets.env (next to settings.py, never in git). See secrets.example.env.