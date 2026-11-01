# 02_features: describe each day

Every player turns prices into numbers per day and writes CSV. Every number on day D only uses data up to D.

- `price_features.py` trend, swings, averages, RSI -> `data/features/<asset>.csv`
- `situations.py` up/flat/down x calm/wild -> `data/situations/<asset>.csv`

Models compute these themselves from their view (`core/features.py`), so the no-future check covers them.
These CSVs are for looking at the data and for the battle's per-situation scores.