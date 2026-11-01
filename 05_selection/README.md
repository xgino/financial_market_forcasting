# 05_selection: who to trust

Every player is a selector. Once a month per asset it picks what to follow, with only what was known that
day: one model, a mix, or the base plan (no opinion on direction). A model is only followed when the skill
book trusts it for today's situation (`settings.MIN_TRUST` with `settings.MIN_PERIODS`).

- `base_plan.py` never follows a model (the bar)
- `best_in_situation.py` the most trusted model for today's situation
- `top_n_blend.py` mix of the top `settings.TOP_N` trusted models
- `bandit.py` reward and punish: recent good months earn trust, bad months lose it
- `best_in_stretch.py` the most trusted model for today's stretch (`core/stretches.py`: the price line cut
  where its behaviour changed, labelled steady/fast up, steady/sharp down, sideways calm/wild)
- `best_in_similar_days.py` the most trusted model on the 150 past days most like today (`core/similar.py`:
  trend, swings, last 20/60 day shape, distance from the 1 year high, real rate, dollar). Only days whose
  result was known count; it needs trust over enough independent periods and a luck test it passes

Every choice also lists the top N models for the situation with their forecasts, trusted or not
(`data/selection/choices.csv`, column `top_models`). Step 5 also prints which grouping finds the model to
follow best: none, situations, stretches or similar days (`data/selection/groupings.csv`). Add a selector: a class based on `Selector` with `choose(ctx)`.