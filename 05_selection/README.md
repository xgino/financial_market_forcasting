# 05_selection: who to trust

Every player is a selector. Once a month per asset it picks what to follow, with only what was known that
day: one model, a mix, or the base plan (no opinion on direction). A model is only followed when the skill
book trusts it for today's situation (`settings.MIN_TRUST` with `settings.MIN_PERIODS`).

- `base_plan.py` never follows a model (the bar)
- `best_in_situation.py` the most trusted model for today's situation
- `top_n_blend.py` mix of the top `settings.TOP_N` trusted models
- `bandit.py` reward and punish: recent good months earn trust, bad months lose it

Every choice also lists the top N models for the situation with their forecasts, trusted or not
(`data/selection/choices.csv`, column `top_models`). Add a selector: a class based on `Selector` with `choose(ctx)`.