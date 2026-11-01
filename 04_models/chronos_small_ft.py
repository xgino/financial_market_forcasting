# financial_market_forcasting1/04_models/chronos_small_ft.py
"""Chronos-Bolt small, FINE-TUNED on your assets (03_training/chronos_finetune.py): one checkpoint per year,
each only trained on data from before its date. Battles chronos_small (the same model, zero shot)."""
SETTINGS = {
    "enabled": True,
    "trainer": "chronos_finetune",   # data/checkpoints/<trainer>/
    "context_days": 512,
    "seen_until": "2024-12-31",      # the base model's release: same memory as chronos_small
    "first_day": "2025-01-01",       # = the first checkpoint: before it there is nothing fine-tuned
    "only_if_helped": True,          # a year whose fine-tuning didn't lower the held-back loss uses the base model
    "base": "amazon/chronos-bolt-small",
}
CARD = {  # shown on the dashboard page Models
    "type": 'pretrained by Amazon, fine-tuned by you',
    "trained": 'Chronos-Bolt small, then fine-tuned walk-forward on your training assets: one checkpoint per year, '
               'only windows whose 20 days had ended before it. See the Training page: held-back loss zero-shot vs fine-tuned.',
    "strengths": 'Learned the shape of your assets (EUR prices, daily, 20 days ahead) on top of the general model.',
    "weaknesses": 'Only from 2025 on (its release), so few results yet. Fine-tuning can also learn noise.',
    "limits": "Can't explain itself. Compare it with chronos_small: same model, same days, zero shot.",
    "source": 'https://github.com/amazon-science/chronos-forecasting',
}

from core.pretrained import ChronosFinetuned


class ChronosSmallFinetuned(ChronosFinetuned):
    pass
