# financial_market_forcasting1/04_models/chronos_base.py
"""Amazon Chronos-Bolt base (205M), pretrained. Memory test 2026-10: IC +0.54 on 2010-2018 but -0.04 on
2025-2026, so it remembered old prices. seen_until keeps those years from counting for trust."""
SETTINGS = {
    "enabled": True,
    "checkpoint": "amazon/chronos-bolt-base",
    "context_days": 512,
    "seen_until": "2024-12-31",
}
CARD = {  # shown on the dashboard page Models
    "type": 'pretrained by Amazon (zero shot, not fine-tuned)',
    "trained": 'Trained by Amazon on public time series; released end of 2024. We never trained or fine-tuned it.',
    "strengths": 'Bigger version of chronos_small.',
    "weaknesses": 'Remembered even more (bigger = more memory): IC +0.54 old years, -0.04 unseen.',
    "limits": "Can't explain itself. Only trusted after its release date.",
    "source": 'https://github.com/amazon-science/chronos-forecasting',
}

from core.pretrained import Chronos


class ChronosBase(Chronos):
    pass