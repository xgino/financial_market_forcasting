# financial_market_forcasting1/04_models/chronos_small.py
"""Amazon Chronos-Bolt small (48M), pretrained. Memory test 2026-10: IC +0.34 on 2010-2018 but -0.12 on
2025-2026, so it remembered old prices. seen_until keeps those years from counting for trust."""
SETTINGS = {
    "enabled": True,
    "checkpoint": "amazon/chronos-bolt-small",
    "context_days": 512,
    "seen_until": "2024-12-31",   # release date: forecasts up to here never earn trust
}
CARD = {  # shown on the dashboard page Models
    "type": 'pretrained by Amazon (zero shot, not fine-tuned)',
    "trained": 'Trained by Amazon on public time series; released end of 2024. We never trained or fine-tuned it.',
    "strengths": 'Strong on general forecasting benchmarks; good range on most data.',
    "weaknesses": 'Remembered old stock index prices (memory test 2026-10): great on 2010-2018, nothing on 2025-2026.',
    "limits": "Can't explain itself. Only trusted on results after its release date.",
    "source": 'https://github.com/amazon-science/chronos-forecasting',
}

from core.pretrained import Chronos


class ChronosSmall(Chronos):
    pass