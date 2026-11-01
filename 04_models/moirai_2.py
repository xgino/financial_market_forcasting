# paper-fund-v2/04_models/moirai_2.py
"""Salesforce Moirai 2.0 small (11M), pretrained. Tiny, quantile forecasts, runs on the CPU (needs float64)."""
SETTINGS = {
    "enabled": True,
    "checkpoint": "Salesforce/moirai-2.0-R-small",
    "context_days": 512,
    "seen_until": "2025-11-30",   # weights out Aug 2025, paper Nov 2025: the later date, to be safe
}
CARD = {  # shown on the dashboard page Models
    "type": 'pretrained by Salesforce (zero shot, not fine-tuned)',
    "trained": 'Trained by Salesforce on 36M public series; released 2025. Not fine-tuned by us. Research licence.',
    "strengths": 'High on the public forecasting benchmark (GIFT-Eval) while being tiny and fast.',
    "weaknesses": 'Not tested here yet.',
    "limits": "Can't explain itself. Only trusted after its release date.",
    "source": 'https://arxiv.org/abs/2511.11698',
}

from core.pretrained import Moirai2


class Moirai2Small(Moirai2):
    pass
