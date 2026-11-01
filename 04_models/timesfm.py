# financial_market_forcasting1/04_models/timesfm.py
"""Google TimesFM 2.5 (200M), pretrained. A different company than Chronos: a second opinion."""
SETTINGS = {
    "enabled": True,
    "checkpoint": "google/timesfm-2.5-200m-pytorch",
    "context_days": 512,
    "seen_until": "2025-09-30",
}
CARD = {  # shown on the dashboard page Models
    "type": 'pretrained by Google (zero shot, not fine-tuned)',
    "trained": 'Trained by Google; TimesFM 2.5 released September 2025. Not fine-tuned by us.',
    "strengths": 'A second opinion from another company; useful to catch memory in one family.',
    "weaknesses": 'No direction skill found; range too narrow on stock indices.',
    "limits": "Can't explain itself. Only trusted after its release date.",
    "source": 'https://github.com/google-research/timesfm',
}

from core.pretrained import TimesFM


class TimesFM25(TimesFM):
    pass