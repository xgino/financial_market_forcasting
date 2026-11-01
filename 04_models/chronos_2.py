# paper-fund-v2/04_models/chronos_2.py
"""Amazon Chronos-2 (120M), the newest Chronos, pretrained."""
SETTINGS = {
    "enabled": True,
    "checkpoint": "amazon/chronos-2",
    "context_days": 512,
    "seen_until": "2025-10-31",
}
CARD = {  # shown on the dashboard page Models
    "type": 'pretrained by Amazon (zero shot, not fine-tuned)',
    "trained": 'Trained by Amazon; released October 2025. Not fine-tuned by us.',
    "strengths": 'Newest Chronos; can take extra input series (macro) in a later version of this project.',
    "weaknesses": 'No direction skill found so far.',
    "limits": "Can't explain itself. Only trusted after its release date.",
    "source": 'https://arxiv.org/abs/2510.15821',
}

from core.pretrained import Chronos


class Chronos2(Chronos):
    pass