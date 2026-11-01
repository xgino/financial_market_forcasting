# financial_market_forcasting1/04_models/tirex.py
"""NX-AI TiRex (35M), pretrained. An xLSTM instead of a transformer: a third family next to Chronos and TimesFM."""
SETTINGS = {
    "enabled": True,
    "checkpoint": "NX-AI/TiRex",
    "context_days": 512,
    "seen_until": "2025-05-31",
}
CARD = {  # shown on the dashboard page Models
    "type": 'pretrained by NX-AI (zero shot, not fine-tuned)',
    "trained": 'Trained by NX-AI; released May 2025. Not fine-tuned by us.',
    "strengths": 'Top of the public forecasting benchmarks (GIFT-Eval) at release; small and fast.',
    "weaknesses": 'Not tested here yet.',
    "limits": "Can't explain itself. Only trusted after its release date.",
    "source": 'https://github.com/NX-AI/tirex',
}

from core.pretrained import TiRex


class TiRexZero(TiRex):
    pass
