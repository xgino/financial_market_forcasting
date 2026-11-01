# paper-fund-v2/04_models/kronos_small.py
"""Kronos-small (25M), pretrained on market candles only. Slow (it draws 20 futures per forecast), so it only
forecasts from first_day on: earlier years can't earn trust anyway, because of seen_until."""
SETTINGS = {
    "enabled": True,
    "checkpoint": "NeoQuasar/Kronos-small",
    "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
    "context_days": 256,          # Kronos can take 512, but that was ~10x slower on the Mac GPU
    "samples": 20,                # possible futures per forecast; the range is read from them
    "seen_until": "2025-08-31",
    "first_day": "2025-09-01",
}
CARD = {  # shown on the dashboard page Models
    "type": 'pretrained on financial candles (zero shot, not fine-tuned)',
    "trained": 'Trained on candles from 45 exchanges; released August 2025. Not fine-tuned by us.',
    "strengths": 'The only model here built for markets; reads open and close, not just the close.',
    "weaknesses": 'Not tested here yet. We have no high, low or volume, so it gets less than it was trained on.',
    "limits": "Can't explain itself. Only forecasts (and is only trusted) after its release date.",
    "source": 'https://github.com/shiyu-coder/Kronos',
}

from core.pretrained import Kronos


class KronosSmall(Kronos):
    pass
