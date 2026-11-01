# paper-fund-v2/04_models/timemoe.py
"""Time-MoE 200M (453M in total, mixture of experts), pretrained. Only a middle guess: the range is added from
the last 60 days' swings. The 2.4B version from the paper was never published."""
SETTINGS = {
    "enabled": True,
    "checkpoint": "Maple728/TimeMoE-200M",
    "revision": "794591bfeb1225fdf742cec0f4c71f20c3f3b87e",   # the weights the copied code in core/timemoe fits
    "context_days": 512,
    "swing_days": 60,
    "seen_until": "2024-09-30",
}
CARD = {  # shown on the dashboard page Models
    "type": 'pretrained by the Time-MoE authors (zero shot, not fine-tuned)',
    "trained": 'Trained on Time-300B (300 billion points of public data); released September 2024. Not fine-tuned by us.',
    "strengths": 'The biggest model here; a mixture of experts picks a specialist part per pattern.',
    "weaknesses": 'Not tested here yet. Gives no range itself: ours is a plain bell curve from recent swings.',
    "limits": "Can't explain itself. Only trusted after its release date.",
    "source": 'https://github.com/Time-MoE/Time-MoE',
}

from core.pretrained import TimeMoE


class TimeMoE200M(TimeMoE):
    pass
