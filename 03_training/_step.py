# financial_market_forcasting1/03_training/_step.py
"""Step 3: every trainer in this folder trains one of your own models (walk-forward checkpoints)."""
from pathlib import Path

from core.contracts import Trainer
from core.plugins import load_players
from core.timeline import Timeline

STEP = "Training"
FOLDER = Path(__file__).resolve().parent


def run(args):
    import settings
    from core import store
    have = [a for a, spec in settings.ALL_ASSETS.items()
            if spec.get("train", True) and store.path("prices", f"{a}.csv").exists()]
    tl = Timeline.load(have)  # traded assets + training-only assets, minus "train": False (crypto)
    print(f"training on {len(have)} assets: {', '.join(have)}")
    out = {}
    for name, p in load_players(FOLDER, Trainer, only=args.only).items():
        print(f"- {name}: {p.about}")
        report = p.train(tl, fresh=args.fresh)
        print(f"    {report}")
        out[name] = report
    return out