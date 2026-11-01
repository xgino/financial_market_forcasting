# paper-fund-v2/02_features/_step.py
"""Step 2: every feature maker in this folder turns the prices into features, saved as CSV."""
from pathlib import Path

from core.contracts import FeatureMaker
from core.plugins import load_players
from core.timeline import Timeline

STEP = "Features"
FOLDER = Path(__file__).resolve().parent


def run(args):
    tl = Timeline.load()
    out = {}
    for name, p in load_players(FOLDER, FeatureMaker, only=args.only).items():
        print(f"- {name}: {p.about}")
        for k, v in p.run(tl).items():
            print(f"    {k}: {v}")
        out[name] = "ok"
    return out