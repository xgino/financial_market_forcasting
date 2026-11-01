# paper-fund-v2/08_notify/_step.py
"""Step 8: every notifier in this folder sends the month's news (only when it matters, see settings.NOTIFY)."""
from pathlib import Path

from core.contracts import Notifier
from core.plugins import load_players

STEP = "Notify"
FOLDER = Path(__file__).resolve().parent


def run(args):
    out = {}
    for name, p in load_players(FOLDER, Notifier, only=args.only).items():
        print(f"- {name}: {p.about}")
        report = p.run(args)
        for k, v in report.items():
            print(f"    {k}: {v}")
        out[name] = report.get("mail", "")
    return out