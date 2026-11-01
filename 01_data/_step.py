# financial_market_forcasting1/01_data/_step.py
"""Step 1: every data source in this folder downloads and cleans its data into data/ as CSV."""
from pathlib import Path

from core.contracts import DataSource
from core.plugins import load_players

STEP = "Data"
FOLDER = Path(__file__).resolve().parent


def run(args):
    players = load_players(FOLDER, DataSource, only=args.only)
    out = {}
    for name, p in players.items():
        print(f"- {name}: {p.about}")
        report = p.run()
        for k, v in report.items():
            print(f"    {k}: {v}")
        out[name] = "ok"
    return out