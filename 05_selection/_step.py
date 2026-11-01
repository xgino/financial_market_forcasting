# paper-fund-v2/05_selection/_step.py
"""
Step 5, who to trust. Every selector in this folder picks, once a month per asset, what to follow, with
only what was known that day. Saves data/selection/choices.csv (every choice + the top N models and their
forecasts) and data/selection/scores.csv (how each selector did, development period).
"""
from pathlib import Path

import pandas as pd

import settings
from core import battle, selection, store
from core.plugins import load_players

STEP = "Who to trust"
FOLDER = Path(__file__).resolve().parent


def run(args):
    journal = store.read_csv(store.path("battle", "journal.csv"), dates=("date", "end_date"))
    book = store.read_csv(store.path("battle", "skillbook.csv"), dates=("as_of",))
    selectors = load_players(FOLDER, selection.Selector, only=args.only)
    print(f"{len(selectors)} selectors: {', '.join(selectors)}")
    choices = selection.run(journal, book, selectors)
    store.write_csv(choices, store.path("selection", "choices.csv"), index=False)

    dev = battle.dev_window(choices)
    scores = selection.evaluate(dev)
    store.write_csv(scores, store.path("selection", store.tag("scores") + ".csv"))
    show = scores.copy()
    for c in ["has_opinion", "direction", "always_up", "range_hit"]:
        show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.0%}")
    for c in ["move_when_up", "move_all", "skill"]:
        show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:+.2%}")
    show["ic"] = show["ic"].map(lambda x: "" if pd.isna(x) else f"{x:+.3f}")
    show["months"] = show["months"].astype(int)
    print("\nEvery selector, development period (a selector helps only if 'move_when_up' beats 'move_all'):")
    print(show.to_string())
    best = choices[choices["selector"] == "best_in_situation"]
    if len(best):
        print("\nWhat best_in_situation followed:", best["main_model"].value_counts().to_dict())
    return {"selectors": len(selectors)}