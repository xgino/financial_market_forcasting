# financial_market_forcasting1/05_selection/_step.py
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
from core.timeline import Timeline

STEP = "Who to trust"
FOLDER = Path(__file__).resolve().parent


def run(args):
    journal = store.read_csv(store.path("battle", "journal.csv"), dates=("date", "end_date"))
    book = store.read_csv(store.path("battle", "skillbook.csv"), dates=("as_of",))
    selectors = load_players(FOLDER, selection.Selector, only=args.only)
    print(f"{len(selectors)} selectors: {', '.join(selectors)}")
    tl = Timeline.load(sorted(journal["asset"].unique()))   # for best_in_similar_days: describes each day
    choices = selection.run(journal, book, selectors, tl)
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
    groupings = {"best_overall": "no grouping (all days)", "best_in_situation": "6 situations (fixed 120/20 day windows)",
                 "best_in_stretch": "stretches (pattern chunks)", "best_in_similar_days": "similar days"}
    cmp = scores.reindex([g for g in groupings if g in scores.index])
    if len(cmp):
        cmp = pd.DataFrame({"grouping": [groupings[g] for g in cmp.index], "months": cmp["months"].astype(int),
                            "has opinion": cmp["has_opinion"].map("{:.0%}".format),
                            "IC": cmp["ic"].map(lambda x: "" if pd.isna(x) else f"{x:+.3f}"),
                            "right vs always up": (cmp["direction"] - cmp["always_up"]).map(lambda x: "" if pd.isna(x) else f"{x:+.1%}"),
                            "move when up vs all": (cmp["move_when_up"] - cmp["move_all"]).map(lambda x: "" if pd.isna(x) else f"{x:+.2%}")},
                           index=cmp.index)
        store.write_csv(cmp, store.path("selection", store.tag("groupings") + ".csv"))
        print("\nWhich grouping finds the model to follow best? (same rule: most trusted model in today's group)")
        print(cmp.to_string())
    for name in ("best_in_situation", "best_in_stretch", "best_in_similar_days"):
        best = choices[choices["selector"] == name]
        if len(best):
            print(f"\nWhat {name} followed:", best["main_model"].value_counts().to_dict())
    return {"selectors": len(selectors)}