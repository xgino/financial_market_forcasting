# paper-fund-v2/04_models/_step.py
"""
Step 4, the battle. Every model in this folder forecasts every trading day as if live (only the past is
visible), gets scored once the 20 days are over, and the leaderboard ranks them.

Saves in data/battle/:
  journal.csv      every forecast of every model: date, asset, situation, price, p10/p50/p90, actual
  leaderboard.csv  the ranking + a verdict per model (development period only)
  skillbook.csv    month by month: each model's score per situation, only what was known then
  winners.csv      per asset per month: whose price guess came closest
"""
from pathlib import Path

import pandas as pd

import settings
from core import battle, store
from core.contracts import Model
from core.plugins import load_players
from core.timeline import Timeline

STEP = "Models battle"
FOLDER = Path(__file__).resolve().parent


def fmt(t):
    out = pd.DataFrame(index=t.index)
    out["rank"] = t["rank"].astype(int)
    out["kind"] = t["kind"]
    out["periods"] = t["periods"].round().astype("Int64")
    out["IC"] = t["ic"].map(lambda x: "" if pd.isna(x) else f"{x:+.3f}")
    out["trust"] = t["trust"].map(lambda x: "" if pd.isna(x) else f"{x:+.1f}")
    out["years +"] = t["years_positive"].fillna("")
    out["luck beats it"] = t["luck_beats_it"].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    out["direction"] = t["direction"].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    out["always up"] = t["always_up"].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    out["skill"] = t["skill"].map(lambda x: "" if pd.isna(x) else f"{x:+.1%}")
    out["range"] = t["range_hit"].map(lambda x: "" if pd.isna(x) else f"{x:.0%}")
    out["verdict"] = t["verdict"]
    return out


def run(args):
    tl = Timeline.load()
    models = load_players(FOLDER, Model, settings.HORIZON_DAYS, only=args.only)
    print(f"{len(models)} models: {', '.join(models)}")

    if args.check:
        ok = True
        for asset in tl.assets:
            for name, m in models.items():
                for d, same in battle.check_no_future(m, tl, asset):
                    ok &= same
                    print(f"  {asset:<7} {name:<16} {d.date()}: {'OK' if same else 'CHANGED, sees the future!'}")
        print("RESULT: no model sees the future" if ok else "RESULT: LOOKAHEAD FOUND")
        return {"no_future": ok}

    journal = battle.build_journal(models, tl)
    store.write_csv(journal, store.path("battle", "journal.csv"), index=False)
    book = battle.skillbook(journal, tl)
    board = battle.leaderboard(journal, models)
    store.write_csv(board, store.path("battle", store.tag("leaderboard") + ".csv"))
    wins = battle.winners(battle.dev_window(journal))
    store.write_csv(wins, store.path("battle", store.tag("winners") + ".csv"), index=False)

    first, last = int(settings.WARMUP_UNTIL[:4]) + 1, int(settings.HOLDOUT_FROM[:4]) - 1
    shown = "INCLUDES HOLDOUT" if settings.SHOW_HOLDOUT else f"{first} to {last}, holdout hidden"
    print(f"\nLeaderboard ({shown}). Pretrained models only count after their seen_until date.")
    print(fmt(board).to_string())
    print("\nMonthly wins (closest price guess), development period:")
    print(wins["winner"].value_counts().to_string())
    print("\nA win in one month is mostly luck. The leaderboard (IC, trust, luck test) is the real ranking.")
    return {"models": len(models), "leader": board.index[0] if len(board) else None}