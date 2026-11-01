# financial_market_forcasting1/07_judge/_step.py
"""
Step 7, the judge. Every check in this folder looks at one area and writes plain sentences.
Saves data/judge/report.md (read it in the dashboard, page Judge) and adds every finding to
data/judge/history.csv, so you can see how the verdict changed over time.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

from core import store
from core.contracts import Check
from core.plugins import load_players

STEP = "Judge"
FOLDER = Path(__file__).resolve().parent
ICON = {"good": "✅", "info": "ℹ️", "warn": "⚠️", "bad": "❌"}


def run(args):
    now = datetime.now().isoformat(timespec="seconds")
    found = []
    for name, c in load_players(FOLDER, Check, only=args.only).items():
        try:
            for level, text in c.run():
                found.append({"time": now, "area": c.area, "level": level, "text": text})
        except Exception as e:
            found.append({"time": now, "area": c.area, "level": "bad", "text": f"check {name} failed: {e}"})
    worst = "bad" if any(f["level"] == "bad" for f in found) else "warn" if any(f["level"] == "warn" for f in found) else "good"
    verdict = {"good": "All good. Keep following the plan and keep the live record going.",
               "warn": "Mostly fine, a few things need a look (see ⚠️).",
               "bad": "Something needs fixing before you trust the results (see ❌)."}[worst]
    lines = [f"# Judge report, {now[:10]}", "", f"**Verdict:** {verdict}", ""]
    for area in dict.fromkeys(f["area"] for f in found):
        lines += [f"## {area}", ""] + [f"- {ICON[f['level']]} {f['text']}" for f in found if f["area"] == area] + [""]
    report = "\n".join(lines)
    p = store.path("judge", "report.md")
    p.write_text(report)
    h = store.path("judge", "history.csv")
    old = pd.read_csv(h) if h.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(found)], ignore_index=True).to_csv(h, index=False)
    print(report)
    return {"verdict": worst}