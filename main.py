# financial_market_forcasting1/main.py
"""
Runs the steps in order. Every step only does what's missing, so you can stop any time and run it
again: it continues where it was. Forgot a month? The next run fills in what's missing.

  python main.py                 every step in order (01_data, 02_features, 03_training, 04_models, ...)
  python main.py 4               only step 4
  python main.py 2-4             steps 2 to 4
  python main.py 4 --only naive,drift   only these players in step 4
  python main.py 4 --check       prove the players in step 4 can't see the future
  python main.py 3 --fresh       redo step 3 from scratch, ignore what's saved
  python main.py list            every step and its players
  python main.py test            run everything in tests/
  python main.py 1-7 --every 24   keep it up to date: run steps 1-7 again every 24 hours (Ctrl+C stops)
  python main.py 4-6 --holdout   THE FINAL TEST: include the holdout. Results go to separate *_holdout files.
                                 Do it once, when models and strategies are final. Every opening is logged.
  streamlit run dashboard.py     the dashboard

One run at a time: a second one stops right away (core/runs.py), two at once would break the files.
Always from the venv in this folder (source venv/bin/activate): any other Python is stopped (core/env.py).
"""
import argparse
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.env import require_venv

require_venv()  # only the venv in this folder, never Anaconda or another Python

import settings
from core import runs
from core.plugins import load_file, player_files, step_folders


def pick(what):
    steps = step_folders()
    if what in (None, "all"):
        return steps
    lo, _, hi = what.partition("-")
    lo, hi = int(lo), int(hi or lo)
    return [(n, p) for n, p in steps if lo <= n <= hi]


def show_list():
    for n, folder in step_folders():
        step = folder / "_step.py"
        name = load_file(step).STEP if step.exists() else "(no _step.py yet)"
        print(f"{n:02d} {name}  ({folder.name}/)")
        for f in player_files(folder):
            text = f.read_text()
            off = '"enabled": False' in text.split("SETTINGS", 1)[-1][:400]
            print(f"     {'off' if off else 'on ':<4}{f.stem}")


def run_tests():
    failed = 0
    for f in sorted((settings.ROOT / "tests").glob("test_*.py")):
        module = load_file(f)
        for name in [n for n in dir(module) if n.startswith("test_")]:
            t0 = time.time()
            try:
                getattr(module, name)()
                print(f"  ok    {f.stem}.{name} ({time.time() - t0:.1f}s)")
            except Exception:
                failed += 1
                print(f"  FAIL  {f.stem}.{name}")
                traceback.print_exc()
    print("RESULT: all tests passed" if not failed else f"RESULT: {failed} test(s) failed")
    return failed == 0


def log_run(rows):
    import pandas as pd

    p = settings.DATA / "runs" / "run_log.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(p, index=False)


def main():
    ap = argparse.ArgumentParser(description="paper fund: run the steps")
    ap.add_argument("what", nargs="?", default="all", help="all, a step number, a range like 2-4, list or test")
    ap.add_argument("--only", help="comma separated player names")
    ap.add_argument("--check", action="store_true", help="no-future check instead of a normal run")
    ap.add_argument("--fresh", action="store_true", help="ignore what's saved and redo it")
    ap.add_argument("--holdout", action="store_true", help="the final test: include the holdout period")
    ap.add_argument("--every", type=float, help="keep going: run again every N hours (stop with Ctrl+C)")
    args = ap.parse_args()
    args.only = [x.strip() for x in args.only.split(",")] if args.only else None

    if args.what == "list":
        return show_list()
    if args.what == "test":
        sys.exit(0 if run_tests() else 1)

    if args.holdout:
        log = settings.DATA / "holdout" / "opened.csv"
        before = len(open(log).readlines()) - 1 if log.exists() else 0
        print(f"THE HOLDOUT is the final test. It has been opened {before} time(s) before.")
        print("Every extra look after changing something makes it less honest. Type OPEN to continue:")
        if input("> ").strip() != "OPEN":
            return print("Not opened.")
        settings.SHOW_HOLDOUT = True
        log.parent.mkdir(parents=True, exist_ok=True)
        new = not log.exists()
        with open(log, "a") as f:
            if new:
                f.write("time,steps\n")
            f.write(f"{datetime.now().isoformat(timespec='seconds')},{args.what}\n")

    runs.take_lock(args.what + (f" every {args.every:g}h" if args.every else ""))
    try:
        while True:
            runs.set_state("running")
            rows = run_steps(args)
            log_run(rows)
            if not args.every or any(r["status"] == "stopped" for r in rows):
                break
            nxt = datetime.now() + timedelta(hours=args.every)
            runs.set_state("sleeping", next_run=nxt.isoformat(timespec="minutes"))
            print(f"\nNext run {nxt:%Y-%m-%d %H:%M}. Stop with Ctrl+C (or Stop on the dashboard).", flush=True)
            time.sleep(max(60.0, args.every * 3600))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        runs.release()


def run_steps(args):
    rows = []
    for n, folder in pick(args.what):
        step = folder / "_step.py"
        if not step.exists():
            continue
        module = load_file(step)
        print(f"\n===== {n:02d} {module.STEP} ({folder.name}) =====")
        t0, status, summary = time.time(), "ok", ""
        try:
            summary = module.run(args)
        except KeyboardInterrupt:
            status = "stopped"
            print("\nStopped. Run it again to continue where it was.")
        except Exception as e:
            status = f"error: {e}"
            traceback.print_exc()
        rows.append({"time": datetime.now().isoformat(timespec="seconds"), "step": folder.name, "status": status,
                     "seconds": round(time.time() - t0, 1), "summary": str(summary)[:300]})
        if status != "ok":
            break
    return rows


if __name__ == "__main__":
    main()