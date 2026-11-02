# financial_market_forcasting1/core/runs.py
"""
One run at a time, and runs that keep going in the background.

  data/runs/running.json   the run that's going: pid, steps, since, and what it's doing (a lock)
  data/runs/live.log       everything the newest background run printed

Two runs at once write the same files at the same time and break them, so main.py takes the lock first
and a second run stops right away. A lock left by a crashed run (its process is gone) doesn't count.

The dashboard starts runs with start(): main.py in its own process, so closing or clicking around the
dashboard never stops it, and stop() ends it the same way Ctrl+C does (it saves what it has).
"""
import json
import os
import signal
import subprocess
import sys
from datetime import datetime

import settings

LOCK = settings.DATA / "runs" / "running.json"
LOG = settings.DATA / "runs" / "live.log"


def running():
    """The run that's going right now ({pid, what, started, state}), or None."""
    try:
        info = json.loads(LOCK.read_text())
        os.kill(int(info["pid"]), 0)
        return info
    except (OSError, ValueError, KeyError):
        return None


def take_lock(what):
    other = running()
    if other:
        sys.exit(f"Another run is going (pid {other['pid']}, steps {other['what']}, since {other['started']}). "
                 f"Wait for it, or stop it with: kill -INT {other['pid']}")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    set_state("running", what=what, started=datetime.now().isoformat(timespec="seconds"), pid=os.getpid())


def set_state(state, **extra):
    """What the run is doing now, for the dashboard (running, sleeping until ...)."""
    try:
        info = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    except ValueError:
        info = {}
    LOCK.write_text(json.dumps({**info, **extra, "state": state}))


def release():
    try:
        if json.loads(LOCK.read_text()).get("pid") == os.getpid():
            LOCK.unlink()
    except (OSError, ValueError):
        pass


def start(args):
    """Start python main.py <args> in the background. -> its pid (None if a run is going already)."""
    if running():
        return None
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w") as out:
        proc = subprocess.Popen([sys.executable, "-u", "main.py", *args], cwd=settings.ROOT, stdout=out,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    return proc.pid


def stop():
    """Like Ctrl+C: the step stops, saves what it has, and the next run continues from there."""
    info = running()
    if info:
        os.kill(int(info["pid"]), signal.SIGINT)
    return info


def tail(n=30):
    """The last n lines the background run printed (progress bars: only their newest state)."""
    if not LOG.exists():
        return []
    text = LOG.read_text(errors="replace")[-200_000:]
    lines = [ln.split("\r")[-1] for ln in text.split("\n")]
    return [ln for ln in lines if ln.strip()][-n:]
