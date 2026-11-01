# financial_market_forcasting1/core/env.py
"""
Only the project's own venv (venv/ next to main.py) may run this project. Another Python (Anaconda, the
system one) has other package versions and gives errors that are hard to trace, so it is stopped with a
message saying how to start it the right way.

  python main.py / streamlit run dashboard.py   after: source venv/bin/activate
  venv/bin/python main.py / venv/bin/streamlit run dashboard.py   without activating
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "venv"


def wrong_python():
    """None when this is the project's venv, else a message with how to start it."""
    if Path(sys.prefix).resolve() == VENV.resolve():
        return None
    return (f"This is not the project's venv.\n  now:  {sys.executable}\n  want: {VENV / 'bin' / 'python'}\n"
            f"Start it with:  source venv/bin/activate   (then python main.py / streamlit run dashboard.py)\n"
            f"           or:  venv/bin/python main.py   /   venv/bin/streamlit run dashboard.py")


def require_venv():
    msg = wrong_python()
    if msg:
        sys.exit(msg)
