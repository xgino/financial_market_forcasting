# financial_market_forcasting1/core/plugins.py
"""
Loads a step folder. In every step folder:
  files starting with _  run the folder (_step.py is the step itself)
  every other .py file   is a player: a data source, a model, a strategy, ...
Drop a new .py file in a folder and the next run picks it up. No registering anything.
"""
import importlib.util
import re
from pathlib import Path

import settings

ROOT = settings.ROOT


def step_folders():
    """01_data, 02_features, ... sorted by their number."""
    out = []
    for p in ROOT.iterdir():
        m = re.match(r"^(\d\d)_", p.name)
        if p.is_dir() and m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def load_file(path):
    path = Path(path)
    name = "step_" + re.sub(r"\W", "_", f"{path.parent.name}_{path.stem}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def player_files(folder):
    return sorted(p for p in Path(folder).glob("*.py") if not p.name.startswith("_"))


def load_players(folder, base_class, *args, only=None, include_disabled=False):
    """{file name: instance} for every enabled player in the folder that has a class based on base_class."""
    out = {}
    for f in player_files(folder):
        if only and f.stem not in only:
            continue
        module = load_file(f)
        cfg = dict(getattr(module, "SETTINGS", {}))
        if not cfg.get("enabled", True) and not include_disabled:
            continue
        classes = [obj for obj in vars(module).values() if isinstance(obj, type)
                   and issubclass(obj, base_class) and obj is not base_class and obj.__module__ == module.__name__]
        if len(classes) != 1:
            raise ValueError(f"{f.name} must define exactly one class based on {base_class.__name__}, found {len(classes)}")
        player = classes[0](*args, cfg)
        player.run_name = f.stem
        player.file = f
        out[f.stem] = player
    return out