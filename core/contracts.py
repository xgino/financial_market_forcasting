# paper-fund-v2/core/contracts.py
"""
What a model looks like. Every file in 04_models/ has SETTINGS at the top and one class based on Model.

  predict(view, asset)        -> [p10, p50, p90] expected return over the horizon (0.02 = +2%)
                                 p10 = bad case, p50 = middle guess, p90 = good case.
                                 The price it expects = today's price x (1 + p50).
  predict_batch(views, asset) -> one row per view (default: predict for each, models can do it faster)

A view (core/timeline.py) only shows data up to its date, so a model can't see the future.

SETTINGS keys every model may use:
  enabled     False = the step skips this file
  seen_until  pretrained models: their release date. Forecasts up to then never count for trust
  device      auto / mps / cpu (doesn't change results)
  first_day   only forecast from this date on (slow models: skip the years that can't earn trust anyway)
Everything else in SETTINGS is part of the cache key: change it and the forecasts are made again.
"""
import numpy as np

# 1.28: fixed number from the bell curve. 10% of a bell curve lies below -1.28 standard deviations and
# 10% above +1.28. So middle -1.28 x swing size = bad case (p10), middle +1.28 x swing size = good case.
Z90 = 1.2815515655446004
NAN3 = np.array([np.nan, np.nan, np.nan])
NOT_IN_CACHE_KEY = {"enabled", "seen_until", "device", "batch_size", "about"}


class Model:
    kind = "rule"          # rule | stats | ml | pretrained
    about = ""             # one line in plain words
    min_history = 260      # days of data needed before the first forecast

    def __init__(self, horizon, settings):
        self.horizon = horizon
        self.s = dict(settings)
        self.run_name = type(self).__name__.lower()
        self.seen_until = self.s.get("seen_until")
        self.min_history = self.s.get("min_history", self.min_history)

    def cache_settings(self):
        """What the forecasts depend on. Change any of it and the cached forecasts are redone."""
        return {k: v for k, v in self.s.items() if k not in NOT_IN_CACHE_KEY}

    def predict(self, view, asset):
        raise NotImplementedError

    def predict_batch(self, views, asset):
        return np.array([self.predict(v, asset) for v in views], dtype=float).reshape(len(views), 3)

    def explain(self, view, asset):
        """Plain sentences: why this forecast. Models override it with their own reasons."""
        return [self.about]


def bell_range(mu, sd):
    """mu, sd = expected log return and its swing size over the whole horizon -> [p10, p50, p90]."""
    return np.exp([mu - Z90 * sd, mu, mu + Z90 * sd]) - 1


def log_returns(view, asset, days):
    c = view.closes(asset, days + 1).to_numpy()
    return np.diff(np.log(c)) if len(c) > 1 else np.array([])


def torch_device(want="auto"):
    """auto = the Mac GPU (mps) if there is one, else an NVIDIA GPU (cuda), else the CPU."""
    import torch

    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if want == "auto":
        return "mps" if mps else "cuda" if torch.cuda.is_available() else "cpu"
    if want == "mps" and not mps:
        print("  Mac GPU (mps) not available, using the CPU")
        return "cpu"
    return want


# ---------- the other kinds of players ----------

class Player:
    """Base for every non-model player. SETTINGS from the top of its file arrive as self.s."""
    about = ""

    def __init__(self, cfg):
        self.s = dict(cfg)
        self.run_name = type(self).__name__.lower()


class DataSource(Player):
    """01_data: downloads and cleans data, writes CSV. run() returns a short report (dict)."""

    def run(self):
        raise NotImplementedError


class FeatureMaker(Player):
    """02_features: turns prices into features, writes CSV. run(timeline) returns a short report."""

    def run(self, tl):
        raise NotImplementedError


class Trainer(Player):
    """03_training: trains one of your own models, walk-forward, and saves checkpoints."""

    def train(self, tl, progress=True):
        raise NotImplementedError


class Check(Player):
    """07_judge: one area of the report. run() returns a list of (level, text), level: good / warn / bad / info."""
    area = ""

    def run(self):
        raise NotImplementedError


class Notifier(Player):
    """08_notify: sends the monthly news somewhere (email, ...). run(args) returns a short report."""

    def run(self, args):
        raise NotImplementedError