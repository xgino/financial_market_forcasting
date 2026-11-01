# paper-fund-v2/core/pretrained.py
"""
Shared code for pretrained models, so each model file in 04_models/ stays a few lines of SETTINGS.

  Chronos   Amazon Chronos-Bolt and Chronos-2   pip install chronos-forecasting
  TimesFM   Google TimesFM 2.5                  pip install "timesfm[torch]"
  TiRex     NX-AI TiRex (xLSTM)                 pip install tirex-ts
  Kronos    Kronos, trained on market candles   code copied into core/kronos (not on pip)
  Moirai2   Salesforce Moirai 2                 pip install --no-deps uni2ts, plus lightning gluonts jaxtyping
  TimeMoE   Time-MoE (mixture of experts)       code copied into core/timemoe (instead of trust_remote_code)

All run on the Mac GPU automatically (settings.DEVICE). Pretrained models can have seen old prices
while they were trained, so their files set seen_until (release date): those forecasts never earn trust.
"""
import numpy as np
import pandas as pd

import settings
from core.contracts import Model, bell_range, torch_device


class Chronos(Model):
    kind = "pretrained"
    about = "Pretrained Amazon transformer, zero shot. Continues the shape of the last 512 prices."

    def __init__(self, horizon, cfg):
        super().__init__(horizon, cfg)
        self.checkpoint = self.s["checkpoint"]
        self.context_days = self.s.get("context_days", 512)
        self.noise = self.s.get("noise", 0.0)
        self.batch_size = self.s.get("batch_size")
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from chronos import BaseChronosPipeline

            device = torch_device(self.s.get("device", settings.DEVICE))
            if self.batch_size is None:
                self.batch_size = 64 if device == "cpu" else 256
            print(f"  loading {self.checkpoint} on {device} (first time downloads it)...")
            self._torch = torch
            self._pipe = BaseChronosPipeline.from_pretrained(self.checkpoint, device_map=device,
                                                             torch_dtype=torch.float32)
        return self._pipe

    def _add_noise(self, w, date):
        rng = np.random.default_rng(date.toordinal())  # same date = same noise
        out = w * np.exp(rng.normal(0.0, self.noise, len(w)))
        out[-1] = w[-1]
        return out

    def predict(self, view, asset):
        return self.predict_batch([view], asset)[0]

    def predict_batch(self, views, asset):
        windows = [v.closes(asset, self.context_days).to_numpy() for v in views]
        if self.noise:
            windows = [self._add_noise(w, v.date) for w, v in zip(windows, views)]
        out = np.full((len(views), 3), np.nan)
        ok = [i for i, w in enumerate(windows) if len(w) >= 30]
        if not ok:
            return out
        pipe = self._load()
        for s in range(0, len(ok), self.batch_size):
            idx = ok[s:s + self.batch_size]
            ctx = [self._torch.tensor(windows[i].astype(np.float32)) for i in idx]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=self.horizon, quantile_levels=[0.1, 0.5, 0.9])
            if isinstance(q, (list, tuple)):   # chronos-2: one tensor per series, shape (1, horizon, 3)
                end = np.stack([x[0, -1, :].float().cpu().numpy() for x in q])
            else:                              # chronos-bolt: shape (batch, horizon, 3)
                end = q[:, -1, :].float().cpu().numpy()
            last = np.array([windows[i][-1] for i in idx], dtype=float)
            out[idx] = end.astype(float) / last[:, None] - 1
        return out

    def explain(self, view, asset):
        return ["A pretrained neural network that continues the shape of the last 512 prices.",
                "It can't explain its own reasoning, so judge it only by its track record, and only on years after "
                f"its release ({self.seen_until}): before that it may have seen these prices while it was trained."]


class TimesFM(Model):
    kind = "pretrained"
    about = "Pretrained Google transformer (TimesFM 2.5), zero shot. A second opinion next to Chronos."

    def __init__(self, horizon, cfg):
        super().__init__(horizon, cfg)
        self.checkpoint = self.s.get("checkpoint", "google/timesfm-2.5-200m-pytorch")
        self.context_days = self.s.get("context_days", 512)
        self.batch_size = self.s.get("batch_size")
        self._m = None

    def _load(self):
        if self._m is None:
            import timesfm
            import torch

            device = torch_device(self.s.get("device", settings.DEVICE))
            if self.batch_size is None:
                self.batch_size = 32 if device == "cpu" else 128
            print(f"  loading {self.checkpoint} on {device} (first time downloads it)...")
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint, torch_compile=False)
            if str(m.model.device) != device:
                m.model.device = torch.device(device)  # TimesFM only picks cuda or cpu by itself
                m.model.to(m.model.device)
            m.compile(timesfm.ForecastConfig(
                max_context=self.context_days, max_horizon=128, normalize_inputs=True,
                use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
                fix_quantile_crossing=True, per_core_batch_size=self.batch_size))
            self._m = m
        return self._m

    def predict(self, view, asset):
        return self.predict_batch([view], asset)[0]

    def predict_batch(self, views, asset):
        windows = [v.closes(asset, self.context_days).to_numpy() for v in views]
        out = np.full((len(views), 3), np.nan)
        ok = [i for i, w in enumerate(windows) if len(w) >= 30]
        if not ok:
            return out
        m = self._load()
        inputs = [windows[i].astype(np.float32) for i in ok]
        # TimesFM fills an incomplete batch with float64 dummies, which the Mac GPU can't handle,
        # so fill it up ourselves with float32 copies and drop their results
        inputs += [inputs[0]] * ((-len(inputs)) % m.global_batch_size)
        _, q = m.forecast(horizon=self.horizon, inputs=inputs)
        end = np.asarray(q)[:len(ok), -1, [1, 5, 9]].astype(float)   # 10% / 50% / 90% at the last day
        last = np.array([windows[i][-1] for i in ok], dtype=float)
        out[ok] = end / last[:, None] - 1
        return out

    def explain(self, view, asset):
        return ["A pretrained neural network that continues the shape of the last 512 prices.",
                "It can't explain its own reasoning, so judge it only by its track record, and only on years after "
                f"its release ({self.seen_until}): before that it may have seen these prices while it was trained."]

class TiRex(Model):
    kind = "pretrained"
    about = "Pretrained NX-AI xLSTM (TiRex), zero shot. A different kind of network than the transformers."

    def __init__(self, horizon, cfg):
        super().__init__(horizon, cfg)
        self.checkpoint = self.s.get("checkpoint", "NX-AI/TiRex")
        self.context_days = self.s.get("context_days", 512)
        self.batch_size = self.s.get("batch_size")
        self._m = None

    def _load(self):
        if self._m is None:
            from tirex import load_model

            device = torch_device(self.s.get("device", settings.DEVICE))
            if self.batch_size is None:
                self.batch_size = 64 if device == "cpu" else 256
            print(f"  loading {self.checkpoint} on {device} (first time downloads it)...")
            self._m = load_model(self.checkpoint, device=device, backend="torch")
        return self._m

    def predict(self, view, asset):
        return self.predict_batch([view], asset)[0]

    def predict_batch(self, views, asset):
        import torch

        windows = [v.closes(asset, self.context_days).to_numpy() for v in views]
        out = np.full((len(views), 3), np.nan)
        ok = [i for i, w in enumerate(windows) if len(w) >= 30]
        if not ok:
            return out
        m = self._load()
        n = max(len(windows[i]) for i in ok)   # pad short windows in front with NaN (TiRex skips NaN)
        ctx = np.full((len(ok), n), np.nan, dtype=np.float32)
        for r, i in enumerate(ok):
            ctx[r, n - len(windows[i]):] = windows[i]
        # the Mac GPU rounds slightly differently for other batch sizes, so always send full batches
        # (filled up with copies, their results dropped): then a forecast doesn't depend on its neighbours
        ctx = np.concatenate([ctx, np.repeat(ctx[:1], (-len(ctx)) % self.batch_size, axis=0)])
        q, _ = m.forecast(context=torch.tensor(ctx), prediction_length=self.horizon,
                          output_type="numpy", batch_size=self.batch_size)
        end = np.asarray(q)[:len(ok), -1, [0, 4, 8]].astype(float)   # quantiles 0.1 ... 0.9: 10% / 50% / 90%
        last = np.array([windows[i][-1] for i in ok], dtype=float)
        out[ok] = end / last[:, None] - 1
        return out

    def explain(self, view, asset):
        return [f"A pretrained neural network that continues the shape of the last {self.context_days} prices.",
                "It can't explain its own reasoning, so judge it only by its track record, and only on years after "
                f"its release ({self.seen_until}): before that it may have seen these prices while it was trained."]


class Kronos(Model):
    kind = "pretrained"
    about = "Pretrained on candles from 45 stock exchanges (Kronos), zero shot. Draws possible futures and counts them."

    def __init__(self, horizon, cfg):
        super().__init__(horizon, cfg)
        self.checkpoint = self.s["checkpoint"]
        self.tokenizer = self.s.get("tokenizer", "NeoQuasar/Kronos-Tokenizer-base")
        self.context_days = self.s.get("context_days", 256)
        self.samples = self.s.get("samples", 20)
        self._p = None

    def _load(self):
        if self._p is None:
            from core.kronos import Kronos as Net, KronosPredictor, KronosTokenizer

            device = torch_device(self.s.get("device", settings.DEVICE))
            print(f"  loading {self.checkpoint} on {device} (first time downloads it)...")
            self._p = KronosPredictor(Net.from_pretrained(self.checkpoint),
                                      KronosTokenizer.from_pretrained(self.tokenizer),
                                      device=device, max_context=self.context_days)
        return self._p

    @staticmethod
    def _stamps(dates):
        return np.stack([dates.minute, dates.hour, dates.weekday, dates.day, dates.month], 1).astype(np.float32)

    def _window(self, view, asset):
        """Kronos wants open, high, low, close, volume, amount. We only have open and close, so high/low are
        the bigger/smaller of the two and volume is 0 (Kronos does the same when volume is missing)."""
        b = view.bars(asset, self.context_days)
        o, c = b["open_eur"].to_numpy(), b["close_eur"].to_numpy()
        x = np.stack([o, np.maximum(o, c), np.minimum(o, c), c, np.zeros_like(c), np.zeros_like(c)], 1)
        mu, sd = x.mean(0), x.std(0)
        future = pd.bdate_range(b.index[-1] + pd.Timedelta(days=1), periods=self.horizon)
        x = np.clip((x - mu) / (sd + 1e-5), -5, 5).astype(np.float32)
        return x, self._stamps(b.index), self._stamps(future), mu[3], sd[3], c[-1]

    def predict(self, view, asset):
        return self.predict_batch([view], asset)[0]

    def predict_batch(self, views, asset):
        import torch

        out = np.full((len(views), 3), np.nan)
        p = None
        for i, v in enumerate(views):
            if v.n_bars(asset) < self.context_days:
                continue
            p = p or self._load()
            x, xs, ys, mu, sd, last = self._window(v, asset)
            # one day at a time with its own seed: the draws (and so the forecast) only depend on that day
            torch.manual_seed(v.date.toordinal())
            k = self.samples
            pr = p.generate(np.repeat(x[None], k, 0), np.repeat(xs[None], k, 0), np.repeat(ys[None], k, 0),
                            self.horizon, 1.0, 0, 0.9, 1, False)
            end = pr[:, -1, 3] * (sd + 1e-5) + mu   # the close on the last day of each drawn future
            out[i] = np.quantile(end, [0.1, 0.5, 0.9]) / last - 1
        return out

    def explain(self, view, asset):
        return [f"A pretrained neural network for market candles. It draws {self.samples} possible next months from "
                f"the last {self.context_days} days and the range is where most of them end.",
                "It can't explain its own reasoning, so judge it only by its track record, and only after "
                f"its release ({self.seen_until})."]


class Moirai2(Model):
    kind = "pretrained"
    about = "Pretrained Salesforce transformer (Moirai 2), zero shot. Tiny and fast, runs on the CPU."

    def __init__(self, horizon, cfg):
        super().__init__(horizon, cfg)
        self.checkpoint = self.s.get("checkpoint", "Salesforce/moirai-2.0-R-small")
        self.context_days = self.s.get("context_days", 512)
        self.batch_size = self.s.get("batch_size", 256)
        self._f = None

    def _load(self):
        if self._f is None:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

            # always the CPU: its scaler counts in float64, which the Mac GPU can't do
            print(f"  loading {self.checkpoint} on cpu (first time downloads it)...")
            self._f = Moirai2Forecast(module=Moirai2Module.from_pretrained(self.checkpoint),
                                      prediction_length=self.horizon, context_length=self.context_days,
                                      target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        return self._f

    def predict(self, view, asset):
        return self.predict_batch([view], asset)[0]

    def predict_batch(self, views, asset):
        windows = [v.closes(asset, self.context_days).to_numpy() for v in views]
        out = np.full((len(views), 3), np.nan)
        ok = [i for i, w in enumerate(windows) if len(w) >= 30]
        if not ok:
            return out
        f = self._load()
        for s in range(0, len(ok), self.batch_size):
            idx = ok[s:s + self.batch_size]
            q = f.predict([windows[i].astype(np.float32) for i in idx])   # (batch, 9 quantiles, horizon)
            end = q[:, [0, 4, 8], -1].astype(float)                       # 10% / 50% / 90% at the last day
            last = np.array([windows[i][-1] for i in idx], dtype=float)
            out[idx] = end / last[:, None] - 1
        return out

    def explain(self, view, asset):
        return [f"A pretrained neural network that continues the shape of the last {self.context_days} prices.",
                "It can't explain its own reasoning, so judge it only by its track record, and only on years after "
                f"its release ({self.seen_until}): before that it may have seen these prices while it was trained."]


class TimeMoE(Model):
    kind = "pretrained"
    about = "Pretrained mixture of experts (Time-MoE), zero shot. Gives one middle guess; the range comes from recent swings."

    def __init__(self, horizon, cfg):
        super().__init__(horizon, cfg)
        self.checkpoint = self.s["checkpoint"]
        self.revision = self.s.get("revision")
        self.context_days = self.s.get("context_days", 512)
        self.swing_days = self.s.get("swing_days", 60)
        self.batch_size = self.s.get("batch_size", 64)
        self._m = None

    def _load(self):
        if self._m is None:
            import torch
            from core.timemoe.modeling_time_moe import TimeMoeForPrediction, TimeMoeRotaryEmbedding

            device = torch_device(self.s.get("device", settings.DEVICE))
            print(f"  loading {self.checkpoint} on {device} (first time downloads it)...")
            m = TimeMoeForPrediction.from_pretrained(self.checkpoint, revision=self.revision, dtype=torch.float32)
            for mod in m.modules():
                if isinstance(mod, TimeMoeRotaryEmbedding):
                    mod.rebuild()
            self._torch, self._device = torch, device
            self._m = m.to(device).eval()
        return self._m

    def predict(self, view, asset):
        return self.predict_batch([view], asset)[0]

    def predict_batch(self, views, asset):
        windows = [v.closes(asset, self.context_days).to_numpy() for v in views]
        out = np.full((len(views), 3), np.nan)
        ok = [i for i, w in enumerate(windows) if len(w) >= self.context_days]   # one batch needs equal lengths
        if not ok:
            return out
        m, torch = self._load(), self._torch
        x = np.stack([windows[i] for i in ok]).astype(np.float32)
        mu, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True) + 1e-8
        x = (x - mu) / sd
        mids = []
        for s in range(0, len(x), self.batch_size):
            with torch.no_grad():   # the 32 day head in one pass, first 20 days kept (no step by step generate)
                o = m(input_ids=torch.tensor(x[s:s + self.batch_size], device=self._device),
                      max_horizon_length=32, use_cache=False, return_dict=True)
            mids.append(o.logits[:, -1, self.horizon - 1].float().cpu().numpy())
        mid = np.concatenate(mids) * sd[:, 0] + mu[:, 0]
        for r, i in enumerate(ok):
            w = windows[i]
            swing = np.diff(np.log(w[-self.swing_days - 1:])).std(ddof=1) * np.sqrt(self.horizon)
            out[i] = bell_range(np.log(mid[r] / w[-1]), swing) if mid[r] > 0 else np.nan
        return out

    def explain(self, view, asset):
        return [f"A pretrained neural network that continues the shape of the last {self.context_days} prices. "
                f"It only gives a middle guess; the bad..good range is the last {self.swing_days} days' swings.",
                "It can't explain its own reasoning, so judge it only by its track record, and only on years after "
                f"its release ({self.seen_until}): before that it may have seen these prices while it was trained."]
