# financial_market_forcasting1/03_training/chronos_finetune.py
"""
Fine-tunes Chronos-Bolt small on your own assets, walk-forward: one checkpoint per year (cutoff = 1 January),
each trained ONLY on price windows whose 20 days had ended before its cutoff. 04_models/chronos_small_ft.py
uses the newest checkpoint from before the day it forecasts and battles the zero-shot chronos_small.

Cutoffs start at the base model's release (its seen_until): before that its forecasts can't earn trust
anyway (it may have seen those prices), so earlier checkpoints would only cost time.

Per checkpoint:
  windows     random 512 day contexts + the next 20 days, from every training asset (settings "train": True),
              target ending before the cutoff; the last `holdout_days` before the cutoff are held back
  training    `steps` steps of the model's own quantile loss (AdamW, small learning rate), Mac GPU
  check       loss on the held-back windows, zero-shot vs fine-tuned (data/training/chronos_finetune/log.csv,
              dashboard page Training): if fine-tuning doesn't lower it, the index says so
  saved       data/checkpoints/chronos_finetune/<cutoff>/ (the weights, ~190 MB, not in git) + index.csv

Run: python main.py 3 --only chronos_finetune   (a few minutes per checkpoint)
"""
SETTINGS = {
    "enabled": True,
    "base": "amazon/chronos-bolt-small",
    "first_cutoff": "2025-01-01",   # = the base model's seen_until + 1 day
    "context_days": 512,
    "steps": 400,
    "batch_size": 32,
    "learning_rate": 1e-5,
    "holdout_days": 120,            # last ~6 months before a cutoff: not trained on, used for the check
    "check_windows": 256,
    "seed": 0,
}

import shutil

import numpy as np
import pandas as pd

import settings
from core import store
from core.contracts import Trainer, free_memory, torch_device
from core.training import Logger


class ChronosFinetune(Trainer):
    about = "Chronos-Bolt small fine-tuned on your assets, one checkpoint per year (walk-forward)"

    def cutoffs(self, tl):
        last = tl.calendar()[-1]
        return [d for d in pd.date_range(self.s["first_cutoff"], last, freq="YS")]

    def train(self, tl, fresh=False, progress=True):
        s = self.s
        folder = settings.DATA / "checkpoints" / self.run_name
        idx_path = folder / "index.csv"
        key = store.fingerprint({k: v for k, v in s.items() if k != "enabled"}, sorted(tl.assets))
        index = pd.read_csv(idx_path) if idx_path.exists() and not fresh else pd.DataFrame()
        if len(index) and (index["settings"] != key).any():
            print("  settings or assets changed: fine-tuning every checkpoint again")
            index = pd.DataFrame()
        done = set(index["cutoff"]) if len(index) else set()
        todo = [c for c in self.cutoffs(tl) if str(c.date()) not in done]
        if not todo:
            return f"{len(index)} checkpoints, nothing new"
        log = Logger(self.run_name, fresh)
        rows = index.to_dict("records") if len(index) else []
        for cutoff in todo:
            log.cutoff = str(cutoff.date())
            r = self._one(tl, cutoff, folder / str(cutoff.date()), log, progress)
            rows.append({"cutoff": str(cutoff.date()), **r, "settings": key})
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(idx_path, index=False)
        free_memory()
        last = rows[-1]
        return (f"{len(rows)} checkpoints; newest {last['cutoff']}: held-back loss zero-shot {last['loss_zero_shot']:.4f}, "
                f"fine-tuned {last['loss_finetuned']:.4f} ({'better' if last['helped'] else 'NOT better'})")

    def _windows(self, tl, cutoff):
        """(train, check) arrays of shape (n, context + 20): the target always ends before the cutoff."""
        c, h = self.s["context_days"], settings.HORIZON_DAYS
        v = tl.view(cutoff - pd.Timedelta(days=1))
        train, check = [], []
        split = cutoff - pd.Timedelta(days=int(self.s["holdout_days"] * 7 / 5))
        for a in v.assets:
            closes = v.closes(a)
            x = closes.to_numpy(np.float32)
            if len(x) < c + h + 10:
                continue
            ends = closes.index[c + h - 1:]                       # the last day of each window's target
            starts = np.arange(len(ends))
            ok_train = ends < split - pd.Timedelta(days=int(h * 7 / 5))   # a gap so train and check don't overlap
            ok_check = ends >= split
            for want, out in ((ok_train, train), (ok_check, check)):
                for i in starts[want]:
                    out.append((x, i))
        return train, check

    def _batch(self, windows, idx):
        import torch

        c, h = self.s["context_days"], settings.HORIZON_DAYS
        w = np.stack([windows[i][0][windows[i][1]:windows[i][1] + c + h] for i in idx])
        return torch.tensor(w[:, :c]), torch.tensor(w[:, c:])

    def _loss(self, model, windows, idx, device):
        import torch

        losses = []
        with torch.no_grad():
            for s in range(0, len(idx), 64):
                ctx, tgt = self._batch(windows, idx[s:s + 64])
                losses.append(float(model(context=ctx.to(device), target=tgt.to(device)).loss) * len(ctx))
        return sum(losses) / len(idx)

    def _one(self, tl, cutoff, out_dir, log, progress):
        import torch
        from chronos import BaseChronosPipeline

        s = self.s
        rng = np.random.default_rng(s["seed"])
        torch.manual_seed(s["seed"])
        device = torch_device(s.get("device", settings.DEVICE))
        train, check = self._windows(tl, cutoff)
        if len(train) < 1000 or len(check) < 50:
            raise RuntimeError(f"{cutoff.date()}: too few windows ({len(train)} train, {len(check)} check)")
        check_idx = rng.choice(len(check), min(s["check_windows"], len(check)), replace=False)
        pipe = BaseChronosPipeline.from_pretrained(s["base"], device_map=device, torch_dtype=torch.float32)
        model = pipe.model
        before = self._loss(model.eval(), check, check_idx, device)
        log(cv_loss=before, rows=len(train), best=False, model="zero-shot")
        opt = torch.optim.AdamW(model.parameters(), lr=s["learning_rate"], weight_decay=0.01)
        model.train()
        from tqdm import tqdm

        bar = tqdm(range(s["steps"]), desc=f"  fine-tune {cutoff.date()}", unit="step", disable=not progress, mininterval=1.0)
        for _ in bar:
            ctx, tgt = self._batch(train, rng.integers(0, len(train), s["batch_size"]))
            loss = model(context=ctx.to(device), target=tgt.to(device)).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bar.set_postfix(loss=f"{float(loss):.4f}")
        after = self._loss(model.eval(), check, check_idx, device)
        log(cv_loss=after, rows=len(train), best=after < before, model="fine-tuned")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        model.save_pretrained(out_dir)
        del pipe, model, opt
        free_memory()
        return {"train_windows": len(train), "check_windows": len(check_idx), "loss_zero_shot": round(before, 5),
                "loss_finetuned": round(after, 5), "helped": bool(after < before)}
