# paper-fund-v2/tests/test_core.py
"""Tests for the fixed building blocks in core/. Run with: python main.py test"""
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import settings
from core import metrics
from core.contracts import Model
from core.plugins import load_file, load_players, player_files
from core.timeline import FutureDataError, self_test
from core.wallet import LedgerError, Wallet

synthetic = load_file(Path(__file__).with_name("synthetic.py"))


def test_timeline_hides_the_future():
    tl = synthetic.timeline()
    assert all(self_test(tl, a) for a in tl.assets)
    v = tl.view("2010-06-01")
    try:
        v.close_on("asset0", "2010-06-02")
        raise AssertionError("asking for tomorrow should fail")
    except FutureDataError:
        pass


def test_wallet_is_append_only_and_tamper_proof():
    with tempfile.TemporaryDirectory() as t:
        db = Path(t) / "w.db"
        w = Wallet.create(1000, db)
        tid = w.open_trade("gold", 200, 100, 0.0025)
        w.close_trade(tid, 110, 0.0025)
        assert float(w.cash()) > 1000
        con = sqlite3.connect(db)
        try:
            con.execute("UPDATE entries SET cash_cents = 999999 WHERE seq = 1")
            con.commit()
            raise AssertionError("editing should be blocked")
        except sqlite3.DatabaseError:
            pass
        con.execute("DROP TRIGGER entries_no_update")
        con.execute("UPDATE entries SET cash_cents = 999999 WHERE seq = 1")
        con.commit()
        con.close()
        try:
            Wallet(db)
            raise AssertionError("a changed wallet should be refused")
        except LedgerError:
            pass


def test_every_model_file_loads():
    folder = settings.ROOT / "04_models"
    models = load_players(folder, Model, settings.HORIZON_DAYS, include_disabled=True)
    assert len(models) == len(player_files(folder)), "every file in 04_models needs exactly one Model class"
    for name, m in models.items():
        assert m.about, f"{name}: write a short `about`"


def test_scores_reward_skill_not_luck():
    rng = np.random.default_rng(0)
    n = 4000
    actual = rng.normal(0, 0.04, n)
    rows = pd.DataFrame({"date": pd.bdate_range("2005-01-01", periods=n), "asset": "a",
                         "actual": actual, "p10": -0.05, "p90": 0.05})
    good = rows.assign(run="good", p50=actual + rng.normal(0, 0.04, n))
    noise = rows.assign(run="noise", p50=rng.normal(0, 0.04, n))
    s_good, s_noise = metrics.score(good, 20), metrics.score(noise, 20)
    assert s_good["ic"] > 0.5 and s_good["trust"] > 2
    assert abs(s_noise["ic"]) < 0.1
    luck = metrics.luck_test(pd.concat([good, noise]), n=60)
    assert luck.loc["good", "luck_chance"] < 0.05 and luck.loc["noise", "luck_chance"] > 0.05