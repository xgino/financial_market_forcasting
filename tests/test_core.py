# financial_market_forcasting1/tests/test_core.py
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

def test_warning_record_rewards_picking_the_bad_days():
    """A model that says 'fall' before real falls gets a high warn trust; one that says 'fall' at random, or
    on almost every day, gets about the 'any day' rate: no edge."""
    rng = np.random.default_rng(1)
    n = 4000
    actual = rng.normal(0.004, 0.04, n)
    rows = pd.DataFrame({"date": pd.bdate_range("2005-01-01", periods=n), "asset": "a",
                         "actual": actual, "p10": -0.05, "p90": 0.05})
    sharp = rows.assign(run="sharp", p50=np.where(actual + rng.normal(0, 0.03, n) < -0.03, -0.02, 0.0))
    random_ = rows.assign(run="random", p50=np.where(rng.random(n) < 0.2, -0.02, 0.0))
    always = rows.assign(run="always", p50=-0.02)
    s, r, a = (metrics.score(x, 20) for x in (sharp, random_, always))
    assert s["warn_hit"] > s["warn_base"] + 0.2 and s["warn_trust"] > 2 and s["warn_move"] < 0
    assert abs(r["warn_edge"]) < 0.05 and r["warn_trust"] < 2
    assert abs(a["warn_edge"]) < 1e-9, "warning every day is just the base rate"
    luck = metrics.call_luck_test(pd.concat([sharp, random_]), "warn", n=60)
    assert luck.loc["sharp", "luck_chance"] < 0.05 and luck.loc["random", "luck_chance"] > 0.05


def test_wait_needs_a_proven_warner():
    """WAIT only when the fall votes come from models with a good warning record (weighted by it)."""
    from core.verdict import verdict

    def table(warn_trust):
        return pd.DataFrame({"p10": -0.08, "p50": [-0.03, -0.03, 0.01], "p90": 0.04, "seen": False,
                             "trust": 0.5, "trust_all": 0.5, "periods": 50, "periods_all": 50, "trusted": False,
                             "warn_trust": warn_trust, "warn_trust_all": warn_trust, "warn_periods": 50,
                             "warn_periods_all": 50, "warn_hit_all": 0.6, "warn_base_all": 0.45,
                             "warner": [t >= settings.MIN_TRUST for t in warn_trust]}, index=["m1", "m2", "m3"])
    assert verdict(table([3.0, 0.0, 0.0]))["call"] == "WAIT"
    assert verdict(table([0.0, 0.0, 0.0]))["call"] == "KEEP YOUR PLAN", "no proven warner, no WAIT"
    assert verdict(table([0.0, 0.0, 6.0]))["call"] != "WAIT", "the best warner doesn't see a fall"


def test_pooling_weighs_own_asset_first():
    """Gold's record: gold counts fully, silver (same group) half, bitcoin (other group) not at all."""
    from core.pooling import pooled, weights

    w = weights("gold", ["gold", "silver", "bitcoin"], "group")
    assert w == {"gold": 1.0, "silver": 0.5, "bitcoin": 0.0}
    rows = pd.DataFrame({"model": "m", "situation": "all", "scope": ["gold", "silver", "bitcoin", "all"],
                         "periods": [40.0, 40.0, 40.0, 120.0], "ic": [0.10, 0.40, -0.90, 0.0],
                         "warn_periods": [10.0, 10.0, 10.0, 30.0], "warn_hit": [0.6, 0.6, 0.0, 0.4],
                         "warn_base": [0.45, 0.45, 0.45, 0.45], "warn_move": [-0.01, -0.01, 0.05, 0.0]})
    p = pooled(rows, "gold", "group").loc["m"]
    assert abs(p["periods"] - 60) < 1e-9, "own 40 + half of silver's 40"
    assert abs(p["ic"] - (40 * 0.10 + 20 * 0.40) / 60) < 1e-9, "bitcoin and the old 'all' row don't count"
    assert abs(p["trust"] - p["ic"] * np.sqrt(60)) < 1e-9
    assert abs(p["warn_hit"] - 0.6) < 1e-9 and p["warn_trust"] > 0
    own = pooled(rows, "gold", "own").loc["m"]
    assert abs(own["ic"] - 0.10) < 1e-9 and abs(own["periods"] - 40) < 1e-9
