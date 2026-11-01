# paper-fund-v2/core/wallet.py
"""
Virtual wallet as an append-only, hash-chained ledger (SQLite, no extra installs).

Rules
- Created once with a start balance (GENESIS entry, with the date). That entry never changes.
- Entries can only be appended. UPDATE and DELETE are blocked by SQLite triggers.
- Every entry stores the hash of the previous one, so editing anything breaks the chain.
- The full chain is verified every time the wallet opens. Broken chain = wallet refuses to run.
- Money is integer cents, qty and prices are Decimal strings. No float rounding drift.
- Balance and positions are never stored, they're recomputed from the entries every time.
- One shared cash pool. Gold, stocks, ETFs all trade from the same budget.

Trades are lots: open_trade() spends cash and returns a trade_id, close_trade() sells that
whole lot and books the result. Put forecasts, metrics, expected return etc in meta, it's
part of the hash too so it can't be "corrected" afterwards either.

Used by the simulation step (every strategy gets its own wallet) and later by live paper trading.
Check a wallet any time:  python core/wallet.py status --db data/wallets/<file>.db
"""
import argparse
import hashlib
import json
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "wallets" / "live.db"
GENESIS_PREV = "0" * 64
QTY_STEP = Decimal("0.0000000001")
ONE = Decimal("1")

FIELDS = ("seq", "ts", "kind", "trade_id", "asset", "qty", "price",
          "cash_cents", "fee_cents", "meta", "prev_hash")

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    seq        INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('GENESIS', 'OPEN', 'CLOSE')),
    trade_id   TEXT,
    asset      TEXT,
    qty        TEXT,
    price      TEXT,
    cash_cents INTEGER NOT NULL,
    fee_cents  INTEGER NOT NULL,
    meta       TEXT NOT NULL,
    prev_hash  TEXT NOT NULL,
    hash       TEXT NOT NULL UNIQUE
);
CREATE TRIGGER IF NOT EXISTS entries_no_update BEFORE UPDATE ON entries
BEGIN SELECT RAISE(ABORT, 'ledger is append-only, entries cannot be changed'); END;
CREATE TRIGGER IF NOT EXISTS entries_no_delete BEFORE DELETE ON entries
BEGIN SELECT RAISE(ABORT, 'ledger is append-only, entries cannot be deleted'); END;
"""


class LedgerError(Exception):
    pass


# ---------- helpers ----------

def to_dec(x):
    return Decimal(str(x))


def to_cents(eur):
    return int((to_dec(eur) * 100).quantize(ONE, ROUND_HALF_UP))


def eur(cents):
    return f"EUR {Decimal(cents) / 100:,.2f}"


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def entry_hash(e):
    payload = json.dumps({k: e[k] for k in FIELDS}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def connect(path):
    db = sqlite3.connect(path, isolation_level=None)  # transactions handled by tx()
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)  # also puts the triggers back if someone dropped them
    return db


@contextmanager
def tx(db):
    db.execute("BEGIN IMMEDIATE")  # locks the db so nothing else writes in between
    try:
        yield
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def append(db, kind, cash_cents, fee_cents=0, trade_id=None, asset=None,
           qty=None, price=None, meta=None):
    # must be called inside tx()
    last = db.execute("SELECT seq, hash FROM entries ORDER BY seq DESC LIMIT 1").fetchone()
    e = {
        "seq": last["seq"] + 1 if last else 1,
        "ts": now_utc(),
        "kind": kind,
        "trade_id": trade_id,
        "asset": asset,
        "qty": qty,
        "price": price,
        "cash_cents": int(cash_cents),
        "fee_cents": int(fee_cents),
        "meta": json.dumps(meta or {}, sort_keys=True, separators=(",", ":"), default=str),
        "prev_hash": last["hash"] if last else GENESIS_PREV,
    }
    e["hash"] = entry_hash(e)
    cols = ", ".join(e)
    db.execute(f"INSERT INTO entries ({cols}) VALUES ({', '.join(':' + c for c in e)})", e)
    return e


def new_state():
    return {"seq": 0, "head": GENESIS_PREV, "genesis": None, "start_cents": 0,
            "cash_cents": 0, "open": {}, "closed": [], "seen": set()}


def apply(st, r):
    """Check one entry (chain + rules) and add it to the state."""
    r = dict(r)
    where = f"entry {r['seq']}"
    if r["seq"] != st["seq"] + 1:
        raise LedgerError(f"entry missing before seq {r['seq']}")
    if r["prev_hash"] != st["head"]:
        raise LedgerError(f"chain broken at {where}")
    if entry_hash(r) != r["hash"]:
        raise LedgerError(f"{where} was modified")
    first = st["seq"] == 0
    if first and r["kind"] != "GENESIS":
        raise LedgerError("first entry must be GENESIS")
    if r["kind"] == "GENESIS":
        if not first or r["cash_cents"] <= 0:
            raise LedgerError(f"{where}: invalid GENESIS")
        st["genesis"], st["start_cents"] = r["hash"], r["cash_cents"]
    elif r["kind"] == "OPEN":
        if r["trade_id"] in st["seen"] or r["cash_cents"] >= 0:
            raise LedgerError(f"{where}: invalid OPEN")
        st["seen"].add(r["trade_id"])
        st["open"][r["trade_id"]] = r
    elif r["kind"] == "CLOSE":
        o = st["open"].pop(r["trade_id"], None)
        if o is None or o["asset"] != r["asset"] or o["qty"] != r["qty"] or r["cash_cents"] < 0:
            raise LedgerError(f"{where}: CLOSE does not match an open trade")
        st["closed"].append((o, r))
    st["cash_cents"] += r["cash_cents"]
    if st["cash_cents"] < 0:
        raise LedgerError(f"{where}: cash went negative")
    st["seq"], st["head"] = r["seq"], r["hash"]


def replay(rows):
    """Rebuild everything from scratch, checking the chain and every rule at every step."""
    st = new_state()
    for r in rows:
        apply(st, r)
    if st["seq"] == 0:
        raise LedgerError("wallet is empty, no GENESIS entry")
    return st


# ---------- wallet ----------

class Wallet:
    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        if not self.path.exists():
            raise LedgerError(f"No wallet at {self.path}. Create one first.")
        self.db = connect(self.path)
        self._cache = None
        self.verify()

    @classmethod
    def create(cls, start_eur, path=DEFAULT_DB, note=""):
        path = Path(path)
        if path.exists():
            raise LedgerError(f"Wallet already exists at {path}. A wallet can only be created once.")
        start = to_cents(start_eur)
        if start <= 0:
            raise LedgerError("start balance must be above 0")
        path.parent.mkdir(parents=True, exist_ok=True)
        db = connect(path)
        with tx(db):
            append(db, "GENESIS", cash_cents=start, meta={"start_eur": str(Decimal(start) / 100), "note": note})
        db.close()
        return cls(path)

    def _state(self):
        # full check once when the wallet opens, after that only new entries are read and
        # checked (chain + rules), so long simulations stay fast
        if self._cache is None:
            self._cache = replay(self.db.execute("SELECT * FROM entries ORDER BY seq").fetchall())
            return self._cache
        try:
            for r in self.db.execute("SELECT * FROM entries WHERE seq > ? ORDER BY seq", (self._cache["seq"],)):
                apply(self._cache, r)
        except LedgerError:
            self._cache = None
            raise
        return self._cache

    # ----- integrity -----

    def verify(self, expect_genesis=None):
        """Full check of the whole chain from the first entry."""
        self._cache = None
        st = self._state()
        if expect_genesis and st["genesis"] != expect_genesis:
            raise LedgerError("genesis hash does not match the one you saved, wallet was replaced")
        return {"entries": st["seq"], "genesis": st["genesis"], "head": st["head"]}

    # ----- trading -----

    def open_trade(self, asset, amount_eur, price, fee_rate=0, meta=None):
        """Spend amount_eur (fee included) on asset at price. Returns trade_id."""
        amount, price, fee_rate = to_cents(amount_eur), to_dec(price), to_dec(fee_rate)
        if amount <= 0 or price <= 0 or not (0 <= fee_rate < 1):
            raise LedgerError("amount and price must be above 0, fee_rate between 0 and 1")
        fee = int((Decimal(amount) * fee_rate).quantize(ONE, ROUND_UP))
        qty = (Decimal(amount - fee) / 100 / price).quantize(QTY_STEP, ROUND_DOWN)
        if qty <= 0:
            raise LedgerError("amount too small to buy anything")
        trade_id = uuid.uuid4().hex[:12]
        with tx(self.db):
            cash = self._state()["cash_cents"]
            if amount > cash:
                raise LedgerError(f"not enough cash: have {eur(cash)}, need {eur(amount)}")
            append(self.db, "OPEN", cash_cents=-amount, fee_cents=fee, trade_id=trade_id,
                   asset=str(asset).strip(), qty=str(qty), price=str(price), meta=meta)
        return trade_id

    def close_trade(self, trade_id, price, fee_rate=0, meta=None):
        """Sell the whole lot at price. Returns the result of the trade."""
        price, fee_rate = to_dec(price), to_dec(fee_rate)
        if price <= 0 or not (0 <= fee_rate < 1):
            raise LedgerError("price must be above 0, fee_rate between 0 and 1")
        with tx(self.db):
            o = self._state()["open"].get(trade_id)
            if o is None:
                raise LedgerError(f"no open trade {trade_id}")
            gross = int((to_dec(o["qty"]) * price * 100).quantize(ONE, ROUND_DOWN))
            fee = int((Decimal(gross) * fee_rate).quantize(ONE, ROUND_UP))
            append(self.db, "CLOSE", cash_cents=gross - fee, fee_cents=fee, trade_id=trade_id,
                   asset=o["asset"], qty=o["qty"], price=str(price), meta=meta)
        cost = -o["cash_cents"]
        pnl = gross - fee - cost
        return {"trade_id": trade_id, "asset": o["asset"], "cost_cents": cost,
                "proceeds_cents": gross - fee, "pnl_cents": pnl, "return": pnl / cost}

    # ----- reading -----

    def cash(self):
        return Decimal(self._state()["cash_cents"]) / 100

    def open_trades(self):
        return [{"trade_id": t, "asset": o["asset"], "qty": Decimal(o["qty"]),
                 "entry_price": Decimal(o["price"]), "cost_cents": -o["cash_cents"],
                 "opened_at": o["ts"], "meta": json.loads(o["meta"])}
                for t, o in self._state()["open"].items()]

    def closed_trades(self):
        out = []
        for o, c in self._state()["closed"]:
            cost = -o["cash_cents"]
            out.append({"trade_id": o["trade_id"], "asset": o["asset"],
                        "opened_at": o["ts"], "closed_at": c["ts"],
                        "entry_price": Decimal(o["price"]), "exit_price": Decimal(c["price"]),
                        "cost_cents": cost, "proceeds_cents": c["cash_cents"],
                        "pnl_cents": c["cash_cents"] - cost,
                        "return": (c["cash_cents"] - cost) / cost,
                        "open_meta": json.loads(o["meta"]), "close_meta": json.loads(c["meta"])})
        return out

    def summary(self, prices=None):
        """prices: optional {asset: current price} to value open trades at market."""
        st = self._state()
        at_cost = sum(-o["cash_cents"] for o in st["open"].values())
        realized = sum(c["cash_cents"] + o["cash_cents"] for o, c in st["closed"])
        s = {"start_cents": st["start_cents"], "cash_cents": st["cash_cents"],
             "open_at_cost_cents": at_cost, "realized_pnl_cents": realized,
             "open_trades": len(st["open"]), "closed_trades": len(st["closed"])}
        if prices:
            missing = {o["asset"] for o in st["open"].values()} - set(prices)
            if not missing:
                market = sum(int((to_dec(o["qty"]) * to_dec(prices[o["asset"]]) * 100).quantize(ONE, ROUND_DOWN))
                             for o in st["open"].values())
                s["open_market_cents"] = market
                s["equity_cents"] = st["cash_cents"] + market
                s["total_return"] = s["equity_cents"] / st["start_cents"] - 1
        return s


# ---------- CLI ----------

def main(argv=None):
    p = argparse.ArgumentParser(description="Virtual wallet ledger")
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--start", required=True, help="start balance in EUR")
    c.add_argument("--note", default="")
    sub.add_parser("status")
    v = sub.add_parser("verify")
    v.add_argument("--expect-genesis", default=None)
    args = p.parse_args(argv)

    try:
        if args.cmd == "create":
            w = Wallet.create(args.start, args.db, args.note)
            info = w.verify()
            print(f"Wallet created with {eur(to_cents(args.start))}")
            print(f"Genesis hash (save this somewhere outside this machine):\n{info['genesis']}")
            return 0

        w = Wallet(args.db)
        info = w.verify(getattr(args, "expect_genesis", None))
        if args.cmd == "verify":
            print(f"OK, {info['entries']} entries, chain intact")
            print(f"genesis {info['genesis']}\nhead    {info['head']}")
            return 0

        s = w.summary()
        print(f"Start:          {eur(s['start_cents'])}")
        print(f"Cash:           {eur(s['cash_cents'])}")
        print(f"Open (at cost): {eur(s['open_at_cost_cents'])} in {s['open_trades']} trades")
        print(f"Realized P&L:   {eur(s['realized_pnl_cents'])} from {s['closed_trades']} closed trades")
        for t in w.open_trades():
            print(f"  open {t['trade_id']} {t['asset']} qty {t['qty']} @ {t['entry_price']} "
                  f"cost {eur(t['cost_cents'])} since {t['opened_at']}")
        print(f"Chain OK, head {info['head'][:16]}...")
        return 0
    except LedgerError as e:
        print(f"WALLET ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())