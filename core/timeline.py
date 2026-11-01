# paper-fund-v2/core/timeline.py
"""
Core, stage 1: the ONLY way the rest of the system reads market data.

The Timeline holds all prices, but nothing gets the raw table. Models, selectors, approaches and the
scoreboard get a View for one date, and a View only shows data up to and including that date.
Asking for anything later raises FutureDataError, so a lookahead bug crashes loudly instead of
quietly making results look great.

  tl = Timeline.load(["gold"])
  v = tl.view("2013-01-02")
  v.closes("gold", 512)                    # last 512 closes, ending 2013-01-02
  v.known_return("gold", "2012-11-01", 20) # 20 day result, None if not finished by 2013-01-02
  tl.next_open("gold", "2013-01-02")       # runner only: the fill price of the next trading day

Tested by tests/test_core.py (python main.py test).
"""
import hashlib

import numpy as np
import pandas as pd


class FutureDataError(Exception):
    pass


def _day(date):
    return pd.Timestamp(date).normalize()


class Timeline:
    def __init__(self, prices, extras=None):
        """prices: {asset: DataFrame with open_eur and close_eur, indexed by date}
        extras: {name: Series indexed by the date it became KNOWN} (macro data), same no-future rules"""
        self._idx, self._open, self._close = {}, {}, {}
        for a, df in prices.items():
            df = df.sort_index()
            idx = pd.DatetimeIndex(df.index).normalize()
            if idx.has_duplicates:
                raise ValueError(f"{a}: duplicate dates in the price data")
            o = df["open_eur"].to_numpy(dtype=float).copy()
            c = df["close_eur"].to_numpy(dtype=float).copy()
            if np.isnan(o).any() or np.isnan(c).any() or (o <= 0).any() or (c <= 0).any():
                raise ValueError(f"{a}: missing or zero prices, re-run market.prices")
            o.setflags(write=False)
            c.setflags(write=False)
            self._idx[a], self._open[a], self._close[a] = idx, o, c
        self.assets = tuple(prices)
        self._xidx, self._xval = {}, {}
        for name, ser in (extras or {}).items():
            ser = ser.dropna().sort_index()
            ser = ser[~ser.index.duplicated(keep="last")]
            v = ser.to_numpy(dtype=float).copy()
            v.setflags(write=False)
            self._xidx[name], self._xval[name] = pd.DatetimeIndex(ser.index).normalize(), v
        self.extra_names = tuple(self._xidx)
        self._calendar = pd.DatetimeIndex(sorted(set().union(*[set(i) for i in self._idx.values()])))

    @classmethod
    def load(cls, assets=None):
        """Every asset in settings.ASSETS (or the ones given), from data/prices/<asset>.csv."""
        import settings
        from core.store import read_macro, read_prices
        return cls({a: read_prices(a) for a in (assets or settings.ASSETS)}, read_macro())

    def extras(self):
        """{name: Series} copies of the extra series (for tests and the no-future check)."""
        return {n: pd.Series(self._xval[n].copy(), index=self._xidx[n]) for n in self.extra_names}

    # ----- dates -----

    def calendar(self, start=None, end=None):
        """All days where at least one asset traded."""
        return self._calendar[(self._calendar >= _day(start or self._calendar[0])) &
                              (self._calendar <= _day(end or self._calendar[-1]))]

    def month_starts(self, start=None, end=None):
        """First trading day of every month, for stepping month by month. Weekdays only: crypto trades on
        weekends, the rest of the market doesn't."""
        cal = self.calendar(start, end)
        cal = cal[cal.weekday < 5]
        s = pd.Series(cal, index=cal)
        return pd.DatetimeIndex(s.groupby([cal.year, cal.month]).min().to_numpy())

    def first_date(self, asset, min_history=0):
        """First date an asset has at least min_history earlier days of data."""
        idx = self._idx[asset]
        return idx[min(min_history, len(idx) - 1)]

    # ----- access -----

    def view(self, date):
        return View(self, _day(date))

    def next_open(self, asset, date):
        """(date, open) of the first bar AFTER date. Runner only, this is the future for a decision on date."""
        idx = self._idx[asset]
        i = idx.searchsorted(_day(date), side="right")
        if i >= len(idx):
            return None
        return idx[i], float(self._open[asset][i])

    def fingerprint(self):
        """Short hash of all the data. Two runs with the same fingerprint used exactly the same prices."""
        h = hashlib.sha256()
        for a in sorted(self.assets):
            h.update(a.encode())
            h.update(self._idx[a].asi8.tobytes())
            h.update(self._open[a].tobytes())
            h.update(self._close[a].tobytes())
        for n in sorted(self.extra_names):
            h.update(n.encode())
            h.update(self._xidx[n].asi8.tobytes())
            h.update(self._xval[n].tobytes())
        return h.hexdigest()[:16]

    def _xcount(self, name, date):
        return int(self._xidx[name].searchsorted(date, side="right"))

    def _count(self, asset, date):
        """How many bars of asset are on or before date."""
        return int(self._idx[asset].searchsorted(date, side="right"))


class View:
    """Read-only window on all data up to and including `date`. Everything returned is a copy."""

    def __init__(self, timeline, date):
        self._tl = timeline
        self.date = date
        self.assets = timeline.assets
        self.extra_names = timeline.extra_names

    def extra(self, name, n=None):
        """An extra series (macro) up to the view date: only values that were already known."""
        end = self._tl._xcount(name, self.date)
        start = 0 if n is None else max(0, end - n)
        return pd.Series(self._tl._xval[name][start:end].copy(), index=self._tl._xidx[name][start:end], name=name)

    def __repr__(self):
        return f"View({self.date.date()})"

    def _check(self, date):
        d = _day(date)
        if d > self.date:
            raise FutureDataError(f"asked for {d.date()} but this view is for {self.date.date()}")
        return d

    def _end(self, asset):
        return self._tl._count(asset, self.date)

    def has_bar(self, asset):
        """True if the asset traded today."""
        n = self._end(asset)
        return n > 0 and self._tl._idx[asset][n - 1] == self.date

    def n_bars(self, asset):
        return self._end(asset)

    def closes(self, asset, n=None):
        """Last n closes (all if n is None), ending today or the last trading day before."""
        end = self._end(asset)
        start = 0 if n is None else max(0, end - n)
        return pd.Series(self._tl._close[asset][start:end].copy(), index=self._tl._idx[asset][start:end],
                         name=asset)

    def bars(self, asset, n=None):
        end = self._end(asset)
        start = 0 if n is None else max(0, end - n)
        idx = self._tl._idx[asset][start:end]
        return pd.DataFrame({"open_eur": self._tl._open[asset][start:end].copy(),
                             "close_eur": self._tl._close[asset][start:end].copy()}, index=idx)

    def last_close(self, asset):
        end = self._end(asset)
        return float(self._tl._close[asset][end - 1]) if end else None

    def close_on(self, asset, date):
        """Close on date, or the last close before it. date may not be after the view date."""
        n = self._tl._count(asset, self._check(date))
        return float(self._tl._close[asset][n - 1]) if n else None

    def known_return(self, asset, start, horizon):
        """Return from the close on start to the close horizon trading days later.
        None if that end day is after the view date: on this date you can't know it yet."""
        s = self._check(start)
        idx = self._tl._idx[asset]
        i = int(idx.searchsorted(s, side="left"))
        if i >= len(idx) or idx[i] != s:
            raise KeyError(f"{s.date()} is not a trading day for {asset}")
        j = i + horizon
        if j >= len(idx) or idx[j] > self.date:
            return None
        c = self._tl._close[asset]
        return float(c[j] / c[i] - 1)


# ---------- self test ----------

def self_test(tl, asset, horizon=20, samples=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = tl._idx[asset]
    days = idx[rng.choice(np.arange(horizon + 5, len(idx) - horizon - 5), size=min(samples, len(idx) // 2))]
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    # 1. a view never returns rows after its date
    check("views never return rows after their date",
          all(tl.view(d).closes(asset).index.max() <= d for d in days))

    # 2. asking for later data raises
    d = days[0]
    try:
        tl.view(d).close_on(asset, d + pd.Timedelta(days=1))
        raised = False
    except FutureDataError:
        raised = True
    check("asking for a later date raises FutureDataError", raised)

    # 3. outcomes only become known once they're finished
    i = idx.get_loc(d)
    start_recent, start_done = idx[i - horizon + 1], idx[i - horizon]
    v = tl.view(d)
    check("a 20 day result that ends after today is unknown (None)", v.known_return(asset, start_recent, horizon) is None)
    expected = tl._close[asset][i] / tl._close[asset][i - horizon] - 1
    got = v.known_return(asset, start_done, horizon)
    check("a 20 day result that ended today is known and correct", got is not None and abs(got - expected) < 1e-12)

    # 4. changing the future changes nothing a view can see
    cutoff = days[len(days) // 2]
    fake = {}
    for a in tl.assets:
        o, c, ix = tl._open[a].copy(), tl._close[a].copy(), tl._idx[a]
        later = ix > cutoff
        o[later] *= rng.uniform(0.5, 1.5, later.sum())
        c[later] *= rng.uniform(0.5, 1.5, later.sum())
        fake[a] = pd.DataFrame({"open_eur": o, "close_eur": c}, index=ix)
    tl_fake = Timeline(fake)
    same = all(tl.view(x).closes(asset).equals(tl_fake.view(x).closes(asset)) for x in days[days <= cutoff])
    check("scrambling data after a date changes nothing visible on or before it", same)

    # 5. what you get back is a copy, editing it can't damage the timeline
    s = tl.view(d).closes(asset, 5)
    before = tl.view(d).last_close(asset)
    s.iloc[-1] = -1
    check("returned data is a copy, editing it changes nothing", tl.view(d).last_close(asset) == before)

    # 6. next_open is strictly after the date
    nxt = tl.next_open(asset, d)
    check("next_open is the next trading day, never the same day", nxt is None or nxt[0] > d)
    return ok