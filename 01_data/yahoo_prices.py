# paper-fund-v2/01_data/yahoo_prices.py
"""
Daily prices from Yahoo Finance for every asset in settings.ASSETS and settings.TRAIN_ONLY_ASSETS, in EUR.
Writes data/prices/<asset>.csv: date, open_eur, close_eur, close_raw, fx_close

- USD assets: the open with that day's EUR/USD open, the close with that day's EUR/USD close, so an
  open never uses information from later that day.
- EUR/USD repair: Yahoo has some bad EUR/USD days (a rate that jumps away and comes straight back,
  e.g. in 2008). Those get the straight line between the good days around them. Every repair is printed.
- Today's bar is left out until it's final (tomorrow), so a re-download never changes old days.
- Assets are NOT repaired, only flagged: a market can really open very low and recover the same day.
- currency "LOCAL": kept in its own currency (training-only assets: only the moves matter there).
  "until": stop at that date, so an asset that continues under another name isn't counted twice.
"""
SETTINGS = {
    "enabled": True,
    "fx_ticker": "EURUSD=X",
    "fx_spike": 0.025,      # EUR/USD jumping 2.5%+ and straight back = bad data, repaired
    "asset_spike": 0.08,    # an asset doing that by 8%+ gets flagged on screen, not changed
    "settle": 0.25,         # "straight back" = the days around it differ by less than 25% of the jump
}

import numpy as np
import pandas as pd

import settings
from core import store
from core.contracts import DataSource


def download(ticker):
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"no data returned for {ticker}, check the ticker on finance.yahoo.com")
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # today's bar isn't final yet (markets still open, crypto never closes): it would change on the next
    # download and make the forecast cache redo old days. It's added tomorrow, when it's final.
    df = df[df.index < pd.Timestamp.today().normalize()]
    df = df[["Open", "Close"]].dropna()
    return df[(df["Open"] > 0) & (df["Close"] > 0)]


def spikes(s, threshold, settle, max_days=2):
    """Values that jump more than threshold away and come straight back within max_days.
    Returns the repaired series and a list of (date, old, new)."""
    lv = np.log(s.to_numpy(dtype=float))
    fixes, i, n = [], 1, len(lv)
    while i < n - 1:
        fixed = False
        for k in range(1, max_days + 1):
            j = i + k
            if j >= n:
                break
            before, after = lv[i - 1], lv[j]
            gap = lv[i:j] - before
            if abs(after - before) > max(0.01, settle * np.abs(gap).min()):
                continue
            if np.all(gap > threshold) or np.all(gap < -threshold):
                for m in range(k):
                    new = before + (after - before) * (m + 1) / (k + 1)
                    fixes.append((s.index[i + m], float(np.exp(lv[i + m])), float(np.exp(new))))
                    lv[i + m] = new
                i, fixed = j, True
                break
        if not fixed:
            i += 1
    return pd.Series(np.exp(lv), index=s.index, name=s.name), fixes


class YahooPrices(DataSource):
    about = "Daily prices from Yahoo Finance, in EUR, with EUR/USD repair"

    def run(self):
        s = self.s
        report = {}
        fx = None
        if any(str(a.get("currency", "EUR")).upper() == "USD" for a in settings.ALL_ASSETS.values()):
            fx = download(s["fx_ticker"])
            repaired = 0
            for col in ("Open", "Close"):
                fx[col], fixes = spikes(fx[col], s["fx_spike"], s["settle"])
                repaired += len(fixes)
                for d, old, new in fixes:
                    print(f"  repaired EURUSD {col.lower()} {d.date()}: {old:.4f} -> {new:.4f}")
            report["EUR/USD repairs"] = repaired
        for name, spec in settings.ALL_ASSETS.items():
            try:
                raw = download(spec["ticker"])
            except Exception as e:  # one broken ticker shouldn't stop the rest
                print(f"  {name} ({spec['ticker']}): FAILED, skipped ({e})")
                continue
            for col in ("Open", "Close"):
                for d, old, new in spikes(raw[col], s["asset_spike"], s["settle"])[1]:
                    if name in settings.ASSETS:   # only worth a look for assets you trade
                        print(f"  check on a chart, NOT changed: {name} {col.lower()} {d.date()} {old:,.2f} (around it ~{new:,.2f})")
            out = pd.DataFrame(index=raw.index)
            if spec.get("until"):
                raw = raw[raw.index <= pd.Timestamp(spec["until"])]   # stop where an overlapping asset starts
            if str(spec.get("currency", "EUR")).upper() == "USD":
                f = fx.reindex(fx.index.union(out.index)).ffill().reindex(out.index)
                out["fx_open"], out["fx_close"] = f["Open"], f["Close"]
            else:
                out["fx_open"] = out["fx_close"] = 1.0
            out["open_eur"] = raw["Open"] / out["fx_open"]
            out["close_eur"] = raw["Close"] / out["fx_close"]
            out["close_raw"] = raw["Close"]
            out = out.dropna()[["open_eur", "close_eur", "close_raw", "fx_close"]].round(6)
            out.index.name = "date"
            store.write_csv(out, store.path("prices", f"{name}.csv"))
            gaps = out.index.to_series().diff().dt.days
            report[name] = f"{len(out)} days, {out.index[0].date()} to {out.index[-1].date()}, longest gap {int(gaps.max())} days"
        return report