# paper-fund-v2/01_data/yahoo_macro.py
"""
Back-up for the macro data when FRED doesn't answer: the same kind of information from Yahoo.
Only fills in what FRED didn't deliver today (a missing or old file), so FRED stays the first choice.
  vix     ^VIX        the fear index
  rate10  ^TNX        10 year US rate (Yahoo gives it x10, corrected here)
  curve   ^TNX - ^IRX 10 year minus 3 month rate (FRED's version is 10 year minus 2 year)
  dollar  DX-Y.NYB    the dollar index
The real interest rate has no Yahoo version, so it's only there when FRED works.
"""
SETTINGS = {"enabled": True}

from datetime import datetime

import pandas as pd

from core import store
from core.contracts import DataSource


def close(ticker):
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
    idx = pd.to_datetime(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return df["Close"].dropna()


def fresh_today(name):
    p = store.path("macro", f"{name}.csv")
    return p.exists() and datetime.fromtimestamp(p.stat().st_mtime).date() == datetime.now().date()


class YahooMacro(DataSource):
    about = "back-up macro data from Yahoo (VIX, 10y rate, curve, dollar) when FRED fails"

    def run(self):
        report = {}
        makers = {
            "vix": lambda: close("^VIX"),
            "rate10": lambda: close("^TNX") / 10 if close("^TNX").median() > 20 else close("^TNX"),
            "curve": lambda: (close("^TNX") - close("^IRX")).dropna() / (10 if close("^TNX").median() > 20 else 1),
            "dollar": lambda: close("DX-Y.NYB"),
        }
        for name, make in makers.items():
            if fresh_today(name):
                report[name] = "FRED delivered it today, kept"
                continue
            try:
                s = make()
                df = pd.DataFrame({"date": s.index, "value": s.to_numpy()})
                store.write_csv(df, store.path("macro", f"{name}.csv"), index=False)
                report[name] = f"from Yahoo: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}"
            except Exception as e:
                report[name] = f"FAILED ({e})"
        return report