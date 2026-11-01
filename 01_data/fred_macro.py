# paper-fund-v2/01_data/fred_macro.py
"""
Macro data from FRED (the US Federal Reserve's free data service), no account or key needed.
Series and names in settings.MACRO_SERIES: real interest rate, 10 year rate, yield curve, dollar, VIX.
If FRED is slow or unreachable, 01_data/yahoo_macro.py fills in what it can from Yahoo.
Writes data/macro/<name>.csv: date, value. Values are only used from the next trading day on
(settings.MACRO_LAG_DAYS), because they're published with a delay.
"""
SETTINGS = {"enabled": True, "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"}

import io
import time
import urllib.request

import pandas as pd

import settings
from core import store
from core.contracts import DataSource


def download(series_id, url, tries=3):
    req = urllib.request.Request(url.format(id=series_id), headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "Accept": "text/csv,*/*"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                df = pd.read_csv(io.StringIO(r.read().decode()))
            break
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    df.columns = ["date", "value"]                       # first column is the date, whatever it's called
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED writes "." for missing days
    return df.dropna()


class FredMacro(DataSource):
    about = "macro data from FRED: real rate, 10y rate, yield curve, dollar, VIX"

    def run(self):
        report = {}
        for name, sid in settings.MACRO_SERIES.items():
            try:
                df = download(sid, self.s["url"])
            except Exception as e:
                report[name] = f"FAILED ({e}), skipped"
                continue
            store.write_csv(df, store.path("macro", f"{name}.csv"), index=False)
            report[name] = f"{sid}: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}, last {df['value'].iloc[-1]}"
        return report