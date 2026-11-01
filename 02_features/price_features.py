# financial_market_forcasting1/02_features/price_features.py
"""
Simple numbers that describe the price line on each day (trend, swings, distance from averages, RSI).
Every number on day D only uses closes up to and including D.
Writes data/features/<asset>.csv so you can look at them. Models compute the same numbers themselves
from their view (core/features.py), so the no-future check covers them too.
"""
SETTINGS = {"enabled": True}

from core import store
from core.contracts import FeatureMaker
from core.features import feature_table


class PriceFeatures(FeatureMaker):
    about = "trend, swings, averages and RSI per day"

    def run(self, tl):
        out = {}
        for asset in tl.assets:
            f = feature_table(tl.view(tl.calendar()[-1]).closes(asset)).round(6)
            f.index.name = "date"
            store.write_csv(f, store.path("features", f"{asset}.csv"))
            out[asset] = f"{len(f.columns)} features x {len(f)} days"
        return out