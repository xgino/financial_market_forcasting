# paper-fund-v2/02_features/macro_features.py
"""Macro features per day (real rate and its change, yield curve, dollar moves, VIX vs its last year),
only values that were known that day. Writes data/features/macro.csv to look at.
Your own models compute the same through their view (core/features.py)."""
SETTINGS = {"enabled": True}

from core import store
from core.contracts import FeatureMaker
from core.features import macro_table


class MacroFeatures(FeatureMaker):
    about = "real rate, yield curve, dollar, VIX per day"

    def run(self, tl):
        if not tl.extra_names:
            return {"macro": "no macro data yet, run step 1 (fred_macro)"}
        m = macro_table(tl.view(tl.calendar()[-1]), None).round(6)
        m.index.name = "date"
        store.write_csv(m, store.path("features", "macro.csv"))
        return {"macro": f"{len(m.columns)} features x {len(m)} days"}