# financial_market_forcasting1/02_features/situations.py
"""
The situation of each asset on each day: trend (up / flat / down) + swings (calm / wild), past only,
and its stretch (core/stretches.py: the pattern chunk it is in, its start and length as known that day).
Settings in settings.SITUATIONS. Writes data/situations/<asset>.csv.
The battle uses the same labels to score every model per situation.
"""
SETTINGS = {"enabled": True}

from core import store
from core.contracts import FeatureMaker
from core.situations import regime_table
from core.stretches import stretch_table


class Situations(FeatureMaker):
    about = "up/flat/down x calm/wild per day"

    def run(self, tl):
        out = {}
        for asset in tl.assets:
            closes = tl.view(tl.calendar()[-1]).closes(asset)
            lab = regime_table(closes).rename("situation").to_frame().join(stretch_table(closes))
            lab.index.name = "date"
            store.write_csv(lab, store.path("situations", f"{asset}.csv"))
            share = lab["situation"].value_counts(normalize=True)
            out[asset] = ", ".join(f"{k} {v:.0%}" for k, v in share.items())
        return out