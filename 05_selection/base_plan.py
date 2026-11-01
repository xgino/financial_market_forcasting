# financial_market_forcasting1/05_selection/base_plan.py
"""Never follows a model on direction. The plain plan: the bar every other selector has to beat."""
SETTINGS = {"enabled": True}

from core.selection import Selector, plain


class BasePlan(Selector):
    about = "never follows a model on direction (the bar)"

    def choose(self, ctx):
        return plain(ctx, "base plan")