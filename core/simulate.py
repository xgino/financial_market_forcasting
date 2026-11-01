# paper-fund-v2/core/simulate.py
"""
What a strategy does with the money, month by month.

Every month (first trading day) the strategy gets what its selector chose per asset and the fund as it is
that day, and returns a target: how much of the fund in each asset (0.20 = 20%), the rest stays cash.
The trades happen at the NEXT trading day's open (you decide in the evening, buy the next morning), with
settings.FEE_RATE + settings.SPREAD on every euro traded. Trades smaller than settings.NO_TRADE_BAND of the
fund are skipped. settings.BOX3_RATE is taken once a year (0 = below the Dutch tax-free allowance).

KPIs (what an investor looks at):
  total / per_year   return over the whole period and per year
  max_drop           worst fall from a top (how much it can hurt)
  sharpe             return per unit of swings (0.5 decent, 1 good, above 2 suspicious)
  months_up          share of months that ended higher
  best / worst month, fees paid, avg invested
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import settings
from core.contracts import Player


@dataclass
class Fund:
    date: pd.Timestamp
    equity: float
    cash: float
    weights: dict        # asset -> current share of the fund
    peak: float          # highest equity so far
    start: float


class Strategy(Player):
    """06_strategies players. SETTINGS["follow"] = which selector's choices it uses.
    targets(fund, picks) -> ({asset: share of fund}, {asset: short reason})"""

    def targets(self, fund, picks):
        raise NotImplementedError


def month_steps(choices):
    """[(decision date, picks per asset)] once a month; decision date = when every asset's pick is known."""
    c = choices.copy()
    c["ym"] = c["date"].dt.to_period("M")
    return [(g["date"].max(), g.set_index("asset")) for _, g in c.groupby("ym")]


def simulate(strategy, choices, prices, start=None, end=None, start_eur=None):
    start_eur = start_eur or settings.START_EUR
    fee, band = settings.FEE_RATE + settings.SPREAD, settings.NO_TRADE_BAND   # every euro traded pays both
    c = choices[choices["selector"] == strategy.s["follow"]]
    if start is not None:
        c = c[c["date"] >= pd.Timestamp(start)]
    if end is not None:
        c = c[c["date"] < pd.Timestamp(end)]
    opens = {a: p["open_eur"] for a, p in prices.items()}
    closes = {a: p["close_eur"] for a, p in prices.items()}
    units = {a: 0.0 for a in prices}
    cash, peak, fees_paid = float(start_eur), float(start_eur), 0.0
    rows, trades = [], []
    taxes, tax_year = 0.0, None
    steps = month_steps(c)
    for i, (d, picks) in enumerate(steps):
        last = {a: float(closes[a][:d].iloc[-1]) if len(closes[a][:d]) else np.nan for a in prices}
        equity = cash + sum(units[a] * last[a] for a in prices if units[a] and np.isfinite(last[a]))
        if settings.BOX3_RATE and tax_year is not None and d.year != tax_year and equity > 0:
            tax = settings.BOX3_RATE * equity          # once a year, taken evenly from cash and every holding
            cash -= tax * cash / equity
            units = {a: u * (1 - tax / equity) for a, u in units.items()}
            taxes, equity = taxes + tax, equity - tax
        tax_year = d.year
        peak = max(peak, equity)
        weights = {a: units[a] * last[a] / equity if equity and units[a] else 0.0 for a in prices}
        target, notes = strategy.targets(Fund(d, equity, cash, weights, peak, start_eur), picks)
        target = {a: max(0.0, float(w)) for a, w in target.items() if a in prices and np.isfinite(last[a])}
        if sum(target.values()) > 1:
            s = sum(target.values())
            target = {a: w / s for a, w in target.items()}
        orders = {a: target.get(a, weights[a]) * equity - weights[a] * equity for a in prices}
        orders = {a: v for a, v in orders.items() if abs(v) >= band * equity}
        for a, value in sorted(orders.items(), key=lambda kv: kv[1]):          # sells first, frees cash
            nxt = opens[a][opens[a].index > d]
            if nxt.empty:
                continue
            day, px = nxt.index[0], float(nxt.iloc[0])
            if value < 0:
                q = min(units[a], -value / px)
                gross = q * px
                units[a] -= q
                cash += gross * (1 - fee)
                fees_paid += gross * fee
                trades.append({"decided": d, "date": day, "asset": a, "action": "SELL", "eur": gross, "price": px,
                               "note": notes.get(a, "")})
            else:
                spend = min(value, cash)
                if spend <= 0:
                    continue
                units[a] += spend * (1 - fee) / px
                cash -= spend
                fees_paid += spend * fee
                trades.append({"decided": d, "date": day, "asset": a, "action": "BUY", "eur": spend, "price": px,
                               "note": notes.get(a, "")})
        if settings.CASH_RATE and i + 1 < len(steps):
            cash *= (1 + settings.CASH_RATE) ** ((steps[i + 1][0] - d).days / 365.25)
        rows.append({"date": d, "equity": equity, "cash_before": cash, "fees_so_far": fees_paid, "tax_so_far": taxes,
                     **{f"w_{a}": target.get(a, weights[a]) for a in prices},
                     **{f"why_{a}": notes.get(a, "") for a in prices}})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    # value at the end: last close of the period
    stop = pd.Timestamp(end) if end is not None else max(s.index[-1] for s in closes.values())
    last = {a: float(closes[a][:stop].iloc[-1]) for a in prices}
    final = cash + sum(units[a] * last[a] for a in prices)
    months = pd.DataFrame(rows)
    months["equity_end"] = list(months["equity"].iloc[1:]) + [final]
    months["month_return"] = months["equity_end"] / months["equity"] - 1
    return months, pd.DataFrame(trades)


def kpis(months, start_eur=None):
    start_eur = start_eur or settings.START_EUR
    if months.empty:
        return {}
    eq = pd.concat([pd.Series([start_eur]), months["equity_end"]], ignore_index=True)
    r = months["month_return"]
    years = max((months["date"].iloc[-1] - months["date"].iloc[0]).days / 365.25 + 1 / 12, 1e-9)
    inv = months[[c for c in months.columns if c.startswith("w_")]].sum(axis=1)
    return {"end_eur": eq.iloc[-1], "total": eq.iloc[-1] / start_eur - 1,
            "per_year": (eq.iloc[-1] / start_eur) ** (1 / years) - 1,
            "max_drop": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan,
            "months_up": float((r > 0).mean()), "best_month": float(r.max()), "worst_month": float(r.min()),
            "fees": float(months["fees_so_far"].iloc[-1]), "tax": float(months["tax_so_far"].iloc[-1]),
            "avg_invested": float(inv.mean())}


def tilt(picks, base, tilt_share, min_edge):
    """Shares per asset: base spread evenly, plus or minus a tilt when the pick expects a move clearly
    bigger than the fees. A pick with no opinion (middle 0%, the base plan) just gets the base."""
    n = max(len(picks), 1)
    need = 2 * (settings.FEE_RATE + settings.SPREAD) + min_edge
    out, why = {}, {}
    for a, p in picks.iterrows():
        e = p["p50"]
        if e > need:
            out[a], why[a] = (base + tilt_share) / n, f"{p['main_model']} expects {e:+.1%}: more"
        elif e < -need:
            out[a], why[a] = max(base - tilt_share, 0) / n, f"{p['main_model']} expects {e:+.1%}: less"
        else:
            out[a], why[a] = base / n, ("no trusted model: base" if e == 0 else f"expects {e:+.1%}: too small, base")
    return out, why