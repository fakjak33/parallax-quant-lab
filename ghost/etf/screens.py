"""Declarative ETF spec + point-in-time selection / weight-schedule builder.

An ``ETFSpec`` describes how to build a fund: a universe, an optional ranking
factor (top/bottom N) or an explicit basket, optional filters, a weighting
scheme, a rebalance cadence, and a direction (long / short / long_short).
``build_weight_schedule`` turns that into signed target weights on each
rebalance date — recomputing selection from data available *as of* that date.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import factors, weighting
from ..data.universe import UNIVERSES
from ..data.stocks500 import STOCKS_500

# factors/inputs that come from the fundamental SNAPSHOT (class B → look-ahead)
CLASS_B_FACTORS = {"fcf_per_share", "pe", "forward_pe", "revenue",
                   "profit_margin", "price_to_book", "dividend_yield",
                   "sector", "industry", "market_cap_value"}

# rank_factor -> callable(panel, asof, market_close, lookback, skip) -> Series
_RANK = {
    "momentum": lambda p, a, m, lb, sk: factors.momentum(p, a, lb, sk),
    "trailing_return": lambda p, a, m, lb, sk: factors.trailing_return(p, a, lb, sk),
    "calendar_return": lambda p, a, m, lb, sk: factors.calendar_return(p, (a.year - 1) if a is not None else p.index.year.max()),
    "low_volatility": lambda p, a, m, lb, sk: factors.low_volatility(p, a, lb),
    "volatility": lambda p, a, m, lb, sk: factors.volatility(p, a, lb),
    "beta": lambda p, a, m, lb, sk: factors.beta(p, a, m, lb),
    "low_beta": lambda p, a, m, lb, sk: factors.low_beta(p, a, m, lb),
    "max_drawdown": lambda p, a, m, lb, sk: factors.max_drawdown(p, a, lb),
}


@dataclass
class FilterRule:
    factor: str
    op: str            # "<=" | ">=" | "==" | "in"
    value: object


@dataclass
class SelectionSpec:
    universe: object = "Sector ETFs"   # named key or explicit list
    explicit: list | None = None       # explicit basket (overrides ranking)
    rank_factor: str | None = None
    rank_lookback: int = 252
    rank_skip: int = 0
    top_n: int | None = None           # longs from the top of the ranking
    bottom_n: int | None = None        # shorts from the bottom of the ranking
    filters: list = field(default_factory=list)


@dataclass
class ETFSpec:
    name: str
    selection: SelectionSpec
    weighting: str = "equal"
    rebalance: str = "Monthly"
    direction: str = "long"            # long | short | long_short
    manual_weights: dict | None = None
    max_weight: float = 1.0
    notes: str = ""
    phase: int = 1
    enabled: bool = True


def resolve_universe(u) -> list[str]:
    if isinstance(u, (list, tuple, set)):
        return list(u)
    if u in ("Stocks (500)", "IWB (large-cap proxy)", "IWB"):
        return list(STOCKS_500)
    return list(UNIVERSES.get(u, []))


def spec_is_lookahead(spec: ETFSpec) -> bool:
    """True if the spec relies on the fundamental snapshot (look-ahead/survivorship)."""
    if spec.weighting in weighting.SNAPSHOT_METHODS:
        return True
    if spec.selection.rank_factor in CLASS_B_FACTORS:
        return True
    return any(f.factor in CLASS_B_FACTORS for f in (spec.selection.filters or []))


def _valid_at(panel: pd.DataFrame, asof, tickers) -> list[str]:
    sub = panel.loc[:asof]
    if sub.empty:
        return []
    last = sub.iloc[-1]
    return [t for t in tickers if t in panel.columns and np.isfinite(last.get(t, np.nan))]


def selected_at(spec: ETFSpec, panel: pd.DataFrame, asof=None,
                market_close: pd.Series | None = None):
    """Return (longs, shorts) chosen from data available up to ``asof``."""
    sel = spec.selection
    if sel.explicit:
        return _valid_at(panel, asof, sel.explicit), []

    uni = [t for t in resolve_universe(sel.universe) if t in panel.columns]
    uni = _valid_at(panel, asof, uni)
    if not uni or not sel.rank_factor:
        return uni, []

    fn = _RANK.get(sel.rank_factor)
    if fn is None:
        return uni, []
    scores = fn(panel[uni], asof, market_close, sel.rank_lookback, sel.rank_skip)
    scores = scores.reindex(uni).dropna().sort_values(ascending=False)
    if scores.empty:
        return [], []

    longs, shorts = [], []
    if spec.direction in ("long", "long_short"):
        longs = list(scores.index[:sel.top_n]) if sel.top_n else list(scores.index)
    if spec.direction in ("short", "long_short"):
        n = sel.bottom_n or sel.top_n
        shorts = list(scores.index[-n:]) if n else []
    if spec.direction == "short" and not shorts:
        shorts = list(scores.index[-(sel.top_n or len(scores)):])
    return longs, shorts


def build_weight_schedule(spec: ETFSpec, panel: pd.DataFrame, rebal_dts,
                          market_close: pd.Series | None = None) -> pd.DataFrame:
    """Signed target weights per rebalance date (longs sum +1, shorts sum -1)."""
    rows = {}
    for asof in rebal_dts:
        longs, shorts = selected_at(spec, panel, asof, market_close)
        w = {}
        if longs:
            lw = weighting.compute(spec.weighting, longs, panel, asof,
                                   spec.manual_weights, spec.max_weight)
            for t, x in lw.items():
                w[t] = w.get(t, 0.0) + x
        if shorts:
            sw = weighting.compute(spec.weighting, shorts, panel, asof,
                                   None, spec.max_weight)
            for t, x in sw.items():
                w[t] = w.get(t, 0.0) - x
        rows[asof] = w
    if not rows:
        return pd.DataFrame(columns=panel.columns)
    return pd.DataFrame(rows).T.reindex(columns=panel.columns).fillna(0.0)
