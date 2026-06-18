"""Spectrum testing — run a whole family of parameter variants at once.

This is the user's core requirement: instead of picking one 'best' lookback,
test a continuum and look for a *plateau* of good performance (robust) versus
an isolated *spike* (overfit). Returns per-variant results plus a tidy frame
for heatmaps and the deflated-Sharpe penalty using the number of variants as
the trial count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BacktestConfig, RiskConfig
from .engine import run_strategy, BacktestResult


def make_spectrum(values_lo, values_hi, n: int, integer: bool = True) -> list:
    """Geometric spread of parameter values (denser at fast end, Carver-style)."""
    vals = np.geomspace(max(values_lo, 1e-6), values_hi, n)
    if integer:
        vals = np.unique(np.round(vals).astype(int))
    return list(vals)


def run_spectrum(
    strategy_cls,
    ohlcv: pd.DataFrame,
    param: str | None = None,
    values: list | None = None,
    bt: BacktestConfig | None = None,
    risk: RiskConfig | None = None,
    fixed_params: dict | None = None,
) -> tuple[dict[str, BacktestResult], pd.DataFrame]:
    """Backtest ``strategy_cls`` across a range of one parameter.

    Returns (results_by_label, table) where table has one row per variant with
    the swept value and key metrics (incl. deflated Sharpe over all variants).
    """
    param = param or strategy_cls.spectrum_param
    fixed_params = fixed_params or {}

    if values is None:
        default, lo, hi, _step = strategy_cls.params[param]
        values = make_spectrum(lo, hi, n=12, integer=isinstance(default, int))

    n_trials = len(values)
    results: dict[str, BacktestResult] = {}
    rows: list[dict] = []

    for v in values:
        strat = strategy_cls(**{**fixed_params, param: v})
        res = run_strategy(strat, ohlcv, bt=bt, risk=risk, n_trials=n_trials)
        results[res.label] = res
        rows.append({param: v, **res.stats})

    table = pd.DataFrame(rows).set_index(param)
    return results, table
