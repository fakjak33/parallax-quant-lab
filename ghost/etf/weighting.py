"""Weighting schemes for a selected set of names.

``compute`` returns a ``ticker -> weight`` dict of NON-NEGATIVE weights summing
to 1.0 over the given names. The sign (long vs short) and cross-leg scaling for
long/short funds is applied by ``screens.build_weight_schedule``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import factors

METHODS = ("equal", "inverse_vol", "vol_target", "market_cap", "manual")
# weighting methods that rely on the fundamental snapshot (class B, caveated)
SNAPSHOT_METHODS = ("market_cap",)


def _cap_normalize(w: pd.Series, max_weight: float) -> pd.Series:
    """Cap each weight at ``max_weight`` and renormalise to sum 1 (iterative)."""
    w = w.clip(lower=0.0)
    if w.sum() <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    w = w / w.sum()
    if max_weight >= 1.0:
        return w
    for _ in range(50):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        under = ~over
        if not under.any() or w[under].sum() <= 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return w / w.sum()


def compute(method: str, tickers: list[str], panel: pd.DataFrame, asof=None,
            manual: dict | None = None, max_weight: float = 1.0,
            vol_lookback: int = 126) -> dict:
    tickers = [t for t in tickers if t in panel.columns]
    if not tickers:
        return {}
    if len(tickers) == 1:
        return {tickers[0]: 1.0}

    if method == "manual" and manual:
        w = pd.Series({t: float(manual.get(t, 0.0)) for t in tickers})
        if w.sum() <= 0:
            w = pd.Series(1.0, index=tickers)
    elif method in ("inverse_vol", "vol_target"):
        vol = factors.volatility(panel[tickers], asof, vol_lookback).reindex(tickers)
        inv = 1.0 / vol.replace(0.0, np.nan)
        w = inv.fillna(inv.median() if inv.notna().any() else 1.0)
    elif method == "market_cap":
        from . import fundamentals
        caps = fundamentals.market_caps(tickers).reindex(tickers)
        w = caps.fillna(caps.median() if caps.notna().any() else 1.0)
        if w.sum() <= 0:
            w = pd.Series(1.0, index=tickers)
    else:  # equal (default)
        w = pd.Series(1.0, index=tickers)

    w = _cap_normalize(w, max_weight)
    return {t: float(w[t]) for t in tickers}
