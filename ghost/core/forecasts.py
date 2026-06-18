"""Forecast scaling and capping — the heart of the Carver framework.

A 'raw' forecast from a trading rule has arbitrary units. We:
  1. normalize it by its own rolling volatility (so its scale is stable),
  2. multiply by a forecast scalar so the long-run average |forecast| ~= 10,
  3. cap at +/- 20 to limit the influence of any single extreme reading.

This continuous, comparable forecast is what makes many rules combinable and
is a core anti-overfitting mechanism: no rule can dominate via raw magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FORECAST_SCALAR_TARGET, FORECAST_CAP


def scale_forecast(
    raw: pd.Series,
    scalar: float | None = None,
    cap: float = FORECAST_CAP,
) -> pd.Series:
    """Scale a raw forecast to target-average magnitude and cap it.

    If ``scalar`` is None it is estimated from the data so that the mean
    absolute *scaled* forecast equals FORECAST_SCALAR_TARGET (~10). Pass a
    fixed scalar (e.g. fitted in-sample) to avoid look-ahead in live use.
    """
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if scalar is None:
        scalar = estimate_forecast_scalar(raw)
    scaled = raw * scalar
    return scaled.clip(-cap, cap)


def estimate_forecast_scalar(raw: pd.Series, target: float = FORECAST_SCALAR_TARGET) -> float:
    """Scalar that maps mean(|raw|) to ``target`` (Carver's forecast scalar)."""
    mean_abs = raw.abs().mean()
    if not np.isfinite(mean_abs) or mean_abs < 1e-12:
        return 1.0
    return float(target / mean_abs)


def normalize_by_vol(signal: pd.Series, vol: pd.Series) -> pd.Series:
    """Divide a raw signal (e.g. a price-difference) by a volatility series."""
    out = signal / vol.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
