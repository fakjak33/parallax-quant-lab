"""Fundamental SNAPSHOT data (class B) — current values only, no history.

yfinance exposes today's market cap, P/E, margins, sector, etc., but not a
point-in-time history. So anything here is a *snapshot* held constant across a
backtest → look-ahead/survivorship bias. Phase 1 uses only ``market_caps`` (for
optional market-cap weighting, clearly flagged); the richer ``snapshot`` powers
the Phase 2 fundamental screens. Everything degrades to NaN and never raises.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import DATA_CACHE

_CACHE = DATA_CACHE / "_fundamentals.parquet"
_FIELDS = ("market_cap", "trailing_pe", "forward_pe", "price_to_book",
           "profit_margin", "dividend_yield", "beta", "shares")


def _load_cache() -> pd.DataFrame:
    if _CACHE.exists():
        try:
            return pd.read_parquet(_CACHE)
        except Exception:
            pass
    return pd.DataFrame(columns=list(_FIELDS) + ["fetched_at"])


def _save_cache(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_CACHE)
    except Exception:
        pass


def _fetch_one(ticker: str) -> dict:
    """Best-effort single-ticker snapshot via yfinance fast_info (fast) + info."""
    import yfinance as yf
    rec = {f: np.nan for f in _FIELDS}
    try:
        fi = yf.Ticker(ticker).fast_info
        rec["market_cap"] = float(fi.get("market_cap")) if fi.get("market_cap") else np.nan
        rec["shares"] = float(fi.get("shares")) if fi.get("shares") else np.nan
    except Exception:
        pass
    return rec


def snapshot(tickers, max_age_days: int = 30, fields=None) -> pd.DataFrame:
    """Return a snapshot DataFrame (tickers x fields), served from cache when
    fresh enough, otherwise fetched (slow, rate-limited) and cached."""
    tickers = [t.upper() for t in tickers]
    cache = _load_cache()
    now = time.time()
    fresh_cut = now - max_age_days * 86400
    out = {}
    to_fetch = []
    for t in tickers:
        if t in cache.index and float(cache.loc[t].get("fetched_at", 0) or 0) >= fresh_cut:
            out[t] = cache.loc[t].to_dict()
        else:
            to_fetch.append(t)
    for t in to_fetch:
        rec = _fetch_one(t)
        rec["fetched_at"] = now
        out[t] = rec
        cache.loc[t] = rec
    if to_fetch:
        _save_cache(cache)
    df = pd.DataFrame(out).T
    cols = list(fields) if fields else list(_FIELDS)
    return df.reindex(columns=cols + ["fetched_at"])


def market_caps(tickers) -> pd.Series:
    """Snapshot market caps (NaN where unavailable). Used for cap weighting."""
    try:
        df = snapshot(tickers, fields=["market_cap"])
        return pd.to_numeric(df["market_cap"], errors="coerce")
    except Exception:
        return pd.Series(np.nan, index=[t.upper() for t in tickers])
