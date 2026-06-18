"""Real market data via yfinance with a local parquet cache.

A single ticker maps to one parquet file under ``data_cache/``. Repeated
backtests read from disk; pass ``force_refresh=True`` to re-download.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import DATA_CACHE

# Canonical OHLCV column order used throughout the lab.
OHLCV = ["open", "high", "low", "close", "volume"]

# Pandas resample rules for the UI timeframe selector.
TIMEFRAMES = {"Daily": "D", "Weekly": "W-FRI", "Monthly": "ME"}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV frame to a coarser timeframe (e.g. 'W-FRI', 'ME')."""
    if rule in (None, "D"):
        return df
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    use = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample(rule).agg(use).dropna(how="any")


def _cache_path(ticker: str) -> Path:
    return DATA_CACHE / f"{ticker.upper().replace('/', '_')}.parquet"


def _download(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    # When no explicit start is given, yfinance defaults to a 1-month window.
    # Use period="max" to cache full available history instead.
    kwargs = dict(auto_adjust=True, progress=False, threads=False)
    if start is None and end is None:
        kwargs["period"] = "max"
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    raw = yf.download(ticker, **kwargs)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker!r}.")

    # yfinance may return a MultiIndex (field, ticker) for single symbols too.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    cols = [c for c in OHLCV if c in raw.columns]
    out = raw[cols].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"
    return out


def get_prices(
    ticker: str,
    start: str | None = "2010-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return adjusted OHLCV for ``ticker`` as a DataFrame indexed by date.

    Cached to parquet. On a cache hit the full cached history is returned and
    then sliced to ``[start, end]`` so different date windows share one file.
    """
    path = _cache_path(ticker)

    if path.exists() and not force_refresh:
        df = pd.read_parquet(path)
    else:
        df = _download(ticker, start=None, end=None)  # cache full history
        df.to_parquet(path)

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def get_panel(
    tickers: list[str],
    start: str | None = "2010-01-01",
    end: str | None = None,
    field: str = "close",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a wide DataFrame of one ``field`` (default close) across tickers.

    Columns are tickers; rows are dates aligned on the union of trading days.
    Tickers that fail to download are skipped with a warning column omitted.
    """
    series: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    for t in tickers:
        try:
            df = get_prices(t, start=start, end=end, force_refresh=force_refresh)
            series[t.upper()] = df[field]
        except Exception as exc:  # noqa: BLE001 — surface per-ticker, keep going
            errors[t.upper()] = str(exc)
            time.sleep(0.1)

    if not series:
        raise ValueError(f"No tickers could be loaded. Errors: {errors}")

    panel = pd.DataFrame(series).sort_index()
    panel.attrs["errors"] = errors
    return panel
