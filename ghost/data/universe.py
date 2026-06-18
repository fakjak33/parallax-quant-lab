"""Named ETF/stock universes for seeding the lab.

These are starting points — the UI lets you type any ticker yfinance supports.
Grouped to mirror the kinds of cross-sectional baskets a Carver-style system
diversifies across.
"""

from __future__ import annotations

UNIVERSES: dict[str, list[str]] = {
    "Broad ETFs": ["SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "PDP", "EFA", "EEM"],
    "Sector ETFs": ["XLP", "XLU", "XLK", "VGT", "XLI", "XLRE", "XLE", "XLF", "XLV", "XLY", "XLB"],
    "Industry ETFs": ["IAK", "XBI", "SOXX", "KRE", "UFO", "IHAK", "IHI", "ITA", "IGV"],
    "Futures-like ETFs": ["PALL", "SLV", "GLD", "USO", "UNG", "UUP", "DBA", "DBC", "CPER"],
    "Mega-cap stocks": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"],
}

# Default benchmark for equity-curve overlays.
BENCHMARK = "SPY"


def all_tickers() -> list[str]:
    """Flattened, de-duplicated list of every ticker across all universes."""
    seen: dict[str, None] = {}
    for tickers in UNIVERSES.values():
        for t in tickers:
            seen.setdefault(t, None)
    return list(seen)
