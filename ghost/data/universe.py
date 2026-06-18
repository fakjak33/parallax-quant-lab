"""Named ETF/stock universes for seeding the lab.

These are starting points — the UI lets you type any ticker yfinance supports.
Grouped to mirror the kinds of cross-sectional baskets a Carver-style system
diversifies across. A large categorized ETF master list is also loaded from
``etf_master_list.csv`` via :func:`load_master_list`.
"""

from __future__ import annotations

import csv
from pathlib import Path

_MASTER_CSV = Path(__file__).with_name("etf_master_list.csv")

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


def _is_category_header(ticker: str, name: str) -> bool:
    """A CSV row is a section header when it has no name and the 'ticker'
    cell is descriptive text (contains a space/dash/paren or is long)."""
    if name.strip():
        return False
    t = ticker.strip()
    return bool(t) and (len(t) > 5 or any(c in t for c in " -(/"))


def load_master_list() -> dict[str, list[tuple[str, str]]]:
    """Parse the bundled ETF master list into {category: [(ticker, name), ...]}.

    Section-header rows (e.g. 'SPDR - SECTOR') start a new category; following
    rows are its tickers. Tickers without a name keep the ticker as the label.
    Cached on the function object after first read.
    """
    cached = getattr(load_master_list, "_cache", None)
    if cached is not None:
        return cached

    groups: dict[str, list[tuple[str, str]]] = {}
    current = "Uncategorized"
    if _MASTER_CSV.exists():
        with _MASTER_CSV.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                ticker = row[0].strip()
                name = (row[1].strip() if len(row) > 1 else "")
                if ticker.lower() == "ticker":
                    continue
                if _is_category_header(ticker, name):
                    current = ticker.title()
                    groups.setdefault(current, [])
                else:
                    groups.setdefault(current, []).append((ticker.upper(), name or ticker.upper()))
    groups = {k: v for k, v in groups.items() if v}
    load_master_list._cache = groups  # type: ignore[attr-defined]
    return groups


def master_categories() -> list[str]:
    return list(load_master_list())


def master_tickers(category: str) -> list[str]:
    return [t for t, _ in load_master_list().get(category, [])]
