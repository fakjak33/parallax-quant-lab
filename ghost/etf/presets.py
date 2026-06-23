"""Ready-made example ETFs (the user's seven designs).

Phase-1 presets are fully backtest-valid (price-derived). Phase-2 presets need
the fundamental snapshot (FCF, margin, sector/industry) and are present but
disabled until that layer lands.
"""

from __future__ import annotations

from .screens import ETFSpec, SelectionSpec, FilterRule

PRESETS: list[ETFSpec] = [
    # Lightweight baskets first (fast to load) — universe screens follow.
    ETFSpec(
        name="Equal-weight all sectors",
        selection=SelectionSpec(universe="Sector ETFs"),
        weighting="equal", rebalance="Quarterly", direction="long",
        notes="Equal-weight basket of the sector SPDRs, rebalanced quarterly.",
    ),
    ETFSpec(
        name="QQQ / IEF / GLD / EFA (quarterly)",
        selection=SelectionSpec(explicit=["QQQ", "IEF", "GLD", "EFA"]),
        weighting="equal", rebalance="Quarterly", direction="long",
        notes="Classic multi-asset basket rebalanced quarterly.",
    ),
    ETFSpec(
        name="IWB Top-10 prior-calendar-year (annual)",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                rank_factor="calendar_return", top_n=10),
        weighting="equal", rebalance="Annual", direction="long",
        notes="Buy last calendar year's 10 best and hold the next year. Universe "
              "is a current large-cap proxy (survivorship caveat). First load "
              "fetches the universe — give it a moment.",
    ),
    ETFSpec(
        name="IWB Top-10 rolling 12-month momentum",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                rank_factor="trailing_return", rank_lookback=252,
                                top_n=10),
        weighting="equal", rebalance="Monthly", direction="long",
        notes="Top 10 by trailing 12-month return, rebalanced monthly. First load "
              "fetches the universe — give it a moment.",
    ),
    ETFSpec(
        name="Long top-5 beta / short bottom-5 beta (market-neutral)",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                rank_factor="beta", top_n=5, bottom_n=5),
        weighting="equal", rebalance="Monthly", direction="long_short",
        notes="Dollar-neutral high-minus-low beta. (The exact 'short least-"
              "profitable' leg needs the Phase-2 fundamental snapshot.)",
    ),
    # --- Phase 2 (need the fundamental snapshot) ---
    ETFSpec(
        name="Free-cash-flow / share ≤ 3 (value)",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                filters=[FilterRule("fcf_per_share", "<=", 3.0)]),
        weighting="equal", rebalance="Quarterly", direction="long",
        phase=2, enabled=False,
        notes="Phase 2: needs fundamental snapshot (FCF/share). Look-ahead/"
              "survivorship caveat applies.",
    ),
    ETFSpec(
        name="Insurance names with 5-yr dividend growth",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                rank_factor="dividend_growth",
                                filters=[FilterRule("industry", "==", "Insurance")]),
        weighting="equal", rebalance="Annual", direction="long",
        phase=2, enabled=False,
        notes="Phase 2: industry classification is a fundamental snapshot; "
              "dividend growth itself is price-history valid.",
    ),
]


def by_name(name: str) -> ETFSpec | None:
    return next((p for p in PRESETS if p.name == name), None)
