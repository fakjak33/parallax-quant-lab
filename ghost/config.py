"""Global configuration: defaults, theme constants, and tunable parameters.

Universe presets live in ``ghost.data.universe``; this module holds the
numeric defaults for the Carver pipeline, costs, and the Ghost-in-the-Shell
color palette used by the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = PROJECT_ROOT / "data_cache"
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_CACHE.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# --- Carver framework defaults --------------------------------------------
TRADING_DAYS = 256          # Carver's business-day convention
FORECAST_SCALAR_TARGET = 10.0   # target average absolute forecast
FORECAST_CAP = 20.0             # hard cap on |forecast|
VOL_LOOKBACK = 32               # span (days) for EW volatility estimate
DEFAULT_TARGET_VOL = 0.20       # 20% annualized portfolio vol target
DEFAULT_CAPITAL = 1_000_000.0


@dataclass
class BacktestConfig:
    """Backtest-level knobs (costs, capital, vol target)."""

    capital: float = DEFAULT_CAPITAL
    target_vol: float = DEFAULT_TARGET_VOL
    cost_bps: float = 1.0           # per-trade cost in basis points of notional
    slippage_bps: float = 0.5       # half-spread slippage in bps
    vol_lookback: int = VOL_LOOKBACK
    trading_days: int = TRADING_DAYS


@dataclass
class RiskConfig:
    """ATR-based take-profit / stop-loss overlay settings."""

    use_atr_stop: bool = False
    use_atr_tp: bool = False
    atr_period: int = 14
    atr_stop_mult: float = 3.0      # k in entry - k*ATR
    atr_tp_mult: float = 6.0        # k in entry + k*ATR
    trailing_stop: bool = True


# --- Ghost in the Shell theme ---------------------------------------------
@dataclass(frozen=True)
class Theme:
    bg: str = "#05080a"
    panel: str = "#0a1014"
    grid: str = "#10242b"
    teal: str = "#23e0d0"
    cyan: str = "#4ff5ff"
    amber: str = "#ffb347"
    magenta: str = "#ff4f87"
    text: str = "#cfeef0"
    muted: str = "#5c7a80"
    font_mono: str = "'JetBrains Mono', 'Share Tech Mono', 'Consolas', monospace"
    # ordered palette for multi-series plots
    series: tuple = (
        "#23e0d0", "#ffb347", "#4ff5ff", "#ff4f87",
        "#9d7bff", "#7dff9b", "#ff8c42", "#5ab0ff",
    )


THEME = Theme()
