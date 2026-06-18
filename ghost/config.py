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

# Trading days represented by one bar at each timeframe — used to scale
# day-based strategy parameters when the candle frequency changes.
TIMEFRAME_DAYS = {"Daily": 1, "Weekly": 5, "Monthly": 21}
# Strategy parameters expressed in (trading-day) time that should be rescaled
# to bars when the timeframe changes.
TIME_SCALED_PARAMS = {"fast", "slow", "lookback", "smooth", "skip"}


@dataclass
class BacktestConfig:
    """Backtest-level knobs (costs, capital, vol target)."""

    capital: float = DEFAULT_CAPITAL
    target_vol: float = DEFAULT_TARGET_VOL
    cost_bps: float = 1.0           # per-trade cost in basis points of notional
    slippage_bps: float = 0.5       # half-spread slippage in bps
    vol_lookback: int = VOL_LOOKBACK
    trading_days: int = TRADING_DAYS
    use_vol_target: bool = True     # back-compat flag (see sizing_mode)
    direction: str = "both"         # "both" | "long" | "short"
    # sizing_mode: "vol_target" | "fixed_pct" | "fixed_dollar"
    sizing_mode: str = "vol_target"
    fixed_pct: float = 1.0          # fraction of capital deployed at full forecast
    fixed_dollar: float = DEFAULT_CAPITAL  # $ notional at full forecast

    def __post_init__(self):
        # keep the legacy use_vol_target flag in sync with sizing_mode
        if not self.use_vol_target and self.sizing_mode == "vol_target":
            self.sizing_mode = "fixed_pct"
        if self.sizing_mode != "vol_target":
            self.use_vol_target = False


@dataclass
class RiskConfig:
    """Stop-loss / take-profit overlay settings (ATR-based or fixed-percent)."""

    # ATR-based stop/TP
    use_atr_stop: bool = False
    use_atr_tp: bool = False
    atr_period: int = 14
    atr_stop_mult: float = 3.0      # k in entry - k*ATR
    atr_tp_mult: float = 6.0        # k in entry + k*ATR
    # fixed-percent stop/TP (alternative to ATR)
    use_pct_stop: bool = False
    use_pct_tp: bool = False
    pct_stop: float = 0.10          # 10% adverse move
    pct_tp: float = 0.20            # 20% favorable move
    # trailing applies to whichever stop is active
    trailing_stop: bool = True


# --- Parallax theme: black base + retro pastel/pantone accents ------------
# Palette sampled from the "Retro Color Palette" reference: navy, teal, mauve,
# mint, coral, orange, mustard, cream. Modernist & minimalist on near-black.
@dataclass(frozen=True)
class Theme:
    bg: str = "#000000"             # pure black base
    panel: str = "#0b0b0b"          # raised panel (near-black)
    grid: str = "#2c2c2c"           # subtle gridlines
    border: str = "#ffffff"         # brutalist hard borders
    teal: str = "#2ec4b6"           # primary accent (brighter)
    coral: str = "#ff5a3c"          # long emphasis (brighter)
    orange: str = "#ff8c2b"
    mustard: str = "#ffc857"
    mauve: str = "#c46b8b"
    mint: str = "#7bdcb5"
    navy: str = "#2a9bc4"
    cream: str = "#ffffff"
    text: str = "#ffffff"           # pure white text
    muted: str = "#b8b8b8"          # brighter muted
    # vintage grainy monospace "code" font across the whole app
    font_display: str = "'VT323', 'Share Tech Mono', 'Courier New', monospace"
    font_body: str = "'Share Tech Mono', 'VT323', 'Courier New', monospace"
    # accent roles
    long_color: str = "#2ec4b6"     # teal-green for longs
    short_color: str = "#ff5a3c"    # coral-red for shorts
    # ordered palette for multi-series plots (retro pantone set, brightened)
    series: tuple = (
        "#2ec4b6", "#ff5a3c", "#ffc857", "#7bdcb5",
        "#c46b8b", "#ff8c2b", "#2a9bc4", "#ffffff",
    )
    # section accents — cycle different colors across UI sections
    section_colors: tuple = (
        "#2ec4b6", "#ffc857", "#ff5a3c", "#c46b8b", "#2a9bc4", "#7bdcb5",
    )


THEME = Theme()
