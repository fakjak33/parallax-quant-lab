"""Strategy base class.

A Strategy turns an OHLCV frame into a *continuous, scaled* Carver forecast
in roughly [-20, +20]. Single-instrument rules implement ``raw_forecast``;
the base ``forecast`` method scales and caps it. Cross-sectional rules
override ``forecast_panel`` instead.

Each subclass declares:
  - ``key``: unique short id used by the registry/UI
  - ``label``: human-friendly name
  - ``params``: {name: (default, lo, hi, step)} drives the UI spectrum sliders
  - ``spectrum_param``: which param the spectrum tester sweeps by default
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..core.forecasts import scale_forecast


@dataclass
class Strategy:
    key: str = ""
    label: str = ""
    params: dict = field(default_factory=dict)
    spectrum_param: str = ""
    cross_sectional: bool = False

    def __init__(self, **kwargs):
        # Instance params start from class defaults, overridden by kwargs.
        self.values = {name: spec[0] for name, spec in self.params.items()}
        for k, v in kwargs.items():
            if k not in self.values:
                raise KeyError(f"{self.key}: unknown param {k!r}")
            self.values[k] = v

    # --- single-instrument API --------------------------------------------
    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Return the unscaled forecast. Override in single-instrument rules."""
        raise NotImplementedError

    def forecast(self, ohlcv: pd.DataFrame, scalar: float | None = None) -> pd.Series:
        """Scaled, capped Carver forecast for one instrument."""
        raw = self.raw_forecast(ohlcv)
        return scale_forecast(raw, scalar=scalar)

    # --- cross-sectional API ----------------------------------------------
    def forecast_panel(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a wide DataFrame (date x ticker) of forecasts.

        Default: apply the single-instrument forecast to each asset. Cross
        sectional rules override this to rank across the universe.
        """
        cols = {tkr: self.forecast(df) for tkr, df in panel.items()}
        return pd.DataFrame(cols)

    def describe(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.values.items())
        return f"{self.label} ({ps})" if ps else self.label
