"""Apply ATR take-profit / stop-loss on top of a continuous position.

The Carver position is continuous, but TP/SL are discrete events. We walk the
series: when a new position of a given sign opens, record the entry price and
an ATR-derived stop and target. While the stop/target is active we zero the
position after a breach until the underlying signal flips or flattens, then
re-arm. Returns the gated position plus an event log for the blotter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RiskConfig
from .atr import atr


def apply_atr_overlay(
    position: pd.Series,
    ohlcv: pd.DataFrame,
    cfg: RiskConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Gate ``position`` by ATR stop-loss / take-profit.

    Returns (gated_position, events) where events has columns
    [date, type, price] with type in {entry, stop, take_profit}.
    """
    if not (cfg.use_atr_stop or cfg.use_atr_tp):
        return position, pd.DataFrame(columns=["date", "type", "price"])

    close = ohlcv["close"].reindex(position.index)
    atr_series = atr(ohlcv, cfg.atr_period).reindex(position.index).bfill()

    pos = position.to_numpy(dtype=float)
    px = close.to_numpy(dtype=float)
    av = atr_series.to_numpy(dtype=float)
    out = pos.copy()

    events: list[dict] = []
    in_trade = False
    sign = 0
    entry_px = stop_px = tp_px = np.nan
    suppressed = False  # True after a breach until signal resets

    for i in range(len(pos)):
        cur_sign = int(np.sign(pos[i]))

        # flat signal -> reset everything
        if cur_sign == 0:
            in_trade = False
            suppressed = False
            sign = 0
            continue

        # new trade or direction flip -> (re)arm levels
        if not in_trade or cur_sign != sign:
            in_trade = True
            suppressed = False
            sign = cur_sign
            entry_px = px[i]
            stop_px = entry_px - sign * cfg.atr_stop_mult * av[i]
            tp_px = entry_px + sign * cfg.atr_tp_mult * av[i]
            events.append({"date": position.index[i], "type": "entry", "price": entry_px})

        if suppressed:
            out[i] = 0.0
            continue

        # trailing stop ratchets in the trade's favor
        if cfg.use_atr_stop and cfg.trailing_stop:
            trail = px[i] - sign * cfg.atr_stop_mult * av[i]
            stop_px = trail if sign > 0 else trail
            stop_px = max(stop_px, entry_px - sign * cfg.atr_stop_mult * av[i]) if sign > 0 \
                else min(stop_px, entry_px - sign * cfg.atr_stop_mult * av[i])

        hit_stop = cfg.use_atr_stop and (
            (sign > 0 and px[i] <= stop_px) or (sign < 0 and px[i] >= stop_px)
        )
        hit_tp = cfg.use_atr_tp and (
            (sign > 0 and px[i] >= tp_px) or (sign < 0 and px[i] <= tp_px)
        )

        if hit_stop:
            out[i] = 0.0
            suppressed = True
            events.append({"date": position.index[i], "type": "stop", "price": px[i]})
        elif hit_tp:
            out[i] = 0.0
            suppressed = True
            events.append({"date": position.index[i], "type": "take_profit", "price": px[i]})

    gated = pd.Series(out, index=position.index)
    ev_df = pd.DataFrame(events) if events else pd.DataFrame(columns=["date", "type", "price"])
    return gated, ev_df
