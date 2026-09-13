"""Ticks to OHLCV bars.

Binning is integer arithmetic on the UTC epoch::

    bin_open_us = ts_us - (ts_us % period_us)

M1, M5, M15 and H1 all divide an hour evenly, so the grid is unambiguous and
DST cannot move a bar edge.

Bars are LEFT-CLOSED and RIGHT-OPEN. ``ts_open`` is the bar's label and is the
BAR OPEN time in UTC. A tick belongs to the bar iff
``ts_open <= tick.ts < ts_end_exclusive``.

Intervals with no ticks produce NO ROW. Prices are never forward-filled.
"""
from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..errors import SchemaError
from ..storage.schemas import BARS, TIMEFRAME_SECONDS
from ..timeutils import from_epoch_us

__all__ = ["ticks_to_bars", "bars_from_table"]

_US = 1_000_000


def ticks_to_bars(
    ts_us: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    bid_volume: np.ndarray,
    ask_volume: np.ndarray,
    *,
    timeframe: str,
) -> pa.Table:
    """Aggregate ticks into bars at ``timeframe``. Input must be sorted by ts."""
    if timeframe not in TIMEFRAME_SECONDS:
        raise KeyError(f"unknown timeframe {timeframe!r}")
    n = len(ts_us)
    if n == 0:
        return BARS.empty_table()
    if not (len(bid) == len(ask) == len(bid_volume) == len(ask_volume) == n):
        raise ValueError("tick arrays must be the same length")
    if np.any(np.diff(ts_us) < 0):
        raise ValueError("ticks must be sorted by timestamp")

    period_us = TIMEFRAME_SECONDS[timeframe] * _US
    key = ts_us - (ts_us % period_us)

    # group boundaries; ticks are sorted so groups are contiguous runs
    starts = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1))
    stops = np.concatenate((starts[1:], [n]))
    mid = (bid + ask) / 2.0
    spread = ask - bid

    def reduce(fn, arr):
        return np.array([fn(arr[a:b]) for a, b in zip(starts, stops)])

    counts = (stops - starts).astype(np.int32)
    bar_open_us = key[starts]

    cols: dict[str, object] = {
        "ts_open": pa.array([from_epoch_us(int(v)) for v in bar_open_us],
                            type=BARS.field("ts_open").type),
        "ts_end_exclusive": pa.array(
            [from_epoch_us(int(v) + period_us) for v in bar_open_us],
            type=BARS.field("ts_end_exclusive").type),
        "tick_count": pa.array(counts, type=pa.int32()),
        "first_tick_ts": pa.array([from_epoch_us(int(ts_us[a])) for a in starts],
                                  type=BARS.field("first_tick_ts").type),
        "last_tick_ts": pa.array([from_epoch_us(int(ts_us[b - 1])) for b in stops],
                                 type=BARS.field("last_tick_ts").type),
    }
    for name, arr in (("bid", bid), ("ask", ask), ("mid", mid)):
        cols[f"{name}_open"] = arr[starts]
        cols[f"{name}_high"] = reduce(np.max, arr)
        cols[f"{name}_low"] = reduce(np.min, arr)
        cols[f"{name}_close"] = arr[stops - 1]
    cols["spread_mean"] = reduce(np.mean, spread)
    cols["spread_max"] = reduce(np.max, spread)
    cols["spread_min"] = reduce(np.min, spread)
    cols["spread_close"] = spread[stops - 1]
    cols["bid_volume"] = reduce(np.sum, bid_volume)
    cols["ask_volume"] = reduce(np.sum, ask_volume)

    table = pa.table({k: (v if isinstance(v, pa.Array) else pa.array(v, type=pa.float64()))
                      for k, v in cols.items()})
    return table.select(BARS.names).cast(BARS)


def bars_from_table(ticks: pa.Table, *, timeframe: str) -> pa.Table:
    """Convenience wrapper over an Arrow ticks table."""
    for col in ("ts", "bid", "ask", "bid_volume", "ask_volume"):
        if col not in ticks.column_names:
            raise SchemaError(f"ticks table missing column {col!r}")
    ticks = ticks.sort_by([("ts", "ascending")])
    ts_us = ticks["ts"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    return ticks_to_bars(
        ts_us,
        ticks["bid"].to_numpy(zero_copy_only=False),
        ticks["ask"].to_numpy(zero_copy_only=False),
        ticks["bid_volume"].to_numpy(zero_copy_only=False),
        ticks["ask_volume"].to_numpy(zero_copy_only=False),
        timeframe=timeframe,
    )
