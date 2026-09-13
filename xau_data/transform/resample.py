"""Aggregate candle-derived M1 bars up to higher timeframes.

Binning is the same integer arithmetic on the UTC epoch used for ticks, so the
grid is identical and DST cannot move an edge. Aggregation is exact for bid and
ask extremes: the high of a window is the max of its M1 highs on that side.

Mid extremes are NOT produced. A mid high is not recoverable from per-side
candles because the bid high and the ask high need not be simultaneous. Use the
bid or ask series explicitly, or fall back to tick-derived bars where an exact
mid is required.

Gaps stay gaps. A window containing no M1 bars produces no row.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pyarrow as pa

from ..errors import SchemaError
from ..storage.schemas import BARS_M1, TIMEFRAME_SECONDS, TS
from ..timeutils import from_epoch_us

__all__ = ["RESAMPLED", "resample_m1", "bar_range"]

_US: Final[int] = 1_000_000

RESAMPLED: Final[pa.Schema] = pa.schema(
    [
        pa.field("ts_open", TS, nullable=False),
        pa.field("ts_end_exclusive", TS, nullable=False),
        pa.field("bid_open", pa.float64(), nullable=False),
        pa.field("bid_high", pa.float64(), nullable=False),
        pa.field("bid_low", pa.float64(), nullable=False),
        pa.field("bid_close", pa.float64(), nullable=False),
        pa.field("ask_open", pa.float64(), nullable=False),
        pa.field("ask_high", pa.float64(), nullable=False),
        pa.field("ask_low", pa.float64(), nullable=False),
        pa.field("ask_close", pa.float64(), nullable=False),
        pa.field("mid_open", pa.float64(), nullable=False),
        pa.field("mid_close", pa.float64(), nullable=False),
        pa.field("spread_open", pa.float64(), nullable=False),
        pa.field("spread_close", pa.float64(), nullable=False),
        pa.field("bid_volume", pa.float64(), nullable=False),
        pa.field("ask_volume", pa.float64(), nullable=False),
        pa.field("m1_count", pa.int32(), nullable=False),
    ],
    metadata={
        b"ts_open": b"BAR OPEN TIME, UTC, left-closed right-open",
        b"m1_count": b"how many M1 bars contributed; < period/60 means an internal gap",
        b"mid_extremes": b"deliberately absent; not recoverable from per-side candles",
    },
)


def resample_m1(bars_m1: pa.Table, *, timeframe: str) -> pa.Table:
    """Aggregate M1 candle bars to ``timeframe``. Input sorted by ts_open."""
    if timeframe not in TIMEFRAME_SECONDS:
        raise KeyError(f"unknown timeframe {timeframe!r}")
    for col in ("ts_open", "bid_open", "ask_open"):
        if col not in bars_m1.column_names:
            raise SchemaError(f"M1 table missing column {col!r}")
    if bars_m1.num_rows == 0:
        return RESAMPLED.empty_table()

    t = bars_m1.sort_by([("ts_open", "ascending")])
    ts_us = t["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    period_us = TIMEFRAME_SECONDS[timeframe] * _US
    key = ts_us - (ts_us % period_us)

    starts = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1))
    stops = np.concatenate((starts[1:], [len(ts_us)]))

    def col(name: str) -> np.ndarray:
        return t[name].to_numpy(zero_copy_only=False)

    def agg(arr: np.ndarray, fn) -> np.ndarray:
        return np.array([fn(arr[a:b]) for a, b in zip(starts, stops)])

    bid_h, bid_l = col("bid_high"), col("bid_low")
    ask_h, ask_l = col("ask_high"), col("ask_low")
    out: dict[str, object] = {
        "ts_open": pa.array([from_epoch_us(int(v)) for v in key[starts]], type=TS),
        "ts_end_exclusive": pa.array(
            [from_epoch_us(int(v) + period_us) for v in key[starts]], type=TS),
        "bid_open": col("bid_open")[starts],
        "bid_high": agg(bid_h, np.max),
        "bid_low": agg(bid_l, np.min),
        "bid_close": col("bid_close")[stops - 1],
        "ask_open": col("ask_open")[starts],
        "ask_high": agg(ask_h, np.max),
        "ask_low": agg(ask_l, np.min),
        "ask_close": col("ask_close")[stops - 1],
        "mid_open": col("mid_open")[starts],
        "mid_close": col("mid_close")[stops - 1],
        "spread_open": col("spread_open")[starts],
        "spread_close": col("spread_close")[stops - 1],
        "bid_volume": agg(col("bid_volume"), np.sum),
        "ask_volume": agg(col("ask_volume"), np.sum),
        "m1_count": pa.array((stops - starts).astype(np.int32), type=pa.int32()),
    }
    tbl = pa.table({k: (v if isinstance(v, pa.Array) else pa.array(v, type=pa.float64()))
                    for k, v in out.items()})
    return tbl.select(RESAMPLED.names).cast(RESAMPLED)


def bar_range(bars: pa.Table, side: str = "bid") -> np.ndarray:
    """High minus low on one side. Exact, unlike a mid range."""
    if side not in ("bid", "ask"):
        raise ValueError("side must be 'bid' or 'ask'; mid extremes are not exact")
    return (bars[f"{side}_high"].to_numpy(zero_copy_only=False)
            - bars[f"{side}_low"].to_numpy(zero_copy_only=False))
