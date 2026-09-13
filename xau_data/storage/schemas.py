"""Arrow schemas. Single source of truth for every table on disk.

Every timestamp column is ``timestamp[us, tz=UTC]``. A test walks these schemas
and fails if any timestamp field is naive, so a new column cannot be added
without a timezone.
"""
from __future__ import annotations

from typing import Final

import pyarrow as pa

__all__ = [
    "TS", "TICKS", "BARS", "BARS_M1", "CANDLE_DAY", "CONTEXT", "CALENDAR",
    "MANIFEST", "SCHEMAS",
    "TIMEFRAMES", "TIMEFRAME_SECONDS", "bars_schema", "timestamp_fields",
]

#: The only timestamp type used anywhere in this package.
TS: Final[pa.DataType] = pa.timestamp("us", tz="UTC")

TIMEFRAME_SECONDS: Final[dict[str, int]] = {
    "M1": 60, "M5": 300, "M15": 900, "H1": 3600,
}
TIMEFRAMES: Final[tuple[str, ...]] = tuple(TIMEFRAME_SECONDS)

TICKS: Final[pa.Schema] = pa.schema(
    [
        pa.field("ts", TS, nullable=False),
        pa.field("bid", pa.float64(), nullable=False),
        pa.field("ask", pa.float64(), nullable=False),
        pa.field("bid_volume", pa.float64(), nullable=False),
        pa.field("ask_volume", pa.float64(), nullable=False),
    ],
    metadata={
        b"description": b"XAUUSD ticks as reported by the feed, UTC.",
        b"spread": b"derived as ask - bid in the DuckDB view, never stored",
        b"volume_units": b"feed-native and opaque; not lots and not ounces",
    },
)

_OHLC = [
    (f"{side}_{part}")
    for side in ("bid", "ask", "mid")
    for part in ("open", "high", "low", "close")
]

BARS: Final[pa.Schema] = pa.schema(
    [
        pa.field("ts_open", TS, nullable=False),
        pa.field("ts_end_exclusive", TS, nullable=False),
        *[pa.field(n, pa.float64(), nullable=False) for n in _OHLC],
        pa.field("spread_mean", pa.float64(), nullable=False),
        pa.field("spread_max", pa.float64(), nullable=False),
        pa.field("spread_min", pa.float64(), nullable=False),
        pa.field("spread_close", pa.float64(), nullable=False),
        pa.field("tick_count", pa.int32(), nullable=False),
        pa.field("bid_volume", pa.float64(), nullable=False),
        pa.field("ask_volume", pa.float64(), nullable=False),
        pa.field("first_tick_ts", TS, nullable=False),
        pa.field("last_tick_ts", TS, nullable=False),
    ],
    metadata={
        b"ts_open": (
            b"BAR OPEN TIME in UTC. This is the bar's label. Intervals are "
            b"left-closed and right-open: a tick belongs to the bar iff "
            b"ts_open <= tick.ts < ts_end_exclusive."
        ),
        b"ts_end_exclusive": b"ts_open + timeframe period. EXCLUSIVE right edge.",
        b"mid": (
            b"mid is computed PER TICK as (bid+ask)/2 and then reduced to OHLC. "
            b"It is NOT (bid_high+ask_high)/2, which is wrong when the spread "
            b"widens at an extreme."
        ),
        b"gaps": b"Intervals with no ticks are ABSENT rows. Prices are never forward-filled.",
    },
)

CONTEXT: Final[pa.Schema] = pa.schema(
    [
        pa.field("series_id", pa.string(), nullable=False),
        pa.field("observation_time", TS, nullable=False),
        pa.field("publication_time", TS, nullable=False),
        pa.field("value", pa.float64(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_series_id", pa.string(), nullable=False),
        pa.field("vintage_seq", pa.int32(), nullable=False),
        pa.field("is_revision", pa.bool_(), nullable=False),
        pa.field("ingested_at", TS, nullable=False),
    ],
    metadata={
        b"join_key": (
            b"publication_time ONLY. observation_time must never be used as a "
            b"join key: it is what the value refers to, not when it was knowable."
        ),
        b"revisions": b"A revision is a new row with a later publication_time.",
    },
)

CALENDAR: Final[pa.Schema] = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("event_time", TS, nullable=False),
        pa.field("publication_time", TS, nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("country", pa.string(), nullable=True),
        pa.field("title", pa.string(), nullable=False),
        pa.field("impact", pa.string(), nullable=False),
        pa.field("actual", pa.float64(), nullable=True),
        pa.field("forecast", pa.float64(), nullable=True),
        pa.field("previous", pa.float64(), nullable=True),
        pa.field("actual_raw", pa.string(), nullable=True),
        pa.field("forecast_raw", pa.string(), nullable=True),
        pa.field("previous_raw", pa.string(), nullable=True),
        pa.field("unit", pa.string(), nullable=True),
        pa.field("snapshot_time", TS, nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ],
    metadata={
        b"actual": b"knowable only at event_time; forecast/previous knowable from snapshot_time",
    },
)

MANIFEST: Final[pa.Schema] = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("hour_utc", TS, nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("bytes", pa.int64(), nullable=False),
        pa.field("sha256", pa.string(), nullable=True),
        pa.field("tick_count", pa.int32(), nullable=False),
        pa.field("fetched_at", TS, nullable=False),
        pa.field("error", pa.string(), nullable=True),
    ],
    metadata={
        b"status": (
            b"ok = payload with ticks; empty = payload with zero ticks (market "
            b"closed); missing = server had no file (404); failed = transport "
            b"error. ok and empty are skipped on resume; missing and failed retry."
        ),
    },
)

#: Candle-derived M1 bars. Deliberately NOT the same shape as tick-derived
#: BARS: a daily candle file cannot supply intrabar spread statistics or tick
#: counts, and mid extremes are not recoverable because the bid high and the
#: ask high need not occur at the same instant. Only exact quantities appear.
BARS_M1: Final[pa.Schema] = pa.schema(
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
    ],
    metadata={
        b"source": b"Dukascopy BID/ASK_candles_min_1.bi5, one file per day per side",
        b"ts_open": b"BAR OPEN TIME in UTC, left-closed right-open, as for BARS",
        b"not_available": (
            b"spread_mean/max/min and tick_count cannot be derived from candles. "
            b"mid_high and mid_low are OMITTED rather than approximated: the bid "
            b"high and ask high need not be simultaneous, so (bid_high+ask_high)/2 "
            b"is not the mid high. Use bid_* or ask_* extremes explicitly."
        ),
        b"synthetic_stripped": (
            b"Minutes with volume==0 AND flat OHLC are Dukascopy forward-fill and "
            b"are REMOVED at ingest. Counts are recorded per day in the manifest."
        ),
    },
)

#: One row per ingested candle day: how many synthetic minutes were stripped.
CANDLE_DAY: Final[pa.Schema] = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("day_utc", TS, nullable=False),
        pa.field("records_bid", pa.int32(), nullable=False),
        pa.field("records_ask", pa.int32(), nullable=False),
        pa.field("stripped_synthetic", pa.int32(), nullable=False),
        pa.field("kept", pa.int32(), nullable=False),
        pa.field("rollover_hour_utc", pa.int32(), nullable=True),
        pa.field("ingested_at", TS, nullable=False),
    ],
    metadata={b"purpose": b"A sudden change in stripped_synthetic means something upstream shifted."},
)

SCHEMAS: Final[dict[str, pa.Schema]] = {
    "bars_m1": BARS_M1,
    "candle_day": CANDLE_DAY,
    "ticks": TICKS, "bars": BARS, "context": CONTEXT,
    "calendar": CALENDAR, "manifest": MANIFEST,
}


def bars_schema(timeframe: str) -> pa.Schema:
    if timeframe not in TIMEFRAME_SECONDS:
        raise KeyError(f"unknown timeframe {timeframe!r}; expected one of {TIMEFRAMES}")
    return BARS


def timestamp_fields(schema: pa.Schema) -> list[pa.Field]:
    return [f for f in schema if pa.types.is_timestamp(f.type)]
