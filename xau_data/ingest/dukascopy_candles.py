"""Dukascopy M1 candle ingest.

One LZMA file per day per side::

    /{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/BID_candles_min_1.bi5
    /{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/ASK_candles_min_1.bi5

Month is zero-indexed, as for ticks. Each file decompresses to 1,440 records of
24 bytes, one per minute of the UTC day.

Record layout, big-endian ``>5if``::

    int32   seconds from the start of the UTC day
    int32   open
    int32   close      <-- note the order
    int32   low
    int32   high
    float32 volume

The field order is open, CLOSE, LOW, HIGH. Decoding it as the conventional
open/high/low/close produces OHLC constraint violations on roughly 90% of bars;
this is verified by a test rather than trusted.

Non-trading minutes are present but forward-filled by the vendor, carrying the
last price with ``volume == 0`` and a flat OHLC. That is the forward-fill this
package forbids, so those rows are STRIPPED at ingest and counted per day.
"""
from __future__ import annotations

import lzma
import struct
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import pyarrow as pa

from ..config import Config
from ..errors import DecodeError, FetchError, PolicyDeniedError
from ..storage.schemas import BARS_M1, CANDLE_DAY
from ..timeutils import UTC, require_utc
from ._http import HttpFetcher

__all__ = [
    "candle_url", "decode_candles", "Candle", "is_synthetic",
    "merge_sides", "DayResult", "ingest_days",
]

_RECORD = struct.Struct(">5if")
_RECORD_SIZE = 24
_MINUTES_PER_DAY = 1440
SIDES = ("BID", "ASK")


def candle_url(
    base_url: str, symbol: str, day: date, side: str, *, month_zero_indexed: bool = True
) -> str:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    month = day.month - 1 if month_zero_indexed else day.month
    return (f"{base_url.rstrip('/')}/{symbol}/{day.year:04d}/{month:02d}/"
            f"{day.day:02d}/{side}_candles_min_1.bi5")


@dataclass(frozen=True, slots=True)
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def flat(self) -> bool:
        return self.open == self.high == self.low == self.close


def is_synthetic(c: Candle) -> bool:
    """Vendor forward-fill: zero volume AND a flat bar.

    Both conditions are required. Verified lossless against tick-derived bars:
    no real minute satisfies either condition in the samples checked, and every
    synthetic minute satisfies both.
    """
    return c.volume == 0.0 and c.flat


def decode_candles(
    payload: bytes, day: date, *, point_digits: int
) -> list[Candle]:
    """Decode one day-side file. Returns ALL minutes, synthetic included."""
    if not payload:
        return []
    try:
        body = lzma.decompress(payload, format=lzma.FORMAT_AUTO)
    except lzma.LZMAError as exc:
        raise DecodeError(f"{day}: not a valid LZMA stream: {exc}") from exc
    if len(body) % _RECORD_SIZE:
        raise DecodeError(
            f"{day}: {len(body)} bytes is not a multiple of the "
            f"{_RECORD_SIZE}-byte candle record"
        )
    scale = 10 ** point_digits
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    out: list[Candle] = []
    for off in range(0, len(body), _RECORD_SIZE):
        secs, o, c, lo, hi, vol = _RECORD.unpack_from(body, off)
        out.append(Candle(start + timedelta(seconds=secs),
                          o / scale, hi / scale, lo / scale, c / scale, float(vol)))
    return out


@dataclass(slots=True)
class DayResult:
    day: date
    records_bid: int = 0
    records_ask: int = 0
    stripped: int = 0
    kept: int = 0
    status: str = "ok"
    error: str | None = None


def merge_sides(
    bid: Sequence[Candle], ask: Sequence[Candle]
) -> tuple[list[dict[str, object]], int]:
    """Join the two sides on timestamp, dropping synthetic minutes.

    A minute survives only if BOTH sides are present and NEITHER is synthetic.
    Returns (rows, stripped_count).
    """
    b = {c.ts: c for c in bid}
    a = {c.ts: c for c in ask}
    rows: list[dict[str, object]] = []
    stripped = 0
    for ts in sorted(b.keys() & a.keys()):
        cb, ca = b[ts], a[ts]
        if is_synthetic(cb) or is_synthetic(ca):
            stripped += 1
            continue
        rows.append({
            "ts_open": ts,
            "ts_end_exclusive": ts + timedelta(minutes=1),
            "bid_open": cb.open, "bid_high": cb.high,
            "bid_low": cb.low, "bid_close": cb.close,
            "ask_open": ca.open, "ask_high": ca.high,
            "ask_low": ca.low, "ask_close": ca.close,
            "mid_open": (cb.open + ca.open) / 2.0,
            "mid_close": (cb.close + ca.close) / 2.0,
            "spread_open": ca.open - cb.open,
            "spread_close": ca.close - cb.close,
            "bid_volume": cb.volume, "ask_volume": ca.volume,
        })
    # minutes present on only one side are also dropped, and counted as stripped
    stripped += len(b.keys() ^ a.keys())
    return rows, stripped


def rows_to_table(rows: Sequence[Mapping[str, object]]) -> pa.Table:
    if not rows:
        return BARS_M1.empty_table()
    cols: dict[str, pa.Array] = {}
    for f in BARS_M1:
        vals = [r[f.name] for r in rows]
        cols[f.name] = pa.array(vals, type=f.type)
    return pa.table(cols).select(BARS_M1.names).cast(BARS_M1)


def day_record(
    symbol: str, res: DayResult, rollover_hour: int | None
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "day_utc": datetime(res.day.year, res.day.month, res.day.day, tzinfo=UTC),
        "records_bid": res.records_bid,
        "records_ask": res.records_ask,
        "stripped_synthetic": res.stripped,
        "kept": res.kept,
        "rollover_hour_utc": rollover_hour,
        "ingested_at": datetime.now(UTC),
    }


def day_table(records: Sequence[Mapping[str, object]]) -> pa.Table:
    if not records:
        return CANDLE_DAY.empty_table()
    cols: dict[str, pa.Array] = {}
    for f in CANDLE_DAY:
        cols[f.name] = pa.array([r[f.name] for r in records], type=f.type)
    return pa.table(cols).select(CANDLE_DAY.names).cast(CANDLE_DAY)


def trading_days(start: date, end: date, *, skip_saturdays: bool = True) -> Iterator[date]:
    """Days to fetch in [start, end]. Saturdays are pure vendor fill."""
    d = start
    while d <= end:
        if not (skip_saturdays and d.weekday() == 5):
            yield d
        d += timedelta(days=1)


def _raw_path(cfg: Config, day: date, side: str) -> Path:
    return (cfg.data_root / "raw" / "dukascopy_candles" / cfg.symbol /
            f"{day.year:04d}" / f"{day.month:02d}" /
            f"{day.day:02d}_{side}_min_1.bi5")


@dataclass(slots=True)
class CandleIngestSummary:
    days_considered: int = 0
    days_written: int = 0
    days_skipped: int = 0
    days_failed: int = 0
    rows: int = 0
    stripped: int = 0
    failures: list[str] = field(default_factory=list)
    per_day: list[dict[str, object]] = field(default_factory=list)


def ingest_days(
    cfg: Config,
    start: date,
    end: date,
    *,
    store,                       # storage.store.Store
    fetcher: HttpFetcher | None = None,
    skip_saturdays: bool = True,
    progress_every: int = 25,
    flush_every_days: int = 30,
) -> CandleIngestSummary:
    """Fetch, strip and persist M1 candle bars for [start, end].

    Resume is decided by the STORE, not by local disk: a month already present
    in the destination is not refetched.
    """
    from ..transform.sessions import SessionCalendar

    dk = cfg.dukascopy
    cal = SessionCalendar(cfg.sessions)
    s = CandleIngestSummary()
    own = fetcher is None
    if own:
        fetcher = HttpFetcher(
            requests_per_second=dk.requests_per_second,
            max_attempts=dk.max_attempts,
            backoff_base_seconds=dk.backoff_base_seconds,
            backoff_max_seconds=dk.backoff_max_seconds,
            connect_timeout_seconds=dk.connect_timeout_seconds,
            read_timeout_seconds=dk.read_timeout_seconds,
            max_connections=dk.max_concurrency,
        )
    assert fetcher is not None

    done_days = store.completed_days(cfg.symbol)
    buf: dict[tuple[int, int], list[dict[str, object]]] = {}
    day_rows: list[dict[str, object]] = []

    def flush() -> None:
        for (y, m), rows in sorted(buf.items()):
            if rows:
                store.append_bars_m1(cfg.symbol, y, m, rows_to_table(rows))
                s.days_written += len({r["ts_open"].date() for r in rows})  # type: ignore[union-attr]
        buf.clear()
        if day_rows:
            store.append_candle_days(cfg.symbol, day_table(day_rows))
            day_rows.clear()

    try:
        for i, d in enumerate(trading_days(start, end, skip_saturdays=skip_saturdays), 1):
            s.days_considered += 1
            if d in done_days:
                s.days_skipped += 1
                continue
            res = DayResult(day=d)
            sides: dict[str, list[Candle]] = {}
            for side in SIDES:
                cached = _raw_path(cfg, d, side)
                payload: bytes | None = None
                if cached.exists():
                    payload = cached.read_bytes()
                else:
                    try:
                        r = fetcher.get(candle_url(dk.base_url, cfg.symbol, d, side,
                                                   month_zero_indexed=dk.month_zero_indexed))
                    except PolicyDeniedError:
                        raise
                    except FetchError as exc:
                        res.status, res.error = "failed", str(exc)[:200]
                        break
                    payload = b"" if r.status == 404 else r.content
                    if dk.keep_raw:
                        cached.parent.mkdir(parents=True, exist_ok=True)
                        tmp = cached.with_suffix(".tmp")
                        tmp.write_bytes(payload)
                        tmp.replace(cached)
                sides[side] = decode_candles(payload, d, point_digits=cfg.point_digits)
            if res.status == "failed":
                s.days_failed += 1
                s.failures.append(f"{d}: {res.error}")
                continue

            bid, ask = sides.get("BID", []), sides.get("ASK", [])
            res.records_bid, res.records_ask = len(bid), len(ask)
            rows, stripped = merge_sides(bid, ask)
            res.stripped, res.kept = stripped, len(rows)
            s.rows += len(rows)
            s.stripped += stripped
            if rows:
                buf.setdefault((d.year, d.month), []).extend(rows)
            roll = cal.rollover_utc(d).hour if rows else None
            day_rows.append(day_record(cfg.symbol, res, roll))
            s.per_day.append({"day": d.isoformat(), "kept": res.kept,
                              "stripped": res.stripped})

            if len({k for k in buf}) and i % flush_every_days == 0:
                flush()
            if progress_every and i % progress_every == 0:
                print(f"  {i} days: rows={s.rows:,} stripped={s.stripped:,} "
                      f"failed={s.days_failed}", flush=True)
        flush()
    finally:
        if own:
            fetcher.close()
    return s
