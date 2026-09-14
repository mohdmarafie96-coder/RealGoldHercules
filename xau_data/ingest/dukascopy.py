"""Dukascopy tick ingest: download, decode, persist, resume.

Wire format. Each hourly ``.bi5`` is an LZMA stream of fixed 20-byte records,
big-endian::

    uint32  milliseconds since the start of the hour
    uint32  ASK, in integer points
    uint32  BID, in integer points
    float32 ask volume
    float32 bid volume

Ask comes BEFORE bid. Getting that backwards produces a silently negated spread
and is covered by a byte-level test.

The URL's month component is ZERO-INDEXED: January is ``00``.
A zero-byte payload means the market was closed for that hour.
"""
from __future__ import annotations

import hashlib
import lzma
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow as pa

from ..config import Config
from ..errors import DecodeError, FetchError, PolicyDeniedError
from ..storage.parquet_store import write_table
from ..storage.paths import ticks_partition
from ..storage.schemas import TICKS
from ..timeutils import UTC, hour_range, require_utc
from ._http import HttpFetcher
from ._manifest import Manifest, ManifestRow

__all__ = ["tick_url", "decode_bi5", "ingest_range", "IngestSummary"]

_RECORD = struct.Struct(">IIIff")
_RECORD_SIZE = 20


def tick_url(base_url: str, symbol: str, hour: datetime, *, month_zero_indexed: bool = True) -> str:
    """Build the hourly tick URL. Month is zero-indexed on Dukascopy."""
    require_utc(hour, name="hour")
    month = hour.month - 1 if month_zero_indexed else hour.month
    return (f"{base_url.rstrip('/')}/{symbol}/{hour.year:04d}/{month:02d}/"
            f"{hour.day:02d}/{hour.hour:02d}h_ticks.bi5")


def decode_bi5(payload: bytes, hour: datetime, *, point_digits: int) -> list[tuple]:
    """Decode one hourly payload into (ts, bid, ask, bid_volume, ask_volume)."""
    require_utc(hour, name="hour")
    if not payload:
        return []
    try:
        body = lzma.decompress(payload, format=lzma.FORMAT_AUTO)
    except lzma.LZMAError as exc:
        raise DecodeError(f"{hour:%Y-%m-%d %H}h: not a valid LZMA stream: {exc}") from exc
    if len(body) % _RECORD_SIZE:
        raise DecodeError(
            f"{hour:%Y-%m-%d %H}h: payload of {len(body)} bytes is not a multiple "
            f"of the {_RECORD_SIZE}-byte record size"
        )
    scale = 10 ** point_digits
    out: list[tuple] = []
    for off in range(0, len(body), _RECORD_SIZE):
        ms, ask_i, bid_i, ask_v, bid_v = _RECORD.unpack_from(body, off)
        out.append((hour + timedelta(milliseconds=ms), bid_i / scale, ask_i / scale,
                    float(bid_v), float(ask_v)))
    return out


def _raw_path(cfg: Config, hour: datetime) -> Path:
    return (cfg.raw_dir / f"{hour.year:04d}" / f"{hour.month:02d}" /
            f"{hour.day:02d}" / f"{hour.hour:02d}h_ticks.bi5")


@dataclass(slots=True)
class IngestSummary:
    hours_considered: int = 0
    fetched: int = 0
    skipped: int = 0
    empty: int = 0
    missing: int = 0
    failed: int = 0
    ticks: int = 0
    months_written: int = 0
    failures: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []


def _ticks_to_table(rows: Sequence[tuple]) -> pa.Table:
    return pa.table({
        "ts": pa.array([r[0] for r in rows], type=TICKS.field("ts").type),
        "bid": pa.array([r[1] for r in rows], type=pa.float64()),
        "ask": pa.array([r[2] for r in rows], type=pa.float64()),
        "bid_volume": pa.array([r[3] for r in rows], type=pa.float64()),
        "ask_volume": pa.array([r[4] for r in rows], type=pa.float64()),
    })


def ingest_range(
    cfg: Config,
    start: datetime,
    end: datetime,
    *,
    fetcher: HttpFetcher | None = None,
    progress: bool = True,
) -> IngestSummary:
    """Download, decode and persist ticks for [start, end). Resume-safe.

    Hours already recorded as ``ok`` or ``empty`` in the manifest are skipped
    without a request. Raw payloads are cached so a decoder fix can be replayed
    without refetching.
    """
    require_utc(start, name="start")
    require_utc(end, name="end")
    dk = cfg.dukascopy
    manifest = Manifest(cfg.manifest_path)
    summary = IngestSummary()

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

    month_rows: dict[tuple[int, int], list[tuple]] = {}
    try:
        for hour in hour_range(start, end):
            summary.hours_considered += 1
            if manifest.should_skip(cfg.symbol, hour):
                summary.skipped += 1
                cached = _raw_path(cfg, hour)
                if cached.exists():
                    payload = cached.read_bytes()
                    rows = decode_bi5(payload, hour, point_digits=cfg.point_digits)
                    if rows:
                        month_rows.setdefault((hour.year, hour.month), []).extend(rows)
                continue

            cached = _raw_path(cfg, hour)
            payload: bytes | None = None
            status: str
            err: str | None = None
            if cached.exists():
                payload = cached.read_bytes()
                status = "empty" if not payload else "ok"
            else:
                try:
                    res = fetcher.get(tick_url(dk.base_url, cfg.symbol, hour,
                                               month_zero_indexed=dk.month_zero_indexed))
                except PolicyDeniedError:
                    raise
                except FetchError as exc:
                    summary.failed += 1
                    err = str(exc)[:300]
                    manifest.record(ManifestRow(cfg.symbol, hour, "failed", 0, None, 0,
                                                datetime.now(UTC), err))
                    summary.failures.append(f"{hour:%Y-%m-%d %H}h: {err[:120]}")
                    continue
                if res.status == 404:
                    status = "missing"
                    payload = b""
                else:
                    payload = res.content
                    status = "empty" if not payload else "ok"
                    summary.fetched += 1
                if dk.keep_raw and status in ("ok", "empty"):
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cached.with_suffix(".tmp")
                    tmp.write_bytes(payload)
                    tmp.replace(cached)

            rows = decode_bi5(payload or b"", hour, point_digits=cfg.point_digits)
            if status == "missing":
                summary.missing += 1
            elif not rows:
                summary.empty += 1
                status = "empty"
            else:
                summary.ticks += len(rows)
                month_rows.setdefault((hour.year, hour.month), []).extend(rows)

            manifest.record(ManifestRow(
                cfg.symbol, hour, status, len(payload or b""),
                hashlib.sha256(payload or b"").hexdigest() if payload else None,
                len(rows), datetime.now(UTC), err,
            ))
            if progress and summary.hours_considered % 50 == 0:
                print(f"  {summary.hours_considered} hours: "
                      f"fetched={summary.fetched} skip={summary.skipped} "
                      f"empty={summary.empty} miss={summary.missing} "
                      f"fail={summary.failed} ticks={summary.ticks:,}", flush=True)
    finally:
        manifest.flush()
        if own:
            fetcher.close()

    for (y, m), rows in sorted(month_rows.items()):
        rows.sort(key=lambda r: r[0])
        write_table(_ticks_to_table(rows),
                    ticks_partition(cfg.ticks_dir, cfg.symbol, y, m),
                    TICKS, sort_by="ts")
        summary.months_written += 1
    return summary
