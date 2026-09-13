"""Individual quality checks. Report only: nothing here modifies data."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Sequence

import numpy as np
import pyarrow as pa

from ..storage.schemas import TIMEFRAME_SECONDS
from ..timeutils import from_epoch_us
from ..transform.sessions import SessionCalendar

__all__ = ["Finding", "Severity", "ALL_CHECKS", "run_checks"]

Severity = Literal["info", "warn", "error"]


@dataclass(slots=True)
class Finding:
    check: str
    severity: Severity
    count: int
    message: str
    samples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.check, "severity": self.severity, "count": self.count,
                "message": self.message, "samples": self.samples[:10]}


def _fmt(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def check_duplicate_bar_timestamps(bars: pa.Table, **_: Any) -> Finding:
    ts = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    uniq, counts = np.unique(ts, return_counts=True)
    dup = uniq[counts > 1]
    return Finding("duplicate_timestamps", "error" if dup.size else "info", int(dup.size),
                   f"{dup.size} duplicated bar open timestamps",
                   [_fmt(from_epoch_us(int(v))) for v in dup[:10]])


def check_monotonic(bars: pa.Table, **_: Any) -> Finding:
    ts = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    bad = int(np.sum(np.diff(ts) < 0))
    return Finding("monotonic_timestamps", "error" if bad else "info", bad,
                   f"{bad} bar timestamps out of order")


def check_non_positive_spread(bars: pa.Table, **_: Any) -> Finding:
    lo = bars["spread_min"].to_numpy(zero_copy_only=False)
    idx = np.flatnonzero(lo <= 0)
    ts = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    return Finding("non_positive_spread", "error" if idx.size else "info", int(idx.size),
                   f"{idx.size} bars contain a tick where ask <= bid",
                   [_fmt(from_epoch_us(int(ts[i]))) for i in idx[:10]])


def check_wide_spread(bars: pa.Table, *, max_spread: float = 25.0, **_: Any) -> Finding:
    hi = bars["spread_max"].to_numpy(zero_copy_only=False)
    idx = np.flatnonzero(hi > max_spread)
    ts = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    return Finding("implausible_spread", "warn" if idx.size else "info", int(idx.size),
                   f"{idx.size} bars with max spread above {max_spread}",
                   [f"{_fmt(from_epoch_us(int(ts[i])))} spread={hi[i]:.3f}" for i in idx[:10]])


def check_ohlc_consistency(bars: pa.Table, **_: Any) -> Finding:
    bad = 0
    samples: list[str] = []
    ts = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    for side in ("bid", "ask", "mid"):
        o = bars[f"{side}_open"].to_numpy(zero_copy_only=False)
        h = bars[f"{side}_high"].to_numpy(zero_copy_only=False)
        l = bars[f"{side}_low"].to_numpy(zero_copy_only=False)
        c = bars[f"{side}_close"].to_numpy(zero_copy_only=False)
        idx = np.flatnonzero((l > np.minimum(o, c)) | (np.maximum(o, c) > h) | (l > h))
        bad += int(idx.size)
        samples += [f"{side} {_fmt(from_epoch_us(int(ts[i])))}" for i in idx[:5]]
    return Finding("ohlc_consistency", "error" if bad else "info", bad,
                   f"{bad} bars violate low <= open,close <= high", samples[:10])


def check_bar_contains_only_own_ticks(bars: pa.Table, **_: Any) -> Finding:
    """Acceptance-grade: no bar may contain a tick at or after its own close."""
    o = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    e = bars["ts_end_exclusive"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    f = bars["first_tick_ts"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    l = bars["last_tick_ts"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    idx = np.flatnonzero((f < o) | (l >= e) | (f > l))
    return Finding("bar_tick_containment", "error" if idx.size else "info", int(idx.size),
                   f"{idx.size} bars contain a tick outside [ts_open, ts_end_exclusive)",
                   [_fmt(from_epoch_us(int(o[i]))) for i in idx[:10]])


def check_thin_bars(bars: pa.Table, *, timeframe: str = "M15",
                    min_tick_count: dict[str, int] | None = None, **_: Any) -> Finding:
    thresh = (min_tick_count or {}).get(timeframe, 1)
    tc = bars["tick_count"].to_numpy(zero_copy_only=False)
    idx = np.flatnonzero(tc < thresh)
    ts = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    return Finding("thin_bars", "warn" if idx.size else "info", int(idx.size),
                   f"{idx.size} bars with tick_count below {thresh}",
                   [f"{_fmt(from_epoch_us(int(ts[i])))} n={tc[i]}" for i in idx[:10]])


def check_price_spikes(bars: pa.Table, *, spike_sigma: float = 8.0, **_: Any) -> Finding:
    c = bars["mid_close"].to_numpy(zero_copy_only=False)
    if c.size < 50:
        return Finding("price_spikes", "info", 0, "too few bars to estimate sigma")
    r = np.diff(np.log(c))
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    sigma = 1.4826 * mad if mad > 0 else np.std(r)
    if sigma <= 0:
        return Finding("price_spikes", "info", 0, "degenerate return distribution")
    z = np.abs(r - med) / sigma
    idx = np.flatnonzero(z > spike_sigma)
    ts = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    return Finding("price_spikes", "warn" if idx.size else "info", int(idx.size),
                   f"{idx.size} bar-to-bar returns beyond {spike_sigma} robust sigma "
                   f"(sigma={sigma:.2e})",
                   [f"{_fmt(from_epoch_us(int(ts[i+1])))} z={z[i]:.1f}" for i in idx[:10]])


def check_out_of_session(bars: pa.Table, *, calendar: SessionCalendar | None = None,
                         **_: Any) -> Finding:
    if calendar is None:
        return Finding("out_of_session", "info", 0, "no session calendar supplied")
    ts = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    bad = [from_epoch_us(int(v)) for v in ts
           if not calendar.is_tradeable(from_epoch_us(int(v)))]
    return Finding("out_of_session", "warn" if bad else "info", len(bad),
                   f"{len(bad)} bars fall outside the trading session "
                   "(weekend, holiday or daily break)",
                   [f"{_fmt(d)} = {d.astimezone(calendar.timezone):%a %H:%M %Z}" for d in bad[:10]])


def check_missing_bars(bars: pa.Table, *, calendar: SessionCalendar | None = None,
                       timeframe: str = "M15", **_: Any) -> Finding:
    if calendar is None or bars.num_rows == 0:
        return Finding("missing_bars", "info", 0, "no session calendar supplied")
    ts = set(bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False).tolist())
    period = TIMEFRAME_SECONDS[timeframe]
    start = from_epoch_us(min(ts))
    end = from_epoch_us(max(ts)) + timedelta(seconds=period)
    expected = list(calendar.expected_bar_opens(start, end, period))
    missing = [d for d in expected if int(d.timestamp() * 1_000_000) not in ts]
    return Finding("missing_bars", "warn" if missing else "info", len(missing),
                   f"{len(missing)} expected bars absent "
                   f"({len(expected)} expected, {len(ts)} present)",
                   [_fmt(d) for d in missing[:10]])


def check_rollover_gap(bars: pa.Table, *, calendar: SessionCalendar | None = None,
                       **_: Any) -> Finding:
    """An absent rollover gap is itself a finding."""
    if calendar is None or bars.num_rows == 0:
        return Finding("rollover_gap", "info", 0, "no session calendar supplied")
    ts = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    days = sorted({from_epoch_us(int(v)).astimezone(calendar.timezone).date() for v in ts})
    present_in_break = 0
    samples: list[str] = []
    for d in days:
        roll = calendar.rollover_utc(d)
        window = [v for v in ts if roll.timestamp() * 1e6 <= v
                  < (roll + timedelta(hours=1)).timestamp() * 1e6]
        if window:
            present_in_break += len(window)
            samples.append(f"{d} rollover {roll:%H:%M}Z has {len(window)} bars")
    return Finding("rollover_gap", "warn" if present_in_break else "info", present_in_break,
                   f"{present_in_break} bars inside the daily rollover break "
                   f"across {len(days)} session days", samples[:10])


ALL_CHECKS = (
    check_bar_contains_only_own_ticks,
    check_duplicate_bar_timestamps,
    check_monotonic,
    check_ohlc_consistency,
    check_non_positive_spread,
    check_wide_spread,
    check_missing_bars,
    check_thin_bars,
    check_price_spikes,
    check_out_of_session,
    check_rollover_gap,
)


def run_checks(bars: pa.Table, **kwargs: Any) -> list[Finding]:
    return [fn(bars, **kwargs) for fn in ALL_CHECKS]
