"""Assemble every feature group into one causal table.

Higher-timeframe features use COMPLETED higher-timeframe bars only. At an M15
bar the current H1 bar is three quarters unformed, so using it would be
lookahead in ordinary clothing. The H1 value attached to 09:15 comes from the
H1 bar that closed at 09:00.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from . import groups as G
from .core import ema, safe_div, shift

__all__ = ["build_features", "FEATURE_NAMES", "feature_schema"]

F = npt.NDArray[np.float64]
_EPOCH: Final[datetime] = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _higher_timeframe_slope(ts_us: npt.NDArray[np.int64], h: F, l: F, c: F,
                            period_seconds: int, span: int = 20,
                            lag: int = 5) -> F:
    """EMA slope on a higher timeframe, in that timeframe's own ATR units.

    Only COMPLETED higher-timeframe bars contribute. The value carried at an
    M15 bar is the slope as of the last higher-timeframe bar that had already
    closed when that M15 bar opened.
    """
    p = period_seconds * 1_000_000
    key = ts_us - (ts_us % p)
    # aggregate to the higher timeframe
    starts = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1))
    stops = np.concatenate((starts[1:], [len(ts_us)]))
    hi = np.array([np.max(h[a:b]) for a, b in zip(starts, stops)])
    lo = np.array([np.min(l[a:b]) for a, b in zip(starts, stops)])
    cl = c[stops - 1]
    e = ema(cl, span)
    tr = np.maximum.reduce([hi - lo,
                            np.abs(hi - shift(cl, 1)),
                            np.abs(lo - shift(cl, 1))])
    from .core import rma
    atr_h = rma(np.where(np.isnan(tr), hi - lo, tr), 14)
    slope = safe_div(e - shift(e, lag), atr_h)

    # map back: an M15 bar sees the slope of the PREVIOUS completed HTF bar
    completed = shift(slope, 1)
    out = np.full(len(ts_us), np.nan)
    for gi, (a, b) in enumerate(zip(starts, stops)):
        out[a:b] = completed[gi]
    return out


def build_features(bars: pa.Table, *, calendar, slippage_per_side: float,
                   rank_window: int = G.BARS_PER_YEAR) -> pa.Table:
    """Return ts_open plus every feature, all causal, warm-up rows NULL."""
    bars = bars.sort_by([("ts_open", "ascending")])
    n = bars.num_rows
    if n == 0:
        raise ValueError("no bars")

    b: dict[str, F] = {
        c: bars[c].to_numpy(zero_copy_only=False).astype(np.float64)
        for c in bars.column_names if c not in ("ts_open", "ts_end_exclusive")
    }
    ts_us = bars["ts_open"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    ts = bars["ts_open"].to_pylist()

    feats: dict[str, F] = {}

    vol = G.volatility(b)
    atr14 = vol.pop("_atr14")
    rv20 = vol.pop("_rv20")
    feats.update(vol)

    higher = {
        "h1_slope": _higher_timeframe_slope(ts_us, b["bid_high"], b["bid_low"],
                                            b["bid_close"], 3600),
        "h4_slope": _higher_timeframe_slope(ts_us, b["bid_high"], b["bid_low"],
                                            b["bid_close"], 4 * 3600),
    }
    feats.update(G.trend(b, atr14, higher))

    st = G.session_time(ts, calendar)
    session_id = st.pop("_session_id").astype(np.int64)
    feats.update(st)

    feats.update(G.momentum(b, atr14, rv20, session_id))
    feats.update(G.microstructure(b, atr14))

    spread = (b["spread_open"] + b["spread_close"]) / 2.0
    cost = spread + 2.0 * slippage_per_side
    feats.update(G.regime(b, atr14, rv20, cost, rank_window))

    # Asian range WIDTH only. Position is a proven dead end for direction.
    feats["asian_range_atr"] = _asian_range_atr(ts, b, atr14, calendar)

    cols: dict[str, pa.Array] = {"ts_open": bars["ts_open"]}
    for k in sorted(feats):
        v = np.asarray(feats[k], dtype=np.float64)
        cols[k] = pa.array(v, type=pa.float64(), mask=~np.isfinite(v))
    return pa.table(cols)


def _asian_range_atr(ts: Sequence[datetime], b: Mapping[str, F], atr14: F,
                     calendar) -> F:
    """Asian session range / ATR, carried forward within the day ONLY after the
    Asian session has closed. Never uses a bar later than the current one."""
    n = len(ts)
    out = np.full(n, np.nan)
    hi = lo = None
    cur_day = None
    closed_range: float | None = None
    for i, t in enumerate(ts):
        d = t.date()
        if d != cur_day:
            cur_day, hi, lo, closed_range = d, None, None, None
        lab = calendar.session_label(t).value
        if lab == "asian":
            hi = b["bid_high"][i] if hi is None else max(hi, b["bid_high"][i])
            lo = b["bid_low"][i] if lo is None else min(lo, b["bid_low"][i])
        else:
            if closed_range is None and hi is not None and lo is not None:
                closed_range = hi - lo
        if closed_range is not None and np.isfinite(atr14[i]) and atr14[i] > 0:
            out[i] = closed_range / atr14[i]
    return out


FEATURE_NAMES: tuple[str, ...] = ()


def feature_schema(table: pa.Table) -> tuple[str, ...]:
    return tuple(c for c in table.column_names if c != "ts_open")
