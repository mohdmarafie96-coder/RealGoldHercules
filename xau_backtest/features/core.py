"""Causal rolling primitives.

Every function here returns an array the same length as its input where element
``i`` depends only on elements ``0..i``. Warm-up positions are NaN, never
back-filled and never dropped.

Two implementations matter for scale:

- ``rolling_median`` uses a chunked sliding window. A 500-bar window over
  135k rows materialised at once would be 67M floats; chunking bounds it.
- ``rolling_rank`` uses a Fenwick tree over a value histogram. A one-year
  window is 23,629 bars, so a sliding window view would be 3.2 billion
  elements. The Fenwick tree is O(log B) per bar instead.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "ema", "rma", "rolling_mean", "rolling_std", "rolling_median",
    "rolling_min", "rolling_max", "rolling_rank", "shift", "safe_div",
    "log_return",
]

F = npt.NDArray[np.float64]
_CHUNK: Final[int] = 8192
_BUCKETS: Final[int] = 4096


def shift(a: F, k: int) -> F:
    """Shift forward by k, filling the head with NaN. k>0 looks BACKWARD."""
    if k < 0:
        raise ValueError("shift must be non-negative; negative shifts read the future")
    out = np.full_like(a, np.nan)
    if k == 0:
        out[:] = a
    elif k < len(a):
        out[k:] = a[:-k]
    return out


def safe_div(num: F, den: F) -> F:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) > 1e-12, num / den, np.nan)


def log_return(close: F, k: int = 1) -> F:
    prev = shift(close, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((close > 0) & (prev > 0), np.log(close / prev), np.nan)


def ema(a: F, span: int) -> F:
    """Exponential moving average, alpha = 2/(span+1).

    Seeds from the first `span` FINITE values. Seeding from a window that
    contains a leading NaN poisons the whole series: that bug made adx_14
    100% null on the first build.
    """
    if span < 1:
        raise ValueError("span must be positive")
    return _recursive_smooth(a, span, 2.0 / (span + 1.0))


def _recursive_smooth(a: F, period: int, alpha: float) -> F:
    n = len(a)
    out = np.full(n, np.nan)
    finite = np.flatnonzero(np.isfinite(a))
    if finite.size < period:
        return out
    seed_end = int(finite[period - 1])
    prev = float(np.mean(a[finite[:period]]))
    out[seed_end] = prev
    for i in range(seed_end + 1, n):
        if not np.isfinite(a[i]):
            out[i] = prev            # carry the level, do not poison it
            continue
        prev = prev + alpha * (a[i] - prev)
        out[i] = prev
    return out


def rma(a: F, period: int) -> F:
    """Wilder's smoothing, alpha = 1/period. Used by ATR, RSI and ADX."""
    if period < 1:
        raise ValueError("period must be positive")
    return _recursive_smooth(a, period, 1.0 / period)


def _cumsum_rolling(a: F, w: int, fn) -> F:
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    for start in range(w - 1, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        lo = start - w + 1
        view = sliding_window_view(a[lo:stop], w)
        out[start:stop] = fn(view, axis=1)
    return out


def rolling_mean(a: F, w: int) -> F:
    """Trailing mean over w bars INCLUDING the current one."""
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    c = np.concatenate(([0.0], np.cumsum(np.nan_to_num(a))))
    valid = np.concatenate(([0.0], np.cumsum(~np.isnan(a))))
    tot = c[w:] - c[:-w]
    cnt = valid[w:] - valid[:-w]
    out[w - 1:] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out


def rolling_std(a: F, w: int, ddof: int = 1) -> F:
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    x = np.nan_to_num(a)
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s1 = c1[w:] - c1[:-w]
    s2 = c2[w:] - c2[:-w]
    var = (s2 - s1 * s1 / w) / max(w - ddof, 1)
    out[w - 1:] = np.sqrt(np.maximum(var, 0.0))
    return out


def rolling_median(a: F, w: int) -> F:
    return _cumsum_rolling(a, w, np.nanmedian)


def rolling_min(a: F, w: int) -> F:
    return _cumsum_rolling(a, w, np.nanmin)


def rolling_max(a: F, w: int) -> F:
    return _cumsum_rolling(a, w, np.nanmax)


class _Fenwick:
    __slots__ = ("_t", "_n")

    def __init__(self, n: int) -> None:
        self._n = n
        self._t = np.zeros(n + 1, dtype=np.int64)

    def add(self, i: int, delta: int) -> None:
        i += 1
        while i <= self._n:
            self._t[i] += delta
            i += i & (-i)

    def prefix(self, i: int) -> int:
        """Count of items in buckets [0, i]."""
        i += 1
        s = 0
        while i > 0:
            s += self._t[i]
            i -= i & (-i)
        return int(s)


def rolling_rank(a: F, w: int) -> F:
    """Percentile rank of the current value within the trailing w bars, 0-1.

    EXACT, and free of any full-sample quantity. An earlier version bucketed
    values on a grid spanning min..max of the WHOLE series, which is a
    look-ahead however mild: truncating future bars moved the grid and changed
    earlier ranks. It also degraded badly here, since ATR ran 2.66 to 10.41 over
    the sample so late values would have crowded the top buckets.

    Algorithm: visit indices in increasing value order, inserting each position
    into a Fenwick tree indexed by POSITION. When position i is visited, every
    position already inserted holds a value <= a[i], so the count inside
    [i-w+1, i] is exactly the rank. O(n log n), no grid, no leakage.
    """
    n = len(a)
    out = np.full(n, np.nan)
    if n < w or w < 1:
        return out
    finite = np.isfinite(a)
    order = np.argsort(np.where(finite, a, np.inf), kind="stable")
    tree = _Fenwick(n)
    counts = np.zeros(n, dtype=np.int64)

    # ties: insert all equal values before querying any of them
    k = 0
    vals = a[order]
    while k < len(order):
        j = k
        while j + 1 < len(order) and finite[order[j + 1]] and \
                finite[order[k]] and vals[j + 1] == vals[k]:
            j += 1
        for m in range(k, j + 1):
            i = int(order[m])
            if not finite[i]:
                continue
            tree.add(i, 1)
        for m in range(k, j + 1):
            i = int(order[m])
            if not finite[i] or i < w - 1:
                continue
            lo = i - w + 1
            counts[i] = tree.prefix(i) - (tree.prefix(lo - 1) if lo > 0 else 0)
        k = j + 1

    # denominator: how many finite values sit in each trailing window
    valid = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    live = valid[w:] - valid[:-w]
    idx = np.arange(w - 1, n)
    ok = (live > 0) & finite[idx]
    out[idx[ok]] = counts[idx[ok]] / live[ok]
    return out
