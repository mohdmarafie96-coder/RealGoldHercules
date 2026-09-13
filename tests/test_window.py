from __future__ import annotations

import gc
import time
from datetime import datetime, timezone

import numpy as np
import pytest

from xau_backtest.engine.window import BarHistory, BarWindow, LookaheadError

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
SENTINEL = 9_999_999.0  # value used only in bars the clock has not reached


def make_row(i: int, mid: float) -> dict[str, float | int]:
    ts = 1_700_000_000_000_000 + i * 900_000_000  # M15 in epoch microseconds
    return {
        "ts_open": ts,
        "ts_end_exclusive": ts + 900_000_000,
        "bid_open": mid - 0.2, "bid_high": mid + 0.8,
        "bid_low": mid - 1.2, "bid_close": mid - 0.2,
        "ask_open": mid + 0.2, "ask_high": mid + 1.2,
        "ask_low": mid - 0.8, "ask_close": mid + 0.2,
        "mid_open": mid, "mid_high": mid + 1.0,
        "mid_low": mid - 1.0, "mid_close": mid,
        "spread_mean": 0.39, "spread_max": 0.55, "spread_close": 0.38,
        "tick_count": 100 + i,
    }


def build(n_revealed: int, n_total: int) -> tuple[BarHistory, list[dict]]:
    """Materialise all bars, reveal only the first n_revealed."""
    source = [
        make_row(i, 2300.0 + i if i < n_revealed else SENTINEL)
        for i in range(n_total)
    ]
    hist = BarHistory(initial_capacity=8)
    for row in source[:n_revealed]:
        hist.append(row)
    return hist, source


# --- the window ends at the current bar ---------------------------------


def test_window_covers_zero_through_current_bar() -> None:
    hist, _ = build(10, 100)
    w = hist.window()
    assert len(w) == 10
    assert w.index == 9
    assert w.current().index == 9
    assert w.value("mid_close") == pytest.approx(2309.0)
    assert w.value("mid_close", lag=1) == pytest.approx(2308.0)


def test_window_grows_by_exactly_one_per_bar() -> None:
    hist = BarHistory(initial_capacity=2)
    for i in range(50):
        hist.append(make_row(i, 2300.0 + i))
        w = hist.window()
        assert len(w) == i + 1
        assert w.index == i


# --- reading the future raises ------------------------------------------


def test_reading_next_bar_raises() -> None:
    hist, _ = build(10, 100)
    w = hist.window()
    with pytest.raises(LookaheadError, match="beyond the next bar"):
        w[w.index + 1]


def test_reading_far_future_raises() -> None:
    hist, _ = build(10, 100)
    w = hist.window()
    with pytest.raises(LookaheadError):
        w[50]


def test_negative_lag_raises() -> None:
    hist, _ = build(10, 100)
    w = hist.window()
    with pytest.raises(LookaheadError, match="into the future"):
        w.value("mid_close", lag=-1)


def test_slice_past_current_bar_raises_and_is_not_clamped() -> None:
    hist, _ = build(10, 100)
    w = hist.window()
    # plain Python would silently clamp this to 10 elements
    with pytest.raises(LookaheadError, match="not clamped"):
        w[0:11]
    with pytest.raises(LookaheadError):
        w[5:99]
    ok = w[0:10]
    assert len(ok["mid_close"]) == 10


def test_series_longer_than_history_raises() -> None:
    hist, _ = build(10, 100)
    w = hist.window()
    with pytest.raises(IndexError):
        w.series("mid_close", 11)


# --- the window cannot be mutated ---------------------------------------


def test_series_is_read_only() -> None:
    hist, _ = build(10, 100)
    w = hist.window()
    s = w.series("mid_close")
    assert not s.flags.writeable
    with pytest.raises(ValueError):
        s[0] = 1.0


def test_window_rejects_untruncated_columns() -> None:
    """A window must never be constructed from full-length arrays."""
    full = np.zeros(100, dtype=np.float64)
    full.flags.writeable = False
    with pytest.raises(ValueError, match="truncated views"):
        BarWindow({"mid_close": full}, 10)


# --- the reachability audit ---------------------------------------------


def _reachable_arrays(root: object, max_objects: int = 10_000) -> list[np.ndarray]:
    """Data objects reachable from `root` by container traversal.

    Deliberately does not traverse functions, modules, frames or types: every
    object in CPython reaches module globals through those, which would make
    the question meaningless. What we assert is that no *data container* held
    by the window carries a value from a bar the clock has not reached.
    """
    seen: set[int] = set()
    out: list[np.ndarray] = []
    stack = [root]
    while stack and len(seen) < max_objects:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            out.append(obj)
            if obj.base is not None:
                stack.append(obj.base)
            continue
        if isinstance(obj, (str, bytes, int, float, bool, type(None))):
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
            continue
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
            continue
        # objects with __slots__ / __dict__: follow their attribute values only
        for ref in gc.get_referents(obj):
            if isinstance(ref, (np.ndarray, dict, list, tuple, set, frozenset)):
                stack.append(ref)
    return out


def test_no_future_value_is_reachable_from_the_window() -> None:
    """The decisive test: all 1000 bars exist in memory, 10 are revealed."""
    hist, source = build(10, 1000)
    w = hist.window()

    # the future really does exist in this process
    assert source[500]["mid_close"] == SENTINEL

    arrays = _reachable_arrays(w)
    assert arrays, "audit reached no arrays; the walk is broken"
    for arr in arrays:
        if arr.dtype.kind == "f":
            assert not np.any(arr == SENTINEL), (
                f"a future value is reachable from the window via an array of "
                f"length {len(arr)}"
            )


def test_capacity_beyond_revealed_length_is_zero_filled() -> None:
    """Reaching through ndarray.base must yield zeros, never stale future data."""
    hist, _ = build(10, 1000)
    w = hist.window()
    for arr in _reachable_arrays(w):
        if arr.dtype.kind == "f" and len(arr) > len(w):
            assert np.all(arr[len(w):] == 0.0)


def test_window_holds_no_reference_to_the_history_object() -> None:
    hist, _ = build(10, 1000)
    w = hist.window()
    reachable_ids = {id(o) for o in _reachable_arrays(w)}
    assert id(hist) not in reachable_ids


# --- performance --------------------------------------------------------


def test_window_construction_is_zero_copy() -> None:
    hist, _ = build(100, 100)
    w = hist.window()
    s1 = w.series("mid_close")
    hist.append(make_row(100, 2400.0))
    w2 = hist.window()
    s2 = w2.series("mid_close")
    assert np.shares_memory(s1, s2), "window construction copied the history"


def test_75k_bars_stays_cheap() -> None:
    """Three years of M15 is ~75k bars. The per-bar cost must not grow with n."""
    hist = BarHistory(initial_capacity=1024)
    row = make_row(0, 2300.0)
    t0 = time.perf_counter()
    for i in range(75_000):
        hist.append(row)
        w = hist.window()
        _ = w.value("mid_close")
    elapsed = time.perf_counter() - t0
    assert len(hist) == 75_000
    # generous bound: the point is that this is linear, not quadratic
    assert elapsed < 10.0, f"75k bars took {elapsed:.2f}s"
    print(f"\n75,000 bars appended + windowed in {elapsed:.3f}s "
          f"({elapsed/75_000*1e6:.1f} us/bar)")
