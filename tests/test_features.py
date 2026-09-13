"""Causality and correctness of the derived feature columns."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pytest

from xau_data.storage.schemas import BARS
from xau_data.transform.features import (
    DEFAULT_ATR_PERIOD,
    add_cost_features,
    atr_wilder,
    true_range,
)

UTC = timezone.utc


def make_bars(n: int, seed: int = 0, spread: float = 0.39) -> pa.Table:
    rng = np.random.default_rng(seed)
    mid = 2300.0 + np.cumsum(rng.normal(0, 1.0, n))
    rng_hl = np.abs(rng.normal(1.2, 0.4, n))
    t0 = datetime(2024, 6, 3, tzinfo=UTC)
    ts = [t0 + timedelta(minutes=15 * i) for i in range(n)]
    sp = np.full(n, spread)
    cols = {
        "ts_open": pa.array(ts, type=BARS.field("ts_open").type),
        "ts_end_exclusive": pa.array([t + timedelta(minutes=15) for t in ts],
                                     type=BARS.field("ts_end_exclusive").type),
        "first_tick_ts": pa.array(ts, type=BARS.field("first_tick_ts").type),
        "last_tick_ts": pa.array([t + timedelta(minutes=14) for t in ts],
                                 type=BARS.field("last_tick_ts").type),
        "tick_count": pa.array(np.full(n, 100), type=pa.int32()),
        "spread_mean": sp, "spread_max": sp + 0.1,
        "spread_min": sp - 0.1, "spread_close": sp,
        "bid_volume": np.full(n, 1.0), "ask_volume": np.full(n, 1.0),
    }
    for side, off in (("bid", -spread / 2), ("ask", spread / 2), ("mid", 0.0)):
        cols[f"{side}_open"] = mid + off
        cols[f"{side}_high"] = mid + rng_hl + off
        cols[f"{side}_low"] = mid - rng_hl + off
        cols[f"{side}_close"] = mid + off
    return pa.table({k: (v if isinstance(v, pa.Array) else pa.array(v, type=pa.float64()))
                     for k, v in cols.items()}).select(BARS.names).cast(BARS)


# --- causality: the whole point ------------------------------------------


def test_features_are_causal_under_truncation() -> None:
    """Feature at bar i must not change when future bars are removed."""
    full = add_cost_features(make_bars(300))
    for cut in (50, 120, 299):
        part = add_cost_features(make_bars(300).slice(0, cut))
        for col in ("atr", "round_trip_cost", "cost_to_atr_ratio", "cost_bps"):
            a = full[col].to_pylist()[:cut]
            b = part[col].to_pylist()
            for i, (x, y) in enumerate(zip(a, b)):
                if x is None and y is None:
                    continue
                assert x == pytest.approx(y, rel=1e-12), (
                    f"{col}[{i}] changed when future bars were removed: {x} vs {y}"
                )


def test_appending_a_bar_never_rewrites_history() -> None:
    base = make_bars(200)
    before = add_cost_features(base.slice(0, 150))["atr"].to_pylist()
    after = add_cost_features(base.slice(0, 151))["atr"].to_pylist()[:150]
    assert before == after


# --- warm-up is NULL, not filled -----------------------------------------


def test_atr_warmup_rows_are_null_not_filled() -> None:
    t = add_cost_features(make_bars(60), atr_period=DEFAULT_ATR_PERIOD)
    atr = t["atr"].to_pylist()
    assert all(v is None for v in atr[:DEFAULT_ATR_PERIOD - 1])
    assert atr[DEFAULT_ATR_PERIOD - 1] is not None
    assert all(v is not None for v in atr[DEFAULT_ATR_PERIOD - 1:])
    ratio = t["cost_to_atr_ratio"].to_pylist()
    assert all(v is None for v in ratio[:DEFAULT_ATR_PERIOD - 1])


def test_shorter_than_period_yields_all_null_atr() -> None:
    t = add_cost_features(make_bars(5), atr_period=14)
    assert all(v is None for v in t["atr"].to_pylist())


# --- arithmetic -----------------------------------------------------------


def test_true_range_uses_previous_close() -> None:
    high = np.array([10.0, 12.0]); low = np.array([9.0, 11.5]); close = np.array([9.5, 12.0])
    tr = true_range(high, low, close)
    assert tr[0] == pytest.approx(1.0)
    # max(12-11.5, |12-9.5|, |11.5-9.5|) = 2.5
    assert tr[1] == pytest.approx(2.5)


def test_atr_matches_hand_computation() -> None:
    n, p = 6, 3
    high = np.array([10.0, 11.0, 12.0, 11.5, 13.0, 12.5])
    low = np.array([9.0, 10.0, 11.0, 10.5, 12.0, 11.5])
    close = np.array([9.5, 10.5, 11.5, 11.0, 12.5, 12.0])
    tr = true_range(high, low, close)
    out = atr_wilder(high, low, close, p)
    seed = float(np.mean(tr[:p]))
    assert out[p - 1] == pytest.approx(seed)
    expect = seed
    for i in range(p, n):
        expect = expect + (tr[i] - expect) / p
        assert out[i] == pytest.approx(expect)


def test_round_trip_cost_is_one_spread_plus_two_slippages() -> None:
    t = add_cost_features(make_bars(40, spread=0.40), slippage_per_side=0.03)
    assert t["round_trip_cost"].to_pylist()[0] == pytest.approx(0.46)


def test_cost_to_atr_ratio_definition() -> None:
    t = add_cost_features(make_bars(100, spread=0.40), slippage_per_side=0.03)
    cost = np.array(t["round_trip_cost"].to_pylist(), dtype=float)
    atr = np.array([np.nan if v is None else v for v in t["atr"].to_pylist()])
    ratio = np.array([np.nan if v is None else v for v in t["cost_to_atr_ratio"].to_pylist()])
    ok = ~np.isnan(atr)
    assert np.allclose(ratio[ok], cost[ok] / atr[ok])


def test_cost_bps_tracks_price_level() -> None:
    """The same dollar cost is fewer bps at a higher price. This is the point."""
    cheap = add_cost_features(make_bars(40, seed=1, spread=0.40))
    t = make_bars(40, seed=1, spread=0.40)
    lifted = {c: t[c] for c in t.column_names}
    for c in t.column_names:
        if (c.endswith(("_open", "_high", "_low", "_close"))
                and not c.startswith(("spread", "ts_"))):
            lifted[c] = pa.array(np.array(t[c].to_pylist(), dtype=float) * 2.0,
                                 type=pa.float64())
    dear = add_cost_features(pa.table(lifted).select(t.column_names).cast(t.schema))
    assert dear["cost_bps"][0].as_py() == pytest.approx(cheap["cost_bps"][0].as_py() / 2, rel=1e-9)
    assert dear["round_trip_cost"][0].as_py() == pytest.approx(cheap["round_trip_cost"][0].as_py())


# --- schema ---------------------------------------------------------------


def test_output_matches_declared_schema_and_keeps_utc() -> None:
    from xau_data.transform.features import FEATURES
    t = add_cost_features(make_bars(50))
    assert t.schema.equals(FEATURES)
    for f in t.schema:
        if pa.types.is_timestamp(f.type):
            assert f.type.tz == "UTC", f"{f.name} lost its timezone"


def test_rejects_missing_columns() -> None:
    from xau_data.errors import SchemaError
    t = make_bars(20).drop_columns(["spread_mean"])
    with pytest.raises(SchemaError, match="spread_mean"):
        add_cost_features(t)
