"""Causality across EVERY feature, not a sample.

The contract, reused from cost_to_atr_ratio: truncating future bars must not
change an earlier value, and appending a bar must not rewrite history.

Two real bugs were caught by writing these, both of which produced
plausible-looking columns:

- Wilder smoothing seeded from a window containing leading NaNs, which made
  adx_14 100% null.
- rolling_rank bucketed values on a grid spanning the whole series, so
  truncating the future moved the grid and shifted earlier ranks.
"""
from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from xau_backtest.features.build import build_features, feature_schema
from xau_backtest.features.core import (ema, rma, rolling_mean, rolling_median,
                                        rolling_rank, rolling_std, shift)
from xau_data.config import load_config
from xau_data.transform.sessions import SessionCalendar

UTC = dt.timezone.utc
REPO = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", message="Mean of empty slice")


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO / "configs/config.yaml", REPO / "configs/sessions.yaml")


@pytest.fixture(scope="module")
def cal(cfg):
    return SessionCalendar(cfg.sessions)


def synth(n: int, seed: int = 0, scale_break_at: int | None = None) -> pa.Table:
    """M15 bars. `scale_break_at` multiplies volatility, mimicking 2025-26."""
    rng = np.random.default_rng(seed)
    step = np.ones(n)
    if scale_break_at is not None:
        step[scale_break_at:] = 4.0
    mid = 2300.0 + np.cumsum(rng.normal(0, 1.0, n) * step)
    wig = np.abs(rng.normal(1.0, 0.3, n)) * step
    spread = np.clip(rng.normal(0.40, 0.10, n), 0.1, None) * np.sqrt(step)
    half = spread / 2
    t0 = dt.datetime(2023, 1, 2, tzinfo=UTC)
    ts = [t0 + dt.timedelta(minutes=15 * i) for i in range(n)]
    cols = {
        "ts_open": pa.array(ts, type=pa.timestamp("us", tz="UTC")),
        "ts_end_exclusive": pa.array([t + dt.timedelta(minutes=15) for t in ts],
                                     type=pa.timestamp("us", tz="UTC")),
        "bid_open": mid - half, "bid_high": mid + wig - half,
        "bid_low": mid - wig - half, "bid_close": mid - half,
        "ask_open": mid + half, "ask_high": mid + wig + half,
        "ask_low": mid - wig + half, "ask_close": mid + half,
        "mid_open": mid, "mid_close": mid,
        "spread_open": spread, "spread_close": spread,
        "bid_volume": np.abs(rng.normal(0.5, 0.2, n)),
        "ask_volume": np.abs(rng.normal(0.5, 0.2, n)),
        "m1_count": np.full(n, 15.0),
    }
    return pa.table({k: (v if isinstance(v, pa.Array) else pa.array(v, type=pa.float64()))
                     for k, v in cols.items()})


def _build(tbl, cal, cfg, rank_window=400):
    return build_features(tbl, calendar=cal,
                          slippage_per_side=cfg.costs.slippage_per_side,
                          rank_window=rank_window)


# --- the contract, applied to every feature -------------------------------


@pytest.mark.parametrize("cut", [600, 900, 1400])
def test_every_feature_is_causal_under_truncation(cfg, cal, cut):
    full = _build(synth(1600, seed=3), cal, cfg)
    part = _build(synth(1600, seed=3).slice(0, cut), cal, cfg)
    offenders: list[str] = []
    for name in feature_schema(full):
        a = full[name].to_numpy(zero_copy_only=False)[:cut]
        b = part[name].to_numpy(zero_copy_only=False)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = ~both_nan & ~np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        if diff.any():
            offenders.append(f"{name} ({int(diff.sum())} of {cut} rows differ)")
    assert not offenders, (
        "features changed when future bars were removed:\n  " + "\n  ".join(offenders)
    )


def test_appending_a_bar_never_rewrites_history(cfg, cal):
    base = synth(1200, seed=5)
    before = _build(base.slice(0, 1000), cal, cfg)
    after = _build(base.slice(0, 1001), cal, cfg)
    for name in feature_schema(before):
        a = before[name].to_numpy(zero_copy_only=False)
        b = after[name].to_numpy(zero_copy_only=False)[:1000]
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), name


def test_a_later_volatility_regime_shift_cannot_reach_back(cfg, cal):
    """Mimics 2025-26: vol quadruples late. Early features must be identical."""
    flat = _build(synth(1600, seed=7), cal, cfg)
    broken = _build(synth(1600, seed=7, scale_break_at=1000), cal, cfg)
    for name in feature_schema(flat):
        a = flat[name].to_numpy(zero_copy_only=False)[:1000]
        b = broken[name].to_numpy(zero_copy_only=False)[:1000]
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), (
            f"{name} leaked a later regime shift backwards"
        )


# --- scale-freedom --------------------------------------------------------


def test_no_feature_scales_with_price_level(cfg, cal):
    """Double every price and volatility. Scale-free features must not move."""
    base = synth(1200, seed=11)
    lifted = {}
    for c in base.column_names:
        if c.startswith("ts_"):
            lifted[c] = base[c]
        elif c in ("bid_volume", "ask_volume", "m1_count"):
            lifted[c] = base[c]
        else:
            lifted[c] = pa.array(
                np.array(base[c].to_pylist(), dtype=float) * 2.0, type=pa.float64())
    a = _build(base, cal, cfg)
    b = _build(pa.table(lifted).select(base.column_names), cal, cfg)
    # cost features legitimately move: slippage is a fixed dollar amount
    exempt = {"cost_to_atr_ratio", "spread_to_atr", "atr_pct",
              "spread_rel_median_500", "spread_z_500", "spread_pct_rank_1y"}
    drifted: list[str] = []
    for name in feature_schema(a):
        if name in exempt:
            continue
        x = a[name].to_numpy(zero_copy_only=False)
        y = b[name].to_numpy(zero_copy_only=False)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 50:
            continue
        if not np.allclose(x[ok], y[ok], rtol=1e-6, atol=1e-9):
            worst = float(np.max(np.abs(x[ok] - y[ok])))
            drifted.append(f"{name} (max delta {worst:.3e})")
    assert not drifted, "features that move with price level:\n  " + "\n  ".join(drifted)


# --- session one-hot ------------------------------------------------------


def test_session_one_hot_off_is_all_zeros(cfg, cal):
    f = _build(synth(2000, seed=13), cal, cfg)
    cols = ["session_asian", "session_london", "session_overlap", "session_ny"]
    m = np.vstack([f[c].to_numpy(zero_copy_only=False) for c in cols])
    total = m.sum(axis=0)
    assert set(np.unique(total)).issubset({0.0, 1.0}), (
        "session one-hot must be mutually exclusive"
    )
    assert (total == 0.0).any(), "the 'off' session must appear as all-zeros"
    assert (total == 1.0).any()


def test_dropped_features_are_absent(cfg, cal):
    f = _build(synth(800, seed=17), cal, cfg)
    names = set(feature_schema(f))
    assert "is_wide_bar" not in names
    assert "regime_rank_is_expanding" not in names
    assert "bar_range_rel_median_500" in names   # carries the same information


# --- primitives -----------------------------------------------------------


def test_rolling_rank_is_exact(cfg):
    rng = np.random.default_rng(0)
    a = rng.normal(size=1500)
    w = 200
    r = rolling_rank(a, w)
    for i in (w - 1, 700, 1499):
        exact = float((a[i - w + 1:i + 1] <= a[i]).sum()) / w
        assert r[i] == pytest.approx(exact, abs=1e-12)


def test_smoothers_survive_leading_nans(cfg):
    a = np.concatenate([np.full(13, np.nan), np.arange(60.0)])
    assert np.isfinite(rma(a, 14)[-1]), "rma poisoned by leading NaN"
    assert np.isfinite(ema(a, 20)[-1]), "ema poisoned by leading NaN"


def test_shift_refuses_to_look_forward(cfg):
    with pytest.raises(ValueError, match="read the future"):
        shift(np.arange(10.0), -1)
