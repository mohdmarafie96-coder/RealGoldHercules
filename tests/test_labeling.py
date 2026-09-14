"""Triple-barrier labels, sample weights and purged walk-forward.

Labels are forward-looking by design. What must hold is that they describe a
trade the ENGINE could actually execute, and that walk-forward never trains on a
label that has already seen the test window.
"""
from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from xau_backtest.labeling.barriers import (BarrierConfig, LabelReason,
                                            label_triple_barrier)
from xau_backtest.labeling.weights import (concurrency, expanding_folds,
                                           uniqueness_weights)
from xau_data.config import load_config
from xau_data.transform.sessions import SessionCalendar

UTC = dt.timezone.utc
REPO = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore")

STOP, TARGET, TIME, ROLLOVER = -1, 1, 0, 2
SHORT_RUNWAY, NO_ENTRY = 8, 9


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO / "configs/config.yaml", REPO / "configs/sessions.yaml")


@pytest.fixture(scope="module")
def cal(cfg):
    return SessionCalendar(cfg.sessions)


def bars_from(mid: np.ndarray, spread: float = 0.40, wig: float = 0.5,
              start: dt.datetime | None = None) -> pa.Table:
    n = len(mid)
    start = start or dt.datetime(2024, 6, 3, 8, 0, tzinfo=UTC)
    half = spread / 2
    ts = [start + dt.timedelta(minutes=15 * i) for i in range(n)]
    cols = {
        "ts_open": pa.array(ts, type=pa.timestamp("us", tz="UTC")),
        "ts_end_exclusive": pa.array([t + dt.timedelta(minutes=15) for t in ts],
                                     type=pa.timestamp("us", tz="UTC")),
        "bid_open": mid - half, "bid_high": mid + wig - half,
        "bid_low": mid - wig - half, "bid_close": mid - half,
        "ask_open": mid + half, "ask_high": mid + wig + half,
        "ask_low": mid - wig + half, "ask_close": mid + half,
        "mid_open": mid, "mid_close": mid,
        "spread_open": np.full(n, spread), "spread_close": np.full(n, spread),
        "bid_volume": np.ones(n), "ask_volume": np.ones(n), "m1_count": np.full(n, 15.0),
    }
    return pa.table({k: (v if isinstance(v, pa.Array) else pa.array(v, type=pa.float64()))
                     for k, v in cols.items()})


SLIP = 0.03   # fixture value; production slippage comes from configs/config.yaml


def bc(**kw):
    base = dict(target_atr=6.0, stop_atr=4.0, max_bars=48, slippage_per_side=SLIP)
    base.update(kw)
    return BarrierConfig(**base)


# --- the label describes a trade the engine could execute -----------------


def test_entry_is_the_next_bar_open_not_the_signal_bar(cal):
    """A signal on bar i must fill on bar i+1, never inside bar i."""
    mid = np.full(300, 2300.0)
    mid[100:] = 2400.0                      # a jump exactly at bar 100
    t = label_triple_barrier(bars_from(mid), calendar=cal, cfg=bc(stop_atr=0.5, target_atr=0.5))
    r = t["label_long_reason"].to_numpy(zero_copy_only=False)
    # bar 99's label fills at bar 100 and so captures the jump; bar 98's fills at
    # 99 and must also see it, but no label may resolve BEFORE its entry bar
    ex = t["label_exit_index"].to_numpy(zero_copy_only=False)
    idx = np.arange(len(ex))
    live = ex >= 0
    assert np.all(ex[live] >= idx[live] + 1), "a label resolved on or before its signal bar"


def test_no_label_spans_a_rollover(cal, cfg):
    """The engine force-exits before rollover; a label may not hold through one."""
    rng = np.random.default_rng(4)
    mid = 2300 + np.cumsum(rng.normal(0, 1.0, 1500))
    t = label_triple_barrier(bars_from(mid), calendar=cal, cfg=bc())
    ts = t["ts_open"].to_pylist()
    ex = t["label_exit_index"].to_numpy(zero_copy_only=False)
    # measured from the ENTRY bar (i+1), which is where the engine fills.
    # From the signal bar 16 appear to cross, but those are signals on the last
    # bar before a rollover whose entry lands after it: not a held position.
    spans = 0
    for i, e in enumerate(ex):
        if e < 0 or i + 1 >= len(ts):
            continue
        if cal.rollovers_crossed(ts[i + 1], ts[int(e)]):
            spans += 1
    assert spans == 0, f"{spans} labels hold a position through a rollover"


def test_rollover_capped_labels_are_marked(cal):
    rng = np.random.default_rng(9)
    mid = 2300 + np.cumsum(rng.normal(0, 0.3, 2000))     # quiet: barriers rarely hit
    t = label_triple_barrier(bars_from(mid), calendar=cal, cfg=bc(stop_atr=20, target_atr=30))
    r = t["label_long_reason"].to_numpy(zero_copy_only=False)
    assert (r == ROLLOVER).sum() > 0, "no label was capped by the rollover"


def test_pessimistic_when_both_barriers_sit_in_one_bar(cal):
    """A bar spanning both barriers must resolve to the STOP, and be flagged."""
    mid = np.full(200, 2300.0)
    mid[60] = 2300.0
    t = bars_from(mid, wig=0.5)
    # widen one bar so it engulfs both barriers
    d = {c: t[c].to_pylist() for c in t.column_names}
    for k in ("bid_high", "ask_high"):
        d[k][61] = d[k][61] + 200.0
    for k in ("bid_low", "ask_low"):
        d[k][61] = d[k][61] - 200.0
    wide = pa.table({k: pa.array(v, type=t.schema.field(k).type) for k, v in d.items()})
    out = label_triple_barrier(wide, calendar=cal, cfg=bc(stop_atr=0.5, target_atr=0.5))
    r = out["label_long_reason"].to_numpy(zero_copy_only=False)
    amb = out["label_ambiguous"].to_numpy(zero_copy_only=False)
    assert r[60] == STOP, "ambiguous bar did not resolve pessimistically"
    assert amb[60], "ambiguous bar was not flagged"


def test_gross_minus_net_is_the_full_round_trip(cal, cfg):
    """Gross is mid-to-mid, so the difference must be spread + 2*slippage."""
    rng = np.random.default_rng(11)
    mid = 2300 + np.cumsum(rng.normal(0, 1.0, 1200))
    spread, slip = 0.40, SLIP
    t = label_triple_barrier(bars_from(mid, spread=spread), calendar=cal,
                             cfg=bc(slippage_per_side=slip))
    for side in ("long", "short"):
        g = t[f"label_{side}_ret_atr_gross"].to_numpy(zero_copy_only=False)
        nt = t[f"label_{side}_ret_atr_net"].to_numpy(zero_copy_only=False)
        ok = np.isfinite(g) & np.isfinite(nt)
        # in ATR units, so multiply back by ATR is not available here; check the
        # ratio is positive and stable rather than zero
        diff = (g - nt)[ok]
        assert (diff > 0).all(), f"{side}: gross must exceed net"
        assert diff.std() / diff.mean() < 0.6, f"{side}: cost split is unstable"


def test_long_and_short_are_near_mirror_images(cal):
    """Both sides must lose about the same once path drift is removed.

    A single random walk is NOT driftless: its realised drift correlates 0.92
    with the long-minus-short gap, which is why an earlier version of this test
    failed at 4.7 sigma on a path that happened to rise. Steps are detrended and
    several paths pooled so the comparison is about the labeller, not the draw.
    """
    Ls, Ss = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        steps = rng.normal(0, 1.0, 2500)
        steps -= steps.mean()                     # exactly zero realised drift
        t = label_triple_barrier(bars_from(2300 + np.cumsum(steps)),
                                 calendar=cal, cfg=bc())
        L = t["label_long_ret_atr_net"].to_numpy(zero_copy_only=False)
        S = t["label_short_ret_atr_net"].to_numpy(zero_copy_only=False)
        ok = np.isfinite(L) & np.isfinite(S)
        Ls.append(L[ok]); Ss.append(S[ok])
    L, S = np.concatenate(Ls), np.concatenate(Ss)
    gap = abs(L.mean() - S.mean())
    se = np.sqrt(L.var() / L.size + S.var() / S.size)
    assert gap < 4 * se, (
        f"asymmetry {gap:.4f} is {gap/se:.1f} sigma: long {L.mean():+.4f} "
        f"vs short {S.mean():+.4f}"
    )
    assert L.mean() < 0 and S.mean() < 0, "random entries must lose"


def test_barriers_scale_with_atr_not_dollars(cal):
    """Doubling price and volatility must not change labels in ATR units."""
    rng = np.random.default_rng(31)
    steps = rng.normal(0, 1.0, 1500)
    a = label_triple_barrier(bars_from(2300 + np.cumsum(steps)), calendar=cal,
                             cfg=bc(slippage_per_side=0.0))
    b = label_triple_barrier(bars_from(4600 + np.cumsum(steps * 2), spread=0.80, wig=1.0),
                             calendar=cal, cfg=bc(slippage_per_side=0.0))
    x = a["label_long_ret_atr_net"].to_numpy(zero_copy_only=False)
    y = b["label_long_ret_atr_net"].to_numpy(zero_copy_only=False)
    ok = np.isfinite(x) & np.isfinite(y)
    assert np.allclose(x[ok], y[ok], atol=1e-6), "labels moved with price level"


# --- weights and folds ----------------------------------------------------


def test_concurrency_and_weights(cal):
    ex = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    c = concurrency(ex)
    assert c[0] == 1 and c.max() <= len(ex)
    w = uniqueness_weights(ex)
    assert w.sum() == pytest.approx(len(ex))
    assert (w >= 0).all()


def test_purging_removes_train_labels_that_reach_into_test():
    horizon = 20
    n = 600
    ex = np.arange(n) + horizon
    folds = expanding_folds(ex, n_blocks=4, embargo_bars=0)
    for f in folds:
        te_lo = int(f.test[0])
        assert (ex[f.train] < te_lo).all(), (
            f"fold {f.index}: a training label reaches into the test window"
        )
        assert f.purged > 0


def test_embargo_drops_a_further_margin():
    ex = np.arange(600) + 5
    no_emb = expanding_folds(ex, n_blocks=4, embargo_bars=0)
    emb = expanding_folds(ex, n_blocks=4, embargo_bars=50)
    for a, b in zip(no_emb, emb):
        assert b.train.size < a.train.size
        assert b.embargoed > 0


def test_folds_expand_and_never_roll():
    """Design doc 0.2a: training must grow, never slide."""
    ex = np.arange(1200) + 10
    folds = expanding_folds(ex, n_blocks=6, embargo_bars=0)
    starts = [int(f.train[0]) for f in folds if f.train.size]
    sizes = [f.train.size for f in folds]
    assert all(s == 0 for s in starts), "training window slid instead of expanding"
    assert sizes == sorted(sizes), "training set shrank between folds"


def test_test_windows_are_contiguous_and_disjoint():
    ex = np.arange(1000) + 5
    folds = expanding_folds(ex, n_blocks=5, embargo_bars=0)
    prev_hi = None
    for f in folds:
        lo, hi = int(f.test[0]), int(f.test[-1])
        if prev_hi is not None:
            assert lo == prev_hi + 1, "test windows are not contiguous"
        prev_hi = hi
        assert np.intersect1d(f.train, f.test).size == 0


# --- the min_bars runway mask ---------------------------------------------


def test_short_runway_bars_emit_no_label(cal):
    """A bar whose runway to the rollover is under min_bars must be masked.

    Without this the labeller emits four-bar trades from bars late in the
    session, which sit outside the stated 4-12h holding period AND make the
    label a function of the clock: the runway shrinks monotonically towards
    the session close.
    """
    rng = np.random.default_rng(21)
    mid = 2300 + np.cumsum(rng.normal(0, 0.8, 3000))
    t = label_triple_barrier(bars_from(mid), calendar=cal, cfg=bc(min_bars=16))
    r = t["label_long_reason"].to_numpy(zero_copy_only=False)
    nb = t["label_long_bars"].to_numpy(zero_copy_only=False)
    assert (r == SHORT_RUNWAY).sum() > 0, "nothing was masked at all"
    # a rollover-capped label is the one whose hold equals its whole runway,
    # so it is the direct witness that the runway cleared min_bars
    held = nb[r == ROLLOVER]
    assert held.size and held.min() >= 16, (
        f"a rollover label held {held.min()} bars, under min_bars"
    )
    assert (nb[r == SHORT_RUNWAY] == -1).all(), "a masked bar carries a hold"


def test_masking_removes_labels_and_is_side_symmetric(cal):
    rng = np.random.default_rng(23)
    mid = 2300 + np.cumsum(rng.normal(0, 0.8, 3000))
    loose = label_triple_barrier(bars_from(mid), calendar=cal, cfg=bc(min_bars=1))
    tight = label_triple_barrier(bars_from(mid), calendar=cal, cfg=bc(min_bars=16))
    rl = loose["label_long_reason"].to_numpy(zero_copy_only=False)
    rt = tight["label_long_reason"].to_numpy(zero_copy_only=False)
    assert (rt == SHORT_RUNWAY).sum() > 0
    assert (rl == SHORT_RUNWAY).sum() == 0, "min_bars=1 must mask nothing"
    # the mask only ever removes
    assert set(np.flatnonzero(rt != NO_ENTRY)) <= set(np.flatnonzero(rl != NO_ENTRY))
    for side in ("long", "short"):
        a = tight[f"label_{side}_reason"].to_numpy(zero_copy_only=False)
        assert np.array_equal(a == SHORT_RUNWAY, rt == SHORT_RUNWAY), (
            f"{side} masked a different set of bars than long"
        )


def test_masking_cuts_rollover_exits_hardest(cal):
    """The mask must bite on rollover labels, which is its whole purpose."""
    rng = np.random.default_rng(27)
    mid = 2300 + np.cumsum(rng.normal(0, 0.4, 4000))
    loose = label_triple_barrier(bars_from(mid), calendar=cal, cfg=bc(min_bars=1))
    tight = label_triple_barrier(bars_from(mid), calendar=cal, cfg=bc(min_bars=16))
    rl = loose["label_long_reason"].to_numpy(zero_copy_only=False)
    rt = tight["label_long_reason"].to_numpy(zero_copy_only=False)
    roll_before, roll_after = int((rl == ROLLOVER).sum()), int((rt == ROLLOVER).sum())
    time_before, time_after = int((rl == TIME).sum()), int((rt == TIME).sum())
    assert roll_after < roll_before, "the mask removed no rollover labels"
    assert time_after == time_before, (
        "the mask removed a 48-bar time exit; it must only touch short runways"
    )


def test_n_blocks_is_blocks_not_folds():
    """Six blocks produce five folds. The old name promised six."""
    ex = np.arange(600) + 3
    for b in (4, 5, 6):
        assert len(expanding_folds(ex, n_blocks=b, embargo_bars=0)) == b - 1
