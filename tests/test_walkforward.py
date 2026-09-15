"""The walk-forward runner: selection must never see an outer test window.

These run before attempt 1 deliberately. Finding a bug after the run would
force a re-run, and a re-run after seeing a result is indistinguishable from
tuning however innocent the intent (0.2i caps attempts at three).
"""
from __future__ import annotations

import numpy as np
import pytest

from xau_backtest.labeling.weights import expanding_folds
from xau_backtest.walkforward.runner import (Q_GRID, WalkForwardConfig,
                                             inner_folds, run_walk_forward,
                                             select_take_fraction,
                                             weighted_mean, weighted_sd)


def test_take_fraction_maximises_expectancy_not_hit_rate():
    """A rule maximising P(net>0) would pick the wrong cut here.

    High-probability bars win often but tiny (rollover-sized); low-probability
    bars win rarely but huge (target-sized). Expectancy must prefer the latter.
    """
    p = np.concatenate([np.full(200, 0.9), np.full(200, 0.4)])
    net = np.concatenate([np.full(200, 0.01), np.tile([6.0, -4.0], 100)])
    q, tau, e = select_take_fraction(p, net, np.ones(400))
    assert tau <= 0.4 + 1e-9, "cut ignored the large-payoff, low-P group"
    assert e > 0.01, f"expectancy {e} did not beat the tiny-win group"


def test_every_take_fraction_sits_at_or_above_the_coverage_floor():
    """Amendment B: selection must not be able to go below the 0.2i floor."""
    assert min(Q_GRID) >= 0.20, "the grid can select coverage below the floor"


def test_take_fraction_never_selects_below_the_floor():
    """Even when the tiny extreme tail has by far the best expectancy."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)
    net = np.where(p > 0.999, 500.0, -1.0)      # a 0.1% tail with a huge payoff
    q, tau, e = select_take_fraction(p, net, np.ones(5000))
    assert q >= 0.20, f"selected q={q}, below the floor"
    assert (p >= tau).mean() >= 0.19, "realised take fell below the floor"


def test_take_fraction_cut_is_read_off_its_own_distribution():
    """tau must be the (1-q) quantile of the SAME predictions it was chosen on.

    Reading it off a different (e.g. test-period) distribution is the
    calibration error that made attempt 1 untestable.
    """
    rng = np.random.default_rng(1)
    p = rng.normal(size=4000)
    net = p + rng.normal(0, 0.5, 4000)
    q, tau, e = select_take_fraction(p, net, np.ones(4000))
    assert abs((p >= tau).mean() - q) < 0.01, (
        f"take fraction {(p >= tau).mean():.4f} does not match selected q={q}"
    )


def test_inner_folds_never_reach_the_outer_test_window():
    n, horizon = 1200, 25
    ex = np.arange(n) + horizon
    outer = expanding_folds(ex, n_blocks=4, embargo_bars=10)
    for f in outer:
        te_lo = int(f.test[0])
        inner = inner_folds(f.train, ex, n_inner=3, embargo_bars=10)
        assert inner, f"fold {f.index} produced no inner folds"
        for g in inner:
            glob = f.train[g.train]
            assert (ex[glob] < te_lo).all(), (
                f"fold {f.index}: an inner-TRAIN label reaches the outer test window"
            )
            assert (f.train[g.test] < te_lo).all(), (
                f"fold {f.index}: an inner-TEST row sits in the outer test window"
            )


def test_inner_folds_expand_and_purge():
    ex = np.arange(900) + 20
    outer = expanding_folds(ex, n_blocks=3, embargo_bars=0)
    inner = inner_folds(outer[-1].train, ex, n_inner=3, embargo_bars=0)
    sizes = [g.train.size for g in inner]
    assert sizes == sorted(sizes), "inner training window shrank"
    assert all(g.purged > 0 for g in inner), "inner folds purged nothing"


def test_weighted_moments():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 2.0])
    assert weighted_mean(x, w) == pytest.approx(2.25)
    assert weighted_sd(x, w) == pytest.approx(np.sqrt(0.6875))


# --- the end-to-end contract ---------------------------------------------


def _toy(n=2400, seed=0, signal=0.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    base = rng.normal(0, 2.0, n)
    net = {"long": base + signal * X[:, 0], "short": -base + signal * X[:, 0]}
    gross = {k: v + 0.2 for k, v in net.items()}
    reason = {k: np.full(n, 1, dtype=np.int32) for k in net}
    ex = np.arange(n) + 20
    w = np.ones(n)
    cost = np.full(n, 0.2)
    return X, [f"f{i}" for i in range(6)], net, gross, reason, w, ex, cost


def test_dual_fire_takes_neither():
    X, names, net, gross, reason, w, ex, cost = _toy()
    res, _ = run_walk_forward(X, names, net, gross, reason, w, ex, cost,
                              cfg=WalkForwardConfig(embargo_bars=5, n_inner=2,
                                                    n_estimators=40),
                              n_blocks=3, stop_atr=4.0, target_atr=6.0)
    assert res
    for f in res:
        fl = f.sides["long"].prob >= f.sides["long"].threshold
        fs = f.sides["short"].prob >= f.sides["short"].threshold
        both = fl & fs
        assert f.dual_fire == int(both.sum())
        assert not f.taken[both].any(), "a dual-fire bar was traded"
        assert np.isnan(f.net[both]).all(), "a dual-fire bar carries a return"


def test_ablation_actually_removes_columns():
    X, names, net, gross, reason, w, ex, cost = _toy()
    cfg = WalkForwardConfig(embargo_bars=5, n_inner=2, n_estimators=40)
    full, _ = run_walk_forward(X, names, net, gross, reason, w, ex, cost, cfg=cfg,
                               n_blocks=3, stop_atr=4.0, target_atr=6.0)
    cut, _ = run_walk_forward(X, names, net, gross, reason, w, ex, cost, cfg=cfg,
                              n_blocks=3, stop_atr=4.0, target_atr=6.0,
                              drop=("f0", "f1", "f2"))
    assert max(s.n_features for f in cut for s in f.sides.values()) <= 3
    assert max(s.n_features for f in full for s in f.sides.values()) > 3


def test_shuffled_train_labels_destroy_any_edge():
    """The control that proves the pipeline does not leak."""
    X, names, net, gross, reason, w, ex, cost = _toy(seed=3, signal=1.5)
    cfg = WalkForwardConfig(embargo_bars=5, n_inner=2, n_estimators=60)
    kw = dict(cfg=cfg, n_blocks=3, stop_atr=4.0, target_atr=6.0)
    real, _ = run_walk_forward(X, names, net, gross, reason, w, ex, cost, **kw)
    shuf, _ = run_walk_forward(X, names, net, gross, reason, w, ex, cost,
                               shuffle_train=True, **kw)

    def e(res):
        n = np.concatenate([f.net[f.taken] for f in res])
        ww = np.concatenate([f.w[f.taken] for f in res])
        return weighted_mean(n, ww)
    assert e(real) > e(shuf), (
        f"a planted signal did not beat shuffled labels: {e(real):.4f} vs {e(shuf):.4f}"
    )


def test_preflight_gate_aborts_without_scoring_the_outer_window():
    """Amendment C: below the floor, the fold must abort before outer scoring.

    An impossible floor of 1.01 forces every fold through the gate, which is
    what lets this assert the gate exists rather than that it rarely fires.
    """
    X, names, net, gross, reason, w, ex, cost = _toy()
    cfg = WalkForwardConfig(embargo_bars=5, n_inner=2, n_estimators=40,
                            coverage_floor=1.01)
    res, aborted = run_walk_forward(X, names, net, gross, reason, w, ex, cost,
                                    cfg=cfg, n_blocks=3, stop_atr=4.0,
                                    target_atr=6.0)
    assert not res, "a fold was scored despite failing the pre-flight gate"
    assert aborted, "the gate did not record an abort"
    for a in aborted:
        assert a.projected_coverage < a.floor
        assert set(a.detail) == {"long", "short"}


def test_preflight_gate_passes_a_healthy_run():
    X, names, net, gross, reason, w, ex, cost = _toy(seed=5, signal=1.0)
    cfg = WalkForwardConfig(embargo_bars=5, n_inner=2, n_estimators=40)
    res, aborted = run_walk_forward(X, names, net, gross, reason, w, ex, cost,
                                    cfg=cfg, n_blocks=3, stop_atr=4.0,
                                    target_atr=6.0)
    assert res, f"every fold aborted: {aborted}"
    for f in res:
        assert f.taken.mean() >= 0.15, (
            f"fold {f.index} scored at {f.taken.mean():.1%} coverage"
        )
