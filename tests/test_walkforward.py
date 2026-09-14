"""The walk-forward runner: selection must never see an outer test window.

These run before attempt 1 deliberately. Finding a bug after the run would
force a re-run, and a re-run after seeing a result is indistinguishable from
tuning however innocent the intent (0.2i caps attempts at three).
"""
from __future__ import annotations

import numpy as np
import pytest

from xau_backtest.labeling.weights import expanding_folds
from xau_backtest.walkforward.runner import (WalkForwardConfig, inner_folds,
                                             run_walk_forward, select_threshold,
                                             weighted_mean, weighted_sd)


def test_select_threshold_maximises_expectancy_not_hit_rate():
    """A rule maximising P(net>0) would pick the wrong threshold here.

    High-probability bars win often but tiny (rollover-sized); low-probability
    bars win rarely but huge (target-sized). Expectancy must prefer the latter.
    """
    p = np.concatenate([np.full(200, 0.9), np.full(200, 0.4)])
    net = np.concatenate([np.full(200, 0.01), np.tile([6.0, -4.0], 100)])
    w = np.ones(400)
    t, e = select_threshold(p, net, w)
    assert t <= 0.4 + 1e-9, "threshold ignored the large-payoff, low-P group"
    assert e > 0.01, f"expectancy {e} did not beat the tiny-win group"


def test_select_threshold_respects_min_take():
    p = np.linspace(0, 1, 100)
    net = np.where(p > 0.99, 100.0, -1.0)
    t, e = select_threshold(p, net, np.ones(100), min_take=30)
    assert (p >= t).sum() >= 30, "threshold took fewer trades than min_take"


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
    res = run_walk_forward(X, names, net, gross, reason, w, ex, cost,
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
    full = run_walk_forward(X, names, net, gross, reason, w, ex, cost, cfg=cfg,
                            n_blocks=3, stop_atr=4.0, target_atr=6.0)
    cut = run_walk_forward(X, names, net, gross, reason, w, ex, cost, cfg=cfg,
                           n_blocks=3, stop_atr=4.0, target_atr=6.0,
                           drop=("f0", "f1", "f2"))
    assert max(s.n_features for f in cut for s in f.sides.values()) <= 3
    assert max(s.n_features for f in full for s in f.sides.values()) > 3


def test_shuffled_train_labels_destroy_any_edge():
    """The control that proves the pipeline does not leak."""
    X, names, net, gross, reason, w, ex, cost = _toy(seed=3, signal=1.5)
    cfg = WalkForwardConfig(embargo_bars=5, n_inner=2, n_estimators=60)
    kw = dict(cfg=cfg, n_blocks=3, stop_atr=4.0, target_atr=6.0)
    real = run_walk_forward(X, names, net, gross, reason, w, ex, cost, **kw)
    shuf = run_walk_forward(X, names, net, gross, reason, w, ex, cost,
                            shuffle_train=True, **kw)

    def e(res):
        n = np.concatenate([f.net[f.taken] for f in res])
        ww = np.concatenate([f.w[f.taken] for f in res])
        return weighted_mean(n, ww)
    assert e(real) > e(shuf), (
        f"a planted signal did not beat shuffled labels: {e(real):.4f} vs {e(shuf):.4f}"
    )
