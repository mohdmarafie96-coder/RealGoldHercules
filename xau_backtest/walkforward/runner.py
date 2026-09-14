"""Purged expanding walk-forward, run exactly as pre-registered in 0.2i-0.2k.

Nothing in this module is tuned against an outer test window. All selection --
lambda_l2, the feature count k, and the probability threshold -- happens on
inner expanding folds cut from the outer TRAINING window with the same purging
and embargo machinery.

The threshold objective is uniqueness-weighted EXPECTANCY, not P(net>0):
39.7% of labels exit at time or rollover with continuous returns at sd
1.78-2.05, so P(net>0) scores a +0.01 rollover exit the same as a +5.99 target
hit (0.2j).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from ..labeling.weights import Fold, expanding_folds

__all__ = ["WalkForwardConfig", "SideResult", "FoldResult", "run_walk_forward",
           "select_threshold", "inner_folds", "weighted_mean", "weighted_sd"]

F = npt.NDArray[np.float64]
I = npt.NDArray[np.int64]

LAMBDA_GRID: tuple[float, ...] = (1.0, 10.0, 50.0)
K_GRID: tuple[int, ...] = (10, 20, 40, 58)


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    embargo_bars: int
    n_inner: int = 3
    max_depth: int = 3
    num_leaves: int = 8
    feature_fraction: float = 0.5
    n_estimators: int = 400
    learning_rate: float = 0.03
    seed: int = 7


def weighted_mean(x: F, w: F) -> float:
    s = w.sum()
    return float(np.sum(w * x) / s) if s > 0 else float("nan")


def weighted_sd(x: F, w: F) -> float:
    s = w.sum()
    if s <= 0:
        return float("nan")
    m = np.sum(w * x) / s
    return float(np.sqrt(np.sum(w * (x - m) ** 2) / s))


def inner_folds(train: I, exit_index: I, *, n_inner: int, embargo_bars: int) -> list[Fold]:
    """Expanding inner folds cut from an outer training window.

    The outer train index is contiguous and starts at 0, so the same purging
    rule applies unchanged: an inner-train label whose horizon reaches into the
    inner-test window has seen it and must go.
    """
    if train.size == 0:
        return []
    sub = exit_index[train]
    local = np.where(sub >= 0, np.searchsorted(train, np.clip(sub, 0, None)), -1)
    local = np.where(sub >= train[-1], train.size - 1, local)
    return expanding_folds(local.astype(np.int64), n_blocks=n_inner + 1,
                           embargo_bars=embargo_bars)


def select_threshold(p: F, net: F, w: F, *, min_take: int = 30) -> tuple[float, float]:
    """Threshold maximising uniqueness-weighted expectancy over taken trades.

    Returns (threshold, expectancy). Candidates are the observed probability
    quantiles, so the search never depends on an arbitrary grid.
    """
    if p.size == 0:
        return 1.0, float("nan")
    # Candidates span the FULL range, not just the upper half. An earlier
    # version started at the median, which silently capped coverage at 50%
    # however broad the edge -- and coverage drives power directly (0.2i).
    cand = np.unique(np.quantile(p, np.linspace(0.0, 0.995, 100), method="lower"))
    best_t, best_e = 1.0, -np.inf
    for t in cand:
        m = p >= t
        if m.sum() < min_take:
            continue
        e = weighted_mean(net[m], w[m])
        if e > best_e:
            best_t, best_e = float(t), e
    return best_t, (best_e if np.isfinite(best_e) else float("nan"))


def _params(cfg: WalkForwardConfig, lam: float, min_child: int) -> dict:
    return dict(
        objective="binary", max_depth=cfg.max_depth, num_leaves=cfg.num_leaves,
        min_data_in_leaf=min_child, feature_fraction=cfg.feature_fraction,
        bagging_fraction=1.0,          # 0.2k: row bagging DISABLED
        lambda_l2=lam, learning_rate=cfg.learning_rate,
        verbose=-1, seed=cfg.seed, deterministic=True, force_col_wise=True,
        num_threads=0,
    )


def _fit(X: F, y: npt.NDArray[np.int8], w: F, params: dict, rounds: int):
    """Native Booster API: no scikit-learn dependency, and nulls pass through."""
    import lightgbm as lgb
    ds = lgb.Dataset(X, label=y.astype(np.float64), weight=w, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=rounds)


def _permutation_importance(model, X: F, y: npt.NDArray[np.int8], w: F,
                            net: F, rng: np.random.Generator) -> F:
    """Drop in weighted expectancy at the model's own median split."""
    base_p = model.predict(X)
    t = float(np.median(base_p))
    base = weighted_mean(net[base_p >= t], w[base_p >= t]) if (base_p >= t).any() else 0.0
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, j] = Xp[rng.permutation(X.shape[0]), j]
        p = model.predict(Xp)
        m = p >= t
        out[j] = base - (weighted_mean(net[m], w[m]) if m.any() else 0.0)
    return out


@dataclass
class SideResult:
    side: str
    threshold: float
    breakeven: float
    lam: float
    k: int
    n_features: int
    inner_expectancy: float
    prob: F = field(repr=False, default=None)


@dataclass
class FoldResult:
    index: int
    train_n: int
    test_n: int
    train_eff: float
    test_eff: float
    purged: int
    embargoed: int
    cost_atr: float
    sides: dict[str, SideResult]
    taken: npt.NDArray[np.bool_] = field(repr=False, default=None)
    taken_side: npt.NDArray[np.int8] = field(repr=False, default=None)
    net: F = field(repr=False, default=None)
    gross: F = field(repr=False, default=None)
    w: F = field(repr=False, default=None)
    dual_fire: int = 0
    reason: npt.NDArray[np.int32] = field(repr=False, default=None)


def run_walk_forward(
    X: F, names: Sequence[str], net: dict[str, F], gross: dict[str, F],
    reason: dict[str, npt.NDArray[np.int32]], w: F, exit_index: I,
    cost: F, *, cfg: WalkForwardConfig, n_blocks: int,
    stop_atr: float, target_atr: float,
    shuffle_train: bool = False, drop: Iterable[str] = (),
) -> list[FoldResult]:
    """One full walk-forward pass. `drop` names features to ablate."""
    keep = np.array([i for i, c in enumerate(names) if c not in set(drop)])
    names = [names[i] for i in keep]
    X = X[:, keep]
    live = np.isfinite(net["long"]) & np.isfinite(net["short"])
    ex = np.where(live, exit_index, -1).astype(np.int64)
    folds = expanding_folds(ex, n_blocks=n_blocks, embargo_bars=cfg.embargo_bars)
    rng = np.random.default_rng(cfg.seed)
    out: list[FoldResult] = []

    for f in folds:
        tr = f.train[live[f.train]]
        te = f.test[live[f.test]]
        tr_eff, te_eff = float(w[tr].sum()), float(w[te].sum())
        min_child = max(50, int(round(0.05 * tr_eff)))
        c_fold = float(np.nanmean(cost[te]))
        inner = inner_folds(tr, ex, n_inner=cfg.n_inner, embargo_bars=cfg.embargo_bars)

        sides: dict[str, SideResult] = {}
        for side in ("long", "short"):
            y_all = (net[side] > 0).astype(np.int8)
            y_tr = y_all[tr].copy()
            if shuffle_train:
                y_tr = y_tr[rng.permutation(y_tr.size)]

            # --- importance ranking, on inner validation only ---
            imp = np.zeros(len(names))
            for g in inner:
                a, b = tr[g.train], tr[g.test]
                if a.size < 200 or b.size < 100:
                    continue
                m = _fit(X[a], y_tr[g.train], w[a],
                         _params(cfg, 10.0, min_child), cfg.n_estimators)
                imp += _permutation_importance(m, X[b], y_tr[g.test], w[b],
                                               net[side][b], rng)
            order = np.argsort(-imp)

            # --- grid on inner folds: lambda_l2 x k ---
            best = (-np.inf, 1.0, 10.0, len(names))
            for k in K_GRID:
                cols = order[:min(k, len(names))]
                for lam in LAMBDA_GRID:
                    ps, ns, ws = [], [], []
                    for g in inner:
                        a, b = tr[g.train], tr[g.test]
                        if a.size < 200 or b.size < 100:
                            continue
                        m = _fit(X[np.ix_(a, cols)], y_tr[g.train], w[a],
                                 _params(cfg, lam, min_child), cfg.n_estimators)
                        ps.append(m.predict(X[np.ix_(b, cols)]))
                        ns.append(net[side][b]); ws.append(w[b])
                    if not ps:
                        continue
                    p = np.concatenate(ps); nn = np.concatenate(ns); ww = np.concatenate(ws)
                    t, e = select_threshold(p, nn, ww)
                    if np.isfinite(e) and e > best[0]:
                        best = (e, t, lam, k)
            inner_e, thr, lam, k = best
            cols = order[:min(k, len(names))]
            model = _fit(X[np.ix_(tr, cols)], y_tr, w[tr],
                         _params(cfg, lam, min_child), cfg.n_estimators)
            prob = model.predict(X[np.ix_(te, cols)])
            sides[side] = SideResult(
                side=side, threshold=thr,
                breakeven=(stop_atr + c_fold) / (stop_atr + target_atr),
                lam=lam, k=int(k), n_features=int(cols.size),
                inner_expectancy=float(inner_e), prob=prob)

        fl = sides["long"].prob >= sides["long"].threshold
        fs = sides["short"].prob >= sides["short"].threshold
        dual = int((fl & fs).sum())
        take_l, take_s = fl & ~fs, fs & ~fl          # 0.2j: dual fire -> neither
        taken = take_l | take_s
        tside = np.where(take_l, 1, np.where(take_s, -1, 0)).astype(np.int8)
        n = np.where(take_l, net["long"][te], np.where(take_s, net["short"][te], np.nan))
        g = np.where(take_l, gross["long"][te], np.where(take_s, gross["short"][te], np.nan))
        r = np.where(take_l, reason["long"][te],
                     np.where(take_s, reason["short"][te], 9)).astype(np.int32)

        out.append(FoldResult(
            index=f.index, train_n=int(tr.size), test_n=int(te.size),
            train_eff=tr_eff, test_eff=te_eff, purged=f.purged,
            embargoed=f.embargoed, cost_atr=c_fold, sides=sides,
            taken=taken, taken_side=tside, net=n, gross=g, w=w[te],
            dual_fire=dual, reason=r))
    return out
