"""Sample weights and purged walk-forward splits.

Overlapping labels are the quiet problem in financial ML. A label spanning bars
i..i+40 shares almost all of its outcome with the label at i+1, so treating them
as independent observations inflates the effective sample by roughly the
horizon. Two corrections, both standard:

- **Uniqueness weights.** Weight each label by the inverse of how many other
  labels were live at the same time. A label overlapping 40 others is worth
  about 1/40 of an independent one.
- **Purging and embargo.** A training label whose horizon reaches into the test
  window has seen test-period prices. It must be removed, not merely ordered
  before. An embargo then drops a further margin after the test window, because
  features are trailing and a train row just after the test set still overlaps
  it through its own lookback.

Without both, walk-forward leaks and every downstream number is optimistic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["concurrency", "uniqueness_weights", "Fold", "expanding_folds"]

I = npt.NDArray[np.int64]


def concurrency(exit_index: I, n: int | None = None) -> I:
    """How many labels are live at each bar."""
    n = n if n is not None else len(exit_index)
    live = np.zeros(n + 1, dtype=np.int64)
    for i, e in enumerate(exit_index):
        if e < i:
            continue
        e = min(int(e), n - 1)            # clamp: a horizon may run past the data
        live[i] += 1
        live[min(e + 1, n)] -= 1
    return np.cumsum(live[:n])


def uniqueness_weights(exit_index: I) -> npt.NDArray[np.float64]:
    """Average inverse concurrency over each label's own lifetime."""
    n = len(exit_index)
    c = concurrency(exit_index, n).astype(np.float64)
    inv = np.where(c > 0, 1.0 / np.maximum(c, 1.0), 0.0)
    cum = np.concatenate(([0.0], np.cumsum(inv)))
    w = np.zeros(n)
    for i, e in enumerate(exit_index):
        if e < i:
            continue
        e = min(int(e), n - 1)
        span = e - i + 1
        w[i] = (cum[e + 1] - cum[i]) / span
    tot = w.sum()
    return w * (n / tot) if tot > 0 else w


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    train: I
    test: I
    purged: int
    embargoed: int


def expanding_folds(
    exit_index: I, *, n_folds: int = 6, embargo_bars: int = 0,
    min_train: int = 1,
) -> list[Fold]:
    """EXPANDING-window walk-forward with purging and an embargo.

    Expanding, never rolling, per design doc 0.2a: volatility regimes cluster in
    time and a rolling window trained on 2023 would never have seen a bar above
    65 USD before meeting one of 287.
    """
    n = len(exit_index)
    if n_folds < 2:
        raise ValueError("need at least two folds")
    edges = np.linspace(0, n, n_folds + 1).astype(np.int64)
    folds: list[Fold] = []
    for k in range(1, n_folds):
        te_lo, te_hi = int(edges[k]), int(edges[k + 1])
        train_all = np.arange(0, te_lo, dtype=np.int64)
        if train_all.size < min_train:
            continue
        # purge: drop train labels whose horizon reaches into the test window
        reach = exit_index[train_all]
        keep = reach < te_lo
        purged = int((~keep).sum())
        train = train_all[keep]
        # embargo: also drop train rows within `embargo_bars` of the test start
        emb = 0
        if embargo_bars > 0 and train.size:
            cut = te_lo - embargo_bars
            before = train.size
            train = train[train < cut]
            emb = before - train.size
        folds.append(Fold(k - 1, train, np.arange(te_lo, te_hi, dtype=np.int64),
                          purged, emb))
    return folds
