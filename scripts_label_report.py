"""Regenerate labels under the min_bars mask and report every phase-6 table.

Produces, on the real history:
  1. exit-reason split before and after the min_bars runway mask
  2. time-exit P&L distribution in ATR units
  3. class balance by year
  4. effective sample by year after uniqueness weighting
  5. per-fold null rate for every feature
  6. the ALWAYS-LONG benchmark through the identical folds

The always-long benchmark exists because gold rose from 1868 to 4394 over the
sample. A model that learns "mostly long, sometimes flat" harvests that drift
and presents as skill. Zero is the wrong bar; always-long is the right one.
"""
from __future__ import annotations

import json, warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xau_data.config import load_config
from xau_data.storage.store import open_store
from xau_data.transform.resample import resample_m1
from xau_data.transform.sessions import SessionCalendar
from xau_backtest.features.build import build_features, feature_schema
from xau_backtest.labeling.barriers import BarrierConfig, label_triple_barrier
from xau_backtest.labeling.weights import (concurrency, expanding_folds,
                                           uniqueness_weights)

warnings.filterwarnings("ignore")

CODE = {1: "target", -1: "stop", 0: "time", 2: "rollover", 3: "end_of_data",
        8: "short_runway", 9: "no_entry"}
OUT = Path("data/_reports")


def bar(title: str) -> str:
    return "\n" + "=" * 78 + f"\n{title}\n" + "=" * 78


def reason_table(r: np.ndarray, label: str) -> str:
    n = len(r)
    lines = [f"{label}  (n = {n:,})", "  reason            count       %"]
    for c, name in sorted(CODE.items(), key=lambda kv: -int((r == kv[0]).sum())):
        k = int((r == c).sum())
        if k == 0:
            continue
        lines.append(f"  {name:<14} {k:>9,}  {100*k/n:>6.2f}")
    return "\n".join(lines)


def quantiles(x: np.ndarray, qs=(1, 5, 25, 50, 75, 95, 99)) -> str:
    v = np.percentile(x, qs)
    return "  ".join(f"p{q}={val:+.4f}" for q, val in zip(qs, v))


def main() -> int:
    cfg = load_config()
    cal = SessionCalendar(cfg.sessions)
    lc = cfg.labeling
    wf = cfg.walk_forward
    bcfg = BarrierConfig(target_atr=float(lc["target_atr"]),
                         stop_atr=float(lc["stop_atr"]),
                         max_bars=int(lc["max_bars"]),
                         min_bars=int(lc["min_bars"]),
                         slippage_per_side=cfg.costs.slippage_per_side)
    unmasked = BarrierConfig(target_atr=bcfg.target_atr, stop_atr=bcfg.stop_atr,
                             max_bars=bcfg.max_bars, min_bars=1,
                             slippage_per_side=bcfg.slippage_per_side)

    m15 = resample_m1(open_store(cfg.data_root).read_bars_m1(cfg.symbol),
                      timeframe="M15")
    feats = build_features(m15, calendar=cal,
                           slippage_per_side=cfg.costs.slippage_per_side)
    n = m15.num_rows
    years = np.array([t.year for t in m15["ts_open"].to_pylist()])

    before = label_triple_barrier(m15, calendar=cal, cfg=unmasked)
    after = label_triple_barrier(m15, calendar=cal, cfg=bcfg)

    OUT.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    L.append(f"bars {n:,}   barriers {bcfg.stop_atr}/{bcfg.target_atr} ATR   "
             f"slippage_per_side {bcfg.slippage_per_side}   "
             f"max_bars {bcfg.max_bars}   min_bars {bcfg.min_bars}")

    # ---- 1. masking -------------------------------------------------------
    L.append(bar("1. RUNWAY MASK (min_bars enforced)"))
    rb = before["label_long_reason"].to_numpy(zero_copy_only=False)
    ra = after["label_long_reason"].to_numpy(zero_copy_only=False)
    L.append(reason_table(rb, "BEFORE mask, long side"))
    L.append("")
    L.append(reason_table(ra, "AFTER mask, long side"))
    masked = int((ra == 8).sum())
    lab_b = int((rb != 9).sum())
    lab_a = int((ra != 9) & (ra != 8)).__int__() if False else int(((ra != 9) & (ra != 8)).sum())
    L.append(f"\n  masked by short runway : {masked:,}"
             f"\n  labelled before        : {lab_b:,}"
             f"\n  labelled after         : {lab_a:,}"
             f"\n  removed                : {lab_b - lab_a:,} "
             f"({100*(lab_b-lab_a)/max(lab_b,1):.2f}% of previously labelled)")

    # ---- effective n ------------------------------------------------------
    def eff(tbl, mask=None):
        ex = tbl["label_exit_index"].to_numpy(zero_copy_only=False).copy()
        if mask is not None:
            ex = np.where(mask, ex, -1)
        live = ex >= 0
        c = concurrency(np.where(live, ex, -1))
        cc = c[live] if live.any() else np.array([0])
        w = uniqueness_weights(np.where(live, ex, -1))
        # effective n = sum of per-label uniqueness (un-normalised)
        raw = np.zeros(len(ex))
        conc = concurrency(np.where(live, ex, -1)).astype(float)
        inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1.0), 0.0)
        cum = np.concatenate(([0.0], np.cumsum(inv)))
        for i, e in enumerate(ex):
            if e < i:
                continue
            e = min(int(e), len(ex) - 1)
            raw[i] = (cum[e + 1] - cum[i]) / (e - i + 1)
        return live, raw, cc

    live_b, raw_b, cc_b = eff(before)
    live_a, raw_a, cc_a = eff(after)
    L.append(f"\n  effective n BEFORE : {raw_b.sum():,.0f}  "
             f"(concurrency mean {cc_b.mean():.1f} max {cc_b.max()})")
    L.append(f"  effective n AFTER  : {raw_a.sum():,.0f}  "
             f"(concurrency mean {cc_a.mean():.1f} max {cc_a.max()})")

    # ---- 2. time-exit P&L distribution ------------------------------------
    L.append(bar("2. TIME-EXIT P&L DISTRIBUTION, ATR UNITS (after mask)"))
    for side in ("long", "short"):
        net = after[f"label_{side}_ret_atr_net"].to_numpy(zero_copy_only=False)
        gro = after[f"label_{side}_ret_atr_gross"].to_numpy(zero_copy_only=False)
        rr = after[f"label_{side}_reason"].to_numpy(zero_copy_only=False)
        L.append(f"\n{side.upper()}")
        for code, name in ((0, "time (48-bar)"), (2, "rollover"),
                           (1, "target"), (-1, "stop")):
            m = (rr == code) & np.isfinite(net)
            if not m.any():
                continue
            L.append(f"  {name:<14} n={int(m.sum()):>7,}  "
                     f"net mean {net[m].mean():+.4f}  sd {net[m].std():.4f}  "
                     f"gross mean {gro[m].mean():+.4f}")
            L.append(f"  {'':<14} {quantiles(net[m])}")
        m = np.isfinite(net)
        L.append(f"  {'ALL':<14} n={int(m.sum()):>7,}  net mean {net[m].mean():+.4f}"
                 f"  gross mean {gro[m].mean():+.4f}"
                 f"  cost {gro[m].mean()-net[m].mean():.4f}")

    # ---- 3/4. by year -----------------------------------------------------
    L.append(bar("3+4. CLASS BALANCE AND EFFECTIVE SAMPLE BY YEAR (after mask)"))
    netL = after["label_long_ret_atr_net"].to_numpy(zero_copy_only=False)
    netS = after["label_short_ret_atr_net"].to_numpy(zero_copy_only=False)
    L.append("  year    bars  labelled  masked   longwin%  shortwin%   "
             "eff n   eff/lab   net long   net short")
    for y in sorted(set(years.tolist())):
        m = years == y
        lab = m & np.isfinite(netL)
        if not lab.any():
            continue
        msk = int((m & (ra == 8)).sum())
        L.append(f"  {y}  {int(m.sum()):>6,}  {int(lab.sum()):>8,}  {msk:>6,}   "
                 f"{100*(netL[lab]>0).mean():>7.2f}   {100*(netS[lab]>0).mean():>8.2f}  "
                 f"{raw_a[lab].sum():>7.1f}  {raw_a[lab].sum()/lab.sum():>7.4f}  "
                 f"{netL[lab].mean():>+9.4f}  {netS[lab].mean():>+9.4f}")

    # ---- 5/6. folds -------------------------------------------------------
    L.append(bar("5+6. FOLDS: NULL RATES AND THE ALWAYS-LONG BENCHMARK"))
    ex = after["label_exit_index"].to_numpy(zero_copy_only=False)
    ex = np.where(live_a, ex, -1)
    folds = expanding_folds(ex, n_folds=int(wf["n_folds"]),
                            embargo_bars=int(wf["embargo_bars"]))
    L.append(f"  n_folds parameter = {wf['n_folds']} (that is the number of "
             f"equal BLOCKS); folds produced = {len(folds)}")
    L.append("\n  fold   train     test  purged  embargo   train_eff  test_eff"
             "   AL net    AL wtd   AL win%")
    fold_rows = []
    for f in folds:
        tr, te = f.train, f.test
        tr = tr[live_a[tr]]
        te_l = te[live_a[te]]
        al = netL[te_l]
        w = raw_a[te_l]
        alw = float(np.sum(al * w) / np.sum(w)) if w.sum() > 0 else float("nan")
        row = dict(fold=f.index, train=int(tr.size), test=int(te_l.size),
                   purged=f.purged, embargoed=f.embargoed,
                   train_eff=float(raw_a[tr].sum()), test_eff=float(w.sum()),
                   al_net=float(al.mean()), al_net_wtd=alw,
                   al_win=float(100 * (al > 0).mean()))
        fold_rows.append(row)
        L.append(f"  {f.index:>4} {tr.size:>7,} {te_l.size:>8,} {f.purged:>7} "
                 f"{f.embargoed:>8}  {row['train_eff']:>10.1f} "
                 f"{row['test_eff']:>9.1f}  {row['al_net']:>+7.4f}  "
                 f"{alw:>+7.4f}  {row['al_win']:>7.2f}")

    L.append("\n  ALWAYS-LONG is the benchmark every model result must beat.")
    L.append("  'AL net' is the unweighted mean net label in ATR over the test")
    L.append("  window; 'AL wtd' weights by uniqueness. A model that cannot")
    L.append("  beat AL wtd per fold has harvested drift, not found structure.")

    # per-fold feature null rates
    names = list(feature_schema(feats))
    L.append(f"\n  PER-FOLD NULL RATE, all {len(names)} features (% null in TRAIN)")
    hdr = "  feature                       " + "".join(
        f"  f{f.index}_tr  f{f.index}_te" for f in folds)
    L.append(hdr)
    fmat = {c: feats[c].to_numpy(zero_copy_only=False) for c in names}
    worst = []
    for c in names:
        v = fmat[c]
        cells = ""
        gaps = []
        for f in folds:
            tr = f.train[live_a[f.train]]
            te = f.test[live_a[f.test]]
            a = 100 * np.isnan(v[tr]).mean() if tr.size else float("nan")
            b = 100 * np.isnan(v[te]).mean() if te.size else float("nan")
            gaps.append(a - b)
            cells += f"  {a:>6.1f} {b:>6.1f}"
        L.append(f"  {c:<28}{cells}")
        worst.append((max(gaps), c, gaps))
    worst.sort(reverse=True)
    L.append("\n  LARGEST TRAIN-MINUS-TEST NULL GAP (feature-space mismatch):")
    for g, c, gaps in worst[:12]:
        L.append(f"    {c:<28} worst gap {g:>6.1f} pp   per fold "
                 + " ".join(f"{x:+.1f}" for x in gaps))

    txt = "\n".join(L) + "\n"
    (OUT / "label_report.txt").write_text(txt)
    (OUT / "fold_benchmark.json").write_text(json.dumps(fold_rows, indent=2))
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
