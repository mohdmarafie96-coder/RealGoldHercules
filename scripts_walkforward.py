"""Attempt 1 of the pre-registered walk-forward (0.2i), plus the C.7 guards.

Run once. Nothing is tuned between the run and the report.
"""
from __future__ import annotations

import json, math, sys, warnings
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
from xau_backtest.labeling.weights import concurrency, expanding_folds
from xau_backtest.walkforward.runner import (WalkForwardConfig, run_walk_forward,
                                             weighted_mean, weighted_sd)

warnings.filterwarnings("ignore")
OUT = Path("data/_reports")
CACHE = Path("data/_cache")
Z = 2.128                       # Bonferroni over three attempts, one-sided
AL_POOLED = 0.0398              # pre-registered comparator, folds 1-4
SESSION_FEATS = ("session_asian", "session_london", "session_overlap",
                 "session_ny", "session_progress", "mins_to_rollover",
                 "tod_sin", "tod_cos")
REASON = {1: "target", -1: "stop", 0: "time", 2: "rollover"}


def uniqueness(ex, live):
    exm = np.where(live, ex, -1).astype(np.int64)
    c = concurrency(exm).astype(float)
    inv = np.where(c > 0, 1.0 / np.maximum(c, 1.0), 0.0)
    cum = np.concatenate(([0.0], np.cumsum(inv)))
    u = np.zeros(len(exm))
    for i, e in enumerate(exm):
        if e < i:
            continue
        e = min(int(e), len(exm) - 1)
        u[i] = (cum[e + 1] - cum[i]) / (e - i + 1)
    return u


def labels_for(m15, cal, cfg, stop, target):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"labels_{stop:g}_{target:g}.parquet"
    if p.exists():
        return pq.read_table(p)
    lc = cfg.labeling
    bc = BarrierConfig(target_atr=target, stop_atr=stop, max_bars=lc["max_bars"],
                       min_bars=lc["min_bars"],
                       slippage_per_side=cfg.costs.slippage_per_side)
    t = label_triple_barrier(m15, calendar=cal, cfg=bc)
    pq.write_table(t, p)
    return t


def pool(res, folds=(1, 2, 3, 4)):
    """Pooled taken-trade stats over the named folds."""
    n, w = [], []
    for f in res:
        if f.index not in folds:
            continue
        m = f.taken
        n.append(f.net[m]); w.append(f.w[m])
    if not n:
        return dict(e=float("nan"), se=float("nan"), eff=0.0, n=0)
    n = np.concatenate(n); w = np.concatenate(w)
    e = weighted_mean(n, w); sd = weighted_sd(n, w); eff = float(w.sum())
    return dict(e=e, se=sd / math.sqrt(eff) if eff > 0 else float("nan"),
                eff=eff, n=int(n.size), sd=sd)


def coverage(res, folds=(1, 2, 3, 4)):
    tk = sum(int(f.taken.sum()) for f in res if f.index in folds)
    tot = sum(int(f.test_n) for f in res if f.index in folds)
    return tk / tot if tot else float("nan")


def fold_excess(f, al_by_fold):
    m = f.taken
    if m.sum() == 0:
        return float("nan")
    return weighted_mean(f.net[m], f.w[m]) - al_by_fold[f.index]


def main() -> int:
    cfg = load_config(); cal = SessionCalendar(cfg.sessions)
    lc, wf = cfg.labeling, cfg.walk_forward
    m15 = resample_m1(open_store(cfg.data_root).read_bars_m1(cfg.symbol), timeframe="M15")
    feats = build_features(m15, calendar=cal,
                           slippage_per_side=cfg.costs.slippage_per_side)
    names = list(feature_schema(feats))
    X = np.column_stack([feats[c].to_numpy(zero_copy_only=False) for c in names])
    spread = ((m15["spread_open"].to_numpy(zero_copy_only=False)
               + m15["spread_close"].to_numpy(zero_copy_only=False)) / 2.0)
    from xau_backtest.features.groups import _atr
    atr = _atr(m15["bid_high"].to_numpy(zero_copy_only=False),
               m15["bid_low"].to_numpy(zero_copy_only=False),
               m15["bid_close"].to_numpy(zero_copy_only=False), lc["atr_period"])
    cost = (spread + 2 * cfg.costs.slippage_per_side) / atr

    wfc = WalkForwardConfig(embargo_bars=int(wf["embargo_bars"]))
    nb = int(wf["n_blocks"])
    L: list[str] = []
    P = L.append

    def load(stop, target):
        t = labels_for(m15, cal, cfg, stop, target)
        net = {s: t[f"label_{s}_ret_atr_net"].to_numpy(zero_copy_only=False)
               for s in ("long", "short")}
        gro = {s: t[f"label_{s}_ret_atr_gross"].to_numpy(zero_copy_only=False)
               for s in ("long", "short")}
        rea = {s: t[f"label_{s}_reason"].to_numpy(zero_copy_only=False).astype(np.int32)
               for s in ("long", "short")}
        ex = t["label_exit_index"].to_numpy(zero_copy_only=False)
        live = np.isfinite(net["long"]) & np.isfinite(net["short"])
        return net, gro, rea, ex, live, uniqueness(ex, live)

    net, gro, rea, ex, live, u = load(lc["stop_atr"], lc["target_atr"])

    # always-long comparator, per fold and pooled
    folds0 = expanding_folds(np.where(live, ex, -1).astype(np.int64),
                             n_blocks=nb, embargo_bars=int(wf["embargo_bars"]))
    al = {}
    for f in folds0:
        te = f.test[live[f.test]]
        al[f.index] = weighted_mean(net["long"][te], u[te])
    te14 = np.concatenate([f.test[live[f.test]] for f in folds0 if f.index >= 1])
    al_pooled = weighted_mean(net["long"][te14], u[te14])

    P("=" * 78)
    P("ATTEMPT 2 OF 3 -- PRE-REGISTERED WALK-FORWARD (0.2i + amendments A/B/C)")
    P("=" * 78)
    P(f"barriers {lc['stop_atr']}/{lc['target_atr']} ATR   features {len(names)}   "
      f"blocks {nb} -> folds {len(folds0)}   embargo {wf['embargo_bars']}")
    P(f"always-long comparator, folds 1-4 pooled: {al_pooled:+.4f} ATR "
      f"(pre-registered {AL_POOLED:+.4f})")

    def run(tag, **kw):
        return run_walk_forward(X, names, net, gro, rea, u, ex, cost, cfg=wfc,
                                n_blocks=nb, stop_atr=lc["stop_atr"],
                                target_atr=lc["target_atr"], **kw)

    print("running headline...", file=sys.stderr)
    res, aborted = run("headline")
    if aborted:
        P("\n" + "=" * 78)
        P("AMENDMENT C -- PRE-FLIGHT COVERAGE GATE FIRED")
        P("=" * 78)
        for a in aborted:
            P(f"  fold {a.index}: projected coverage {a.projected_coverage:.1%} "
              f"< floor {a.floor:.0%}   ABORTED BEFORE OUTER SCORING")
            for sd, d in a.detail.items():
                P(f"     {sd:<5} q {d['q']:.2f}  tau {d['tau']:.4f}  "
                  f"lambda_l2 {d['lam']:g}  k {d['k']}")
        P("  0.2i: a run aborted before outer scoring does NOT consume an attempt.")
    if not res:
        P("\n  VERDICT: ABORTED -- no outer window was scored. Attempt NOT consumed.")
        txt = "\n".join(L) + "\n"
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "walkforward_attempt2.txt").write_text(txt)
        print(txt)
        return 0

    # ---------------- C.6 per-fold block ----------------
    P("\n" + "=" * 78)
    P("C.6  PER-FOLD REPORT")
    P("=" * 78)
    for f in res:
        m = f.taken
        e = weighted_mean(f.net[m], f.w[m]) if m.any() else float("nan")
        gr = weighted_mean(f.gross[m], f.w[m]) if m.any() else float("nan")
        eff_t = float(f.w[m].sum())
        tag = "  [FOLD 0 -- REPORTED SEPARATELY, 0.2g: excluded from headline]" \
            if f.index == 0 else ""
        P(f"\nFOLD {f.index}{tag}")
        P(f"  train {f.train_n:,} labels (eff {f.train_eff:.1f})   "
          f"test {f.test_n:,} (eff {f.test_eff:.1f})   "
          f"purged {f.purged}  embargoed {f.embargoed}")
        P(f"  realised cost c = {f.cost_atr:.4f} ATR    "
          f"min_child_samples = {max(50, int(round(0.05*f.train_eff)))}")
        P(f"  taken {int(m.sum()):,} / {f.test_n:,} = {m.mean():.1%} coverage "
          f"(eff n {eff_t:.1f})   dual-fire {f.dual_fire:,} "
          f"({f.dual_fire/max(f.test_n,1):.2%})")
        P(f"  NET   {e:+.4f} ATR      GROSS {gr:+.4f} ATR      "
          f"always-long {al[f.index]:+.4f}")
        P(f"  E_excess {e - al[f.index]:+.4f} ATR")
        P(f"  pre-flight projected coverage {f.projected_coverage:.1%} "
          f"(floor {wfc.coverage_floor:.0%})")
        for s, sr in f.sides.items():
            P(f"  {s:<5} q {sr.take_fraction:.2f}  tau {sr.threshold:.4f}  "
              f"breakeven {sr.breakeven:.4f}  lambda_l2 {sr.lam:g}  "
              f"k {sr.n_features}  inner E {sr.inner_expectancy:+.4f}")
        if m.any():
            win = m & (f.net > 0)
            P(f"  win rate {win.sum()/m.sum():.2%}  "
              f"(fold breakeven {f.sides['long'].breakeven:.2%})")
            tot = max(int(win.sum()), 1)
            mix = {REASON.get(int(k), str(k)): int((f.reason[win] == k).sum())
                   for k in (1, -1, 0, 2)}
            P("  WINNER exit-reason mix: " + "  ".join(
                f"{k} {v:,} ({100*v/tot:.1f}%)" for k, v in mix.items()))
            P(f"  {'':<22} neutral baseline: target 46.2%  time 24.4%  "
              f"rollover 29.4%   (0.2i-NOTE)")
            lm = m & (f.taken_side == 1)
            P(f"  side split: long {int(lm.sum()):,}  short {int((m&(f.taken_side==-1)).sum()):,}")

    # ---------------- THE RESULT ----------------
    p14 = pool(res); cov = coverage(res)
    E = p14["e"] - al_pooled
    lb = E - Z * p14["se"]
    per_fold = {f.index: fold_excess(f, al) for f in res}
    pos = sum(1 for k, v in per_fold.items() if k >= 1 and v > 0)
    worst = min(v for k, v in per_fold.items() if k >= 1)
    c1, c2, c3, c4 = lb > 0, E >= 0.15, pos >= 3, worst > -0.30

    P("\n" + "=" * 78)
    P("THE RESULT -- the single pre-registered scalar (0.2i)")
    P("=" * 78)
    P(f"  model  Wnet (folds 1-4 pooled)   {p14['e']:+.4f} ATR")
    P(f"  always-long, same bars           {al_pooled:+.4f} ATR")
    P(f"  E_excess                         {E:+.4f} ATR")
    P(f"  coverage                         {cov:.1%}   "
      f"effective n {p14['eff']:.1f}   label sd {p14['sd']:.4f}")
    P(f"  SE at realised coverage          {p14['se']:.4f}")
    P(f"  one-sided lower bound (z=2.128)  {lb:+.4f}")
    P("")
    P(f"  per-fold E_excess:  " + "   ".join(
        f"f{k} {v:+.4f}" for k, v in sorted(per_fold.items())))
    P("")
    P(f"  condition 1  E_excess - 2.128*SE > 0     {'PASS' if c1 else 'FAIL'}"
      f"   ({lb:+.4f})")
    P(f"  condition 2  E_excess >= +0.15           {'PASS' if c2 else 'FAIL'}"
      f"   ({E:+.4f})")
    P(f"  condition 3  >=3 of 4 folds positive     {'PASS' if c3 else 'FAIL'}"
      f"   ({pos} of 4)")
    P(f"  condition 4  no fold worse than -0.30    {'PASS' if c4 else 'FAIL'}"
      f"   (worst {worst:+.4f})")
    verdict = "PASS" if (c1 and c2 and c3 and c4) else "FAIL"
    if cov < 0.20:
        verdict = "UNTESTABLE (coverage below the 20% floor)"
    P(f"\n  VERDICT: {verdict}")

    # ---------------- C.7 guards ----------------
    P("\n" + "=" * 78)
    P("C.7  GUARDS")
    P("=" * 78)

    print("guard 1: shuffled labels...", file=sys.stderr)
    sh, _ = run("shuffle", shuffle_train=True)
    ps = pool(sh)
    pooled_label = weighted_mean(
        np.concatenate([net["long"][te14], net["short"][te14]]),
        np.concatenate([u[te14], u[te14]]))
    P(f"\n1. SHUFFLED-LABEL CONTROL (train targets permuted, test real)")
    P(f"   Wnet {ps['e']:+.4f}   E_excess {ps['e']-al_pooled:+.4f}   "
      f"coverage {coverage(sh):.1%}")
    P(f"   pooled label mean, both sides, same bars: {pooled_label:+.4f}")
    P(f"   -> a leaking pipeline would show a positive edge here. "
      f"{'CLEAN' if ps['e'] - al_pooled < 0.10 else 'INVESTIGATE'}")

    print("guard 2: time-of-day ablation...", file=sys.stderr)
    ab, _ = run("ablate", drop=SESSION_FEATS)
    pa_ = pool(ab)
    P(f"\n2. TIME-OF-DAY ABLATION ({len(SESSION_FEATS)} session features dropped)")
    P(f"   Wnet {pa_['e']:+.4f}   E_excess {pa_['e']-al_pooled:+.4f}   "
      f"coverage {coverage(ab):.1%}")
    P(f"   headline E_excess {E:+.4f}  ->  ablated {pa_['e']-al_pooled:+.4f}   "
      f"delta {(pa_['e']-al_pooled)-E:+.4f}")

    P(f"\n3. COST-REGIME SPLIT")
    hi, lo = pool(res, folds=(1, 2)), pool(res, folds=(3, 4))
    alhi = weighted_mean(*(lambda t: (net["long"][t], u[t]))(
        np.concatenate([f.test[live[f.test]] for f in folds0 if f.index in (1, 2)])))
    allo = weighted_mean(*(lambda t: (net["long"][t], u[t]))(
        np.concatenate([f.test[live[f.test]] for f in folds0 if f.index in (3, 4)])))
    chi = np.mean([f.cost_atr for f in res if f.index in (1, 2)])
    clo = np.mean([f.cost_atr for f in res if f.index in (3, 4)])
    P(f"   folds 1-2 (c = {chi:.4f}):  Wnet {hi['e']:+.4f}  AL {alhi:+.4f}  "
      f"E_excess {hi['e']-alhi:+.4f}")
    P(f"   folds 3-4 (c = {clo:.4f}):  Wnet {lo['e']:+.4f}  AL {allo:+.4f}  "
      f"E_excess {lo['e']-allo:+.4f}")

    P(f"\n4. GEOMETRY ROBUSTNESS (reported, never selected on -- 0.2h)")
    P(f"   {'geometry':<12} {'E_excess':>10} {'coverage':>10} {'Wnet':>10} {'AL':>10}")
    P(f"   {'4.0/6.0':<12} {E:>+10.4f} {cov:>10.1%} {p14['e']:>+10.4f} {al_pooled:>+10.4f}")
    for st, tg in ((3.0, 4.5), (6.0, 9.0)):
        print(f"guard 4: geometry {st}/{tg}...", file=sys.stderr)
        n2, g2, r2, e2, l2, u2 = load(st, tg)
        f0 = expanding_folds(np.where(l2, e2, -1).astype(np.int64), n_blocks=nb,
                             embargo_bars=int(wf["embargo_bars"]))
        t2 = np.concatenate([f.test[l2[f.test]] for f in f0 if f.index >= 1])
        al2 = weighted_mean(n2["long"][t2], u2[t2])
        r, _ = run_walk_forward(X, names, n2, g2, r2, u2, e2, cost, cfg=wfc,
                                n_blocks=nb, stop_atr=st, target_atr=tg)
        pr = pool(r)
        P(f"   {f'{st:g}/{tg:g}':<12} {pr['e']-al2:>+10.4f} {coverage(r):>10.1%} "
          f"{pr['e']:>+10.4f} {al2:>+10.4f}")

    P("\n" + "=" * 78)
    P("HOW TO READ A FAIL (0.2i)")
    P("=" * 78)
    P("  A FAIL means: no edge above roughly 0.25 ATR was demonstrated on this")
    P("  sample.")
    P("  A FAIL does NOT mean: no edge exists.")
    P(f"  At the realised coverage of {cov:.1%}, the joint power against a true")
    P("  edge of +0.15 ATR is well below 50%, so a true edge at the economic")
    P("  floor fails this test more often than it passes.")

    txt = "\n".join(L) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "walkforward_attempt2.txt").write_text(txt)
    (OUT / "walkforward_attempt2.json").write_text(json.dumps(dict(
        E_excess=E, lower_bound=lb, coverage=cov, wnet=p14["e"],
        always_long=al_pooled, se=p14["se"], eff_n=p14["eff"],
        per_fold=per_fold, conditions=dict(c1=bool(c1), c2=bool(c2),
                                           c3=bool(c3), c4=bool(c4)),
        verdict=verdict), indent=2))
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
