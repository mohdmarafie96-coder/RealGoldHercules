"""Decompose walk-forward attempt 2. RE-READ ONLY -- consumes no attempt.

Attempt 2's trade-level arrays were never persisted; only the summary was. That
is an instrumentation gap in scripts_walkforward.py, not a reason to test
anything new. This script re-executes the SAME deterministic passes
(seed fixed, LightGBM deterministic=True) and ASSERTS that the headline
reproduces attempt 2 exactly before reporting anything. If it does not
reproduce, it stops: a non-reproducing run would be a new experiment wearing a
re-read's clothes.

Nothing here is a new configuration. No geometry pass is re-run -- the
decomposition does not need one.
"""
from __future__ import annotations

import json, math, sys, warnings
from pathlib import Path

import numpy as np

from xau_data.config import load_config
from xau_data.storage.store import open_store
from xau_data.transform.resample import resample_m1
from xau_data.transform.sessions import SessionCalendar
from xau_backtest.features.build import build_features, feature_schema
from xau_backtest.labeling.weights import expanding_folds
from xau_backtest.walkforward.runner import (WalkForwardConfig, run_walk_forward,
                                             weighted_mean, weighted_sd)
from scripts_walkforward import (AL_POOLED, SESSION_FEATS, labels_for, uniqueness,
                                 coverage, pool)

warnings.filterwarnings("ignore")
OUT = Path("data/_reports")
Z = 2.128
REASON = {1: "target", -1: "stop", 0: "time", 2: "rollover"}


def stats(res, folds):
    n, w = [], []
    for f in res:
        if f.index not in folds:
            continue
        m = f.taken
        n.append(f.net[m]); w.append(f.w[m])
    n = np.concatenate(n); w = np.concatenate(w)
    e = weighted_mean(n, w); sd = weighted_sd(n, w); eff = float(w.sum())
    return dict(wnet=e, sd=sd, eff=eff, se=sd / math.sqrt(eff), n=int(n.size))


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

    t = labels_for(m15, cal, cfg, lc["stop_atr"], lc["target_atr"])
    net = {s: t[f"label_{s}_ret_atr_net"].to_numpy(zero_copy_only=False)
           for s in ("long", "short")}
    gro = {s: t[f"label_{s}_ret_atr_gross"].to_numpy(zero_copy_only=False)
           for s in ("long", "short")}
    rea = {s: t[f"label_{s}_reason"].to_numpy(zero_copy_only=False).astype(np.int32)
           for s in ("long", "short")}
    ex = t["label_exit_index"].to_numpy(zero_copy_only=False)
    live = np.isfinite(net["long"]) & np.isfinite(net["short"])
    u = uniqueness(ex, live)

    wfc = WalkForwardConfig(embargo_bars=int(wf["embargo_bars"]))
    nb = int(wf["n_blocks"])
    folds0 = expanding_folds(np.where(live, ex, -1).astype(np.int64), n_blocks=nb,
                             embargo_bars=int(wf["embargo_bars"]))
    al, alc = {}, {}
    for f in folds0:
        te = f.test[live[f.test]]
        al[f.index] = weighted_mean(net["long"][te], u[te])
        alc[f.index] = float(np.nanmean(cost[te]))

    def run(**kw):
        r, _ = run_walk_forward(X, names, net, gro, rea, u, ex, cost, cfg=wfc,
                                n_blocks=nb, stop_atr=lc["stop_atr"],
                                target_atr=lc["target_atr"], **kw)
        return r

    print("re-executing headline...", file=sys.stderr)
    res = run()

    # ---- REPRODUCTION GATE ----
    ref = json.loads((OUT / "walkforward_attempt2.json").read_text())
    te14 = np.concatenate([f.test[live[f.test]] for f in folds0 if f.index >= 1])
    al_pooled = weighted_mean(net["long"][te14], u[te14])   # COMPUTED, not the
    # rounded 0.0398 constant: attempt 2 subtracted the computed value, and
    # using the rounded one shifts E_excess by 3.5e-5 and trips the gate.
    p14 = stats(res, (1, 2, 3, 4)); cov = coverage(res)
    E = p14["wnet"] - al_pooled
    checks = [("E_excess", E, ref["E_excess"]), ("coverage", cov, ref["coverage"]),
              ("wnet", p14["wnet"], ref["wnet"]), ("se", p14["se"], ref["se"]),
              ("eff_n", p14["eff"], ref["eff_n"])]
    L: list[str] = []
    P = L.append
    P("=" * 78)
    P("ATTEMPT 2 DECOMPOSITION -- RE-READ, NO ATTEMPT CONSUMED")
    P("=" * 78)
    P("\nREPRODUCTION GATE (this must pass or nothing below is a re-read)")
    bad = []
    for k, got, want in checks:
        ok = abs(got - want) < 1e-9
        bad += [] if ok else [k]
        P(f"  {k:<10} recomputed {got:>12.8f}   attempt 2 {want:>12.8f}   "
          f"{'MATCH' if ok else 'DIFFERS'}")
    if bad:
        P(f"\n  ABORT: {bad} did not reproduce. This is a new run, not a re-read.")
        (OUT / "attempt2_decomposition.txt").write_text("\n".join(L) + "\n")
        print("\n".join(L)); return 1
    P("  -> bit-identical. Everything below re-reads attempt 2.")

    # ---- 1. FOLD-4 CONCENTRATION ----
    P("\n" + "=" * 78)
    P("1. FOLD-4 CONCENTRATION")
    P("=" * 78)
    P(f"\n  {'fold':>4} {'eff n':>8} {'Wnet':>9} {'AL':>9} {'E_excess':>10} "
      f"{'contribution':>13} {'SE':>8}")
    tot_c = tot_n = 0.0
    for f in res:
        if f.index == 0:
            continue
        st = stats(res, (f.index,))
        e = st["wnet"] - al[f.index]
        tot_c += st["eff"] * e; tot_n += st["eff"]
        P(f"  {f.index:>4} {st['eff']:>8.1f} {st['wnet']:>+9.4f} {al[f.index]:>+9.4f} "
          f"{e:>+10.4f} {st['eff']*e:>+13.2f} {st['se']:>8.4f}")
    P(f"  {'ALL':>4} {tot_n:>8.1f} {p14['wnet']:>+9.4f} {al_pooled:>+9.4f} "
      f"{E:>+10.4f} {tot_c:>+13.2f} {p14['se']:>8.4f}")

    s13 = stats(res, (1, 2, 3)); s4 = stats(res, (4,))
    te13 = np.concatenate([f.test[live[f.test]] for f in folds0 if f.index in (1, 2, 3)])
    te4 = np.concatenate([f.test[live[f.test]] for f in folds0 if f.index == 4])
    al13 = weighted_mean(net["long"][te13], u[te13])
    al4 = weighted_mean(net["long"][te4], u[te4])
    P("\n  POOLED, FOLDS 1-3 ONLY")
    P(f"    Wnet          {s13['wnet']:+.4f} ATR")
    P(f"    always-long   {al13:+.4f} ATR")
    P(f"    E_excess      {s13['wnet']-al13:+.4f} ATR")
    P(f"    eff n {s13['eff']:.1f}   sd {s13['sd']:.4f}   SE {s13['se']:.4f}   "
      f"lower bound {s13['wnet']-al13-Z*s13['se']:+.4f}")
    P(f"    E_excess / SE  {(s13['wnet']-al13)/s13['se']:+.3f} sigma")
    P("\n  FOLD 4 ALONE")
    P(f"    Wnet          {s4['wnet']:+.4f} ATR")
    P(f"    always-long   {al4:+.4f} ATR")
    P(f"    E_excess      {s4['wnet']-al4:+.4f} ATR")
    P(f"    eff n {s4['eff']:.1f}   sd {s4['sd']:.4f}   SE {s4['se']:.4f}   "
      f"lower bound {s4['wnet']-al4-Z*s4['se']:+.4f}")
    P(f"    E_excess / SE  {(s4['wnet']-al4)/s4['se']:+.3f} sigma")
    P(f"\n  fold 4 share of the total eff-n-weighted contribution: "
      f"{100*(s4['eff']*(s4['wnet']-al4))/tot_c:.1f}%")

    # ---- 3. COST REGIME PER FOLD ----
    P("\n" + "=" * 78)
    P("3. COST REGIME, PER FOLD (superseding the two-group split)")
    P("=" * 78)
    P(f"\n  {'fold':>4} {'cost c':>9} {'Wnet':>9} {'AL':>9} {'E_excess':>10} {'SE':>8}")
    for f in res:
        if f.index == 0:
            continue
        st = stats(res, (f.index,))
        P(f"  {f.index:>4} {alc[f.index]:>9.4f} {st['wnet']:>+9.4f} "
          f"{al[f.index]:>+9.4f} {st['wnet']-al[f.index]:>+10.4f} {st['se']:>8.4f}")
    P("\n  The attempt 2 report grouped folds 3-4 together at c = 0.1277 and")
    P("  reported +0.1605, which averaged the one dissenting fold with the one")
    P("  driving fold. Per fold the cost ordering and the excess ordering do not")
    P("  line up: folds 1 and 2 sit at the two highest costs with small positive")
    P("  excess, fold 3 at the second-lowest cost is the only negative, and fold")
    P("  4 at the lowest cost carries everything.")

    # ---- 2/4. GUARDS PER FOLD ----
    print("re-executing shuffled control...", file=sys.stderr)
    sh = run(shuffle_train=True)
    print("re-executing ablation...", file=sys.stderr)
    ab = run(drop=SESSION_FEATS)
    P("\n" + "=" * 78)
    P("4. GUARDS, PER FOLD")
    P("=" * 78)
    P(f"\n  {'fold':>4} | {'headline':>9} {'cov':>6} | {'ablated':>9} {'cov':>6} "
      f"{'delta':>9} | {'shuffled':>9} {'cov':>6}")
    for f in res:
        i = f.index
        h = stats(res, (i,)); a = stats(ab, (i,)); s_ = stats(sh, (i,))
        eh, ea, es = h["wnet"]-al[i], a["wnet"]-al[i], s_["wnet"]-al[i]
        ch = coverage(res, (i,)); ca = coverage(ab, (i,)); cs = coverage(sh, (i,))
        P(f"  {i:>4} | {eh:>+9.4f} {ch:>6.1%} | {ea:>+9.4f} {ca:>6.1%} "
          f"{ea-eh:>+9.4f} | {es:>+9.4f} {cs:>6.1%}")
    ah = stats(ab, (1, 2, 3, 4)); shh = stats(sh, (1, 2, 3, 4))
    P(f"  {'1-4':>4} | {E:>+9.4f} {cov:>6.1%} | {ah['wnet']-al_pooled:>+9.4f} "
      f"{coverage(ab):>6.1%} {(ah['wnet']-al_pooled)-E:>+9.4f} | "
      f"{shh['wnet']-al_pooled:>+9.4f} {coverage(sh):>6.1%}")
    a13 = stats(ab, (1, 2, 3)); a4 = stats(ab, (4,))
    P(f"\n  ABLATION DELTA, DECOMPOSED")
    P(f"    folds 1-3  headline {s13['wnet']-al13:+.4f} -> ablated "
      f"{a13['wnet']-al13:+.4f}   delta {(a13['wnet']-al13)-(s13['wnet']-al13):+.4f}")
    P(f"    fold 4     headline {s4['wnet']-al4:+.4f} -> ablated "
      f"{a4['wnet']-al4:+.4f}   delta {(a4['wnet']-al4)-(s4['wnet']-al4):+.4f}")

    txt = "\n".join(L) + "\n"
    (OUT / "attempt2_decomposition.txt").write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
