"""Coverage, correlation and catalogue for the feature set."""
from __future__ import annotations

import json, warnings
from pathlib import Path
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

from xau_data.config import load_config
from xau_data.storage.store import open_store
from xau_data.storage.parquet_store import write_table
from xau_data.transform.resample import resample_m1
from xau_data.transform.sessions import SessionCalendar
from xau_backtest.features.build import build_features, feature_schema

warnings.filterwarnings("ignore", message="Mean of empty slice")

GROUP_OF = {
    "atr_pct":"volatility","atr_ratio_14_50":"volatility","atr_ratio_14_200":"volatility",
    "atr_ratio_50_200":"volatility","gk_vol_14":"volatility","parkinson_14":"volatility",
    "gk_to_rv_ratio":"volatility","rv_20":"volatility","rv_vs_median_500":"volatility",
    "vol_of_vol_100":"volatility",
    "ema_dist_20_atr":"trend","ema_dist_50_atr":"trend","ema_dist_200_atr":"trend",
    "ema_slope_20_atr":"trend","ema_slope_50_atr":"trend","ema_slope_200_atr":"trend",
    "ema_stack":"trend","adx_14":"trend","di_diff_14":"trend","donchian_pos_20":"trend",
    "donchian_pos_96":"trend","donchian_width_20_atr":"trend",
    "mtf_trend_agreement":"trend","h1_ema_slope_atr":"trend","h4_ema_slope_atr":"trend",
    "rsi_14":"momentum","rsi_50":"momentum","roc_atr_4":"momentum","roc_atr_16":"momentum",
    "roc_atr_48":"momentum","roc_atr_96":"momentum","ret_z_20":"momentum",
    "vwap_dist_atr":"momentum","twap_dist_atr":"momentum",
    "spread_rel_median_500":"microstructure","spread_z_500":"microstructure",
    "spread_to_atr":"microstructure","volume_rel_median_500":"microstructure",
    "volume_z_500":"microstructure","bar_range_rel_median_500":"microstructure",
    "vol_imbalance":"microstructure","bar_completeness":"microstructure",
    "session_asian":"session","session_london":"session","session_overlap":"session",
    "session_ny":"session","session_progress":"session","mins_to_rollover":"session",
    "mins_since_rollover":"session","dow_sin":"session","dow_cos":"session",
    "tod_sin":"session","tod_cos":"session",
    "cost_to_atr_ratio":"regime","rv_pct_rank_1y":"regime","atr_pct_rank_1y":"regime",
    "spread_pct_rank_1y":"regime","range_pct_rank_1y":"regime","asian_range_atr":"regime",
}


def main() -> int:
    cfg = load_config(); cal = SessionCalendar(cfg.sessions)
    m15 = resample_m1(open_store(cfg.data_root).read_bars_m1(cfg.symbol), timeframe="M15")
    f = build_features(m15, calendar=cal, slippage_per_side=cfg.costs.slippage_per_side)
    names = list(feature_schema(f))
    n = f.num_rows
    rep = cfg.data_root / "_reports"; rep.mkdir(parents=True, exist_ok=True)

    # ---- coverage ----
    cov = []
    for c in names:
        v = f[c].to_numpy(zero_copy_only=False)
        m = np.isnan(v)
        first = int(np.argmax(~m)) if (~m).any() else -1
        cov.append({"feature": c, "group": GROUP_OF.get(c, "?"),
                    "null_pct": float(m.mean()*100), "warmup_bars": first,
                    "min": float(np.nanmin(v)) if (~m).any() else None,
                    "max": float(np.nanmax(v)) if (~m).any() else None,
                    "mean": float(np.nanmean(v)) if (~m).any() else None})
    (rep/"feature_coverage.json").write_text(json.dumps(cov, indent=2))

    # ---- correlation ----
    M = np.vstack([f[c].to_numpy(zero_copy_only=False) for c in names])
    ok = np.all(np.isfinite(M), axis=0)
    C = np.corrcoef(M[:, ok])
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            r = C[i, j]
            if np.isfinite(r) and abs(r) >= 0.9:
                pairs.append({"a": names[i], "b": names[j], "r": float(r)})
    pairs.sort(key=lambda d: -abs(d["r"]))
    (rep/"feature_correlation.json").write_text(json.dumps(
        {"rows_used": int(ok.sum()), "pairs_above_0.9": pairs}, indent=2))

    # ---- catalogue ----
    L = ["# Feature catalogue", "",
         f"Generated from code. {len(names)} features over {n:,} M15 bars, "
         f"{ok.sum():,} rows complete on every feature.", "",
         "| # | feature | group | null % | warm-up bars | min | mean | max |",
         "|---:|---|---|---:|---:|---:|---:|---:|"]
    for k, r in enumerate(sorted(cov, key=lambda d: (d["group"], d["feature"])), 1):
        fmt = lambda x: "—" if x is None else f"{x:,.4f}"
        L.append(f"| {k} | `{r['feature']}` | {r['group']} | {r['null_pct']:.2f} | "
                 f"{r['warmup_bars']:,} | {fmt(r['min'])} | {fmt(r['mean'])} | {fmt(r['max'])} |")
    L += ["", "## Correlated pairs at or above 0.9", "",
          "Flagged only. No selection happens here; that belongs inside "
          "walk-forward folds in phase 7.", "",
          "| a | b | r |", "|---|---|---:|"]
    for p_ in pairs:
        L.append(f"| `{p_['a']}` | `{p_['b']}` | {p_['r']:+.4f} |")
    if not pairs:
        L.append("| _none_ | | |")
    (Path("docs")/"FEATURE_CATALOGUE.md").write_text("\n".join(L))

    # ---- materialise, partitioned year/month ----
    ts = f["ts_open"].to_pylist()
    months = sorted({(t.year, t.month) for t in ts})
    import pyarrow.compute as pc
    base = cfg.data_root/"features_m15"/f"symbol={cfg.symbol}"
    written = 0
    typ = f.schema.field("ts_open").type
    import datetime as dt
    for y, mo in months:
        lo = pa.scalar(dt.datetime(y, mo, 1, tzinfo=dt.timezone.utc), type=typ)
        hi = pa.scalar(dt.datetime(y+(mo==12), (mo%12)+1, 1, tzinfo=dt.timezone.utc), type=typ)
        part = f.filter(pc.and_(pc.greater_equal(f["ts_open"], lo), pc.less(f["ts_open"], hi)))
        if part.num_rows == 0: continue
        d = base/f"year={y:04d}"/f"month={mo:02d}"; d.mkdir(parents=True, exist_ok=True)
        pq.write_table(part, d/"part-0.parquet", compression="zstd",
                       coerce_timestamps="us")
        written += part.num_rows

    print(f"features {len(names)}  rows {n:,}  complete rows {ok.sum():,}")
    print(f"pairs |r|>=0.9: {len(pairs)}")
    for p_ in pairs[:12]:
        print(f"   {p_['r']:+.4f}  {p_['a']}  ~  {p_['b']}")
    print(f"materialised {written:,} rows to {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
