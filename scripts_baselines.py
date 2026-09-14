"""Phase 4 baselines, reported in ATR units as well as dollars.

Dollar expectancy cannot separate a worse rule from a wider regime: a breakout
taking 3x larger stops in a 3x wider market loses 3x more dollars while being
the same rule. Every figure here is therefore also normalised by the causal ATR
recorded at entry.
"""
from __future__ import annotations

import datetime as dt, json
import numpy as np

from xau_data.config import load_config
from xau_data.storage.store import open_store
from xau_data.transform.resample import resample_m1
from xau_data.transform.sessions import SessionCalendar
from xau_backtest.data.source import COLUMNS, TableSource
from xau_backtest.engine.broker import SlippageModel, SwapModel
from xau_backtest.engine.loop import run_backtest
from xau_backtest.engine.strategy import (
    AsianRangeFadeStrategy, LondonBreakoutStrategy, RandomStrategy)

UTC = dt.timezone.utc


def stats(trades):
    d = np.array([t.net_pnl / t.size for t in trades])
    a = np.array([t.net_pnl_atr for t in trades if t.net_pnl_atr is not None])
    atr = np.array([t.entry_atr for t in trades if t.entry_atr])
    out = {"n": len(d), "usd": float(d.mean()) if len(d) else float("nan"),
           "atr_units": float(a.mean()) if len(a) else float("nan"),
           "mean_atr": float(atr.mean()) if len(atr) else float("nan"),
           "win": float((d > 0).mean()) if len(d) else float("nan")}
    if len(a) > 1:
        out["atr_se"] = float(a.std(ddof=1) / np.sqrt(len(a)))
    return out


def by_year(trades):
    buckets = {}
    for t in trades:
        buckets.setdefault(t.entry_ts.year, []).append(t)
    return {y: stats(v) for y, v in sorted(buckets.items())}


def show(name, trades):
    s = stats(trades)
    print()
    print("=" * 78)
    print(f"{name}   n={s['n']:,}  expectancy {s['usd']:+.4f} USD/unit  "
          f"{s['atr_units']:+.4f} ATR  (mean ATR {s['mean_atr']:.2f})")
    print("=" * 78)
    print(f"{'year':>6} {'trades':>7} {'USD/unit':>10} {'ATR units':>11} "
          f"{'+/-':>8} {'mean ATR':>9} {'win%':>6}")
    rows = by_year(trades)
    for y, v in rows.items():
        se = v.get("atr_se", float("nan"))
        print(f"{y:>6} {v['n']:>7} {v['usd']:>+10.4f} {v['atr_units']:>+11.4f} "
              f"{1.96*se:>8.4f} {v['mean_atr']:>9.2f} {v['win']*100:>5.1f}%")
    u = [v["usd"] for v in rows.values()]
    a = [v["atr_units"] for v in rows.values()]
    print(f"{'spread':>6} {'':>7} {max(u)-min(u):>10.4f} {max(a)-min(a):>11.4f}"
          "    <- year-to-year range")
    return {"overall": s, "by_year": rows}


def main() -> int:
    cfg = load_config(); cal = SessionCalendar(cfg.sessions)
    m15 = resample_m1(open_store(cfg.data_root).read_bars_m1(cfg.symbol), timeframe="M15")
    src = TableSource(m15)
    slip = SlippageModel(per_side=cfg.costs.slippage_per_side)
    print(f"M15 bars {len(src):,}   slippage/side {cfg.costs.slippage_per_side}")
    out = {}

    pooled = []
    for seed in range(8):
        r = run_backtest(src, RandomStrategy(np.random.default_rng(seed),
                         entry_probability=0.02, max_bars=32),
                         calendar=cal, period_seconds=900, slippage=slip,
                         swap=SwapModel(), columns=COLUMNS)
        pooled += r.trades
    out["random"] = show("BASELINE 1 - RandomStrategy (control, 8 seeds pooled)", pooled)

    r = run_backtest(src, LondonBreakoutStrategy(), calendar=cal, period_seconds=900,
                     slippage=slip, swap=SwapModel(), columns=COLUMNS)
    out["london_breakout"] = show("BASELINE 2 - LondonBreakout (momentum)", r.trades)

    r2 = run_backtest(src, AsianRangeFadeStrategy(), calendar=cal, period_seconds=900,
                      slippage=slip, swap=SwapModel(), columns=COLUMNS)
    out["asian_fade"] = show("BASELINE 3 - AsianRangeFade (mean reversion)", r2.trades)

    (cfg.data_root / "_reports").mkdir(parents=True, exist_ok=True)
    (cfg.data_root / "_reports" / "baselines_atr.json").write_text(
        json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
