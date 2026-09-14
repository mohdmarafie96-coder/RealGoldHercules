"""Phase 4 baselines: RandomStrategy and LondonBreakout over the full M1 history."""
from __future__ import annotations

import datetime as dt, json, sys
import numpy as np

from xau_data.config import load_config
from xau_data.storage.store import open_store
from xau_data.transform.resample import resample_m1
from xau_data.transform.sessions import SessionCalendar
from xau_backtest.data.source import COLUMNS, TableSource
from xau_backtest.engine.broker import SlippageModel, SwapModel
from xau_backtest.engine.loop import run_backtest
from xau_backtest.engine.strategy import LondonBreakoutStrategy, RandomStrategy
from xau_backtest.reporting.metrics import by_session, by_year, compute_metrics, render_metrics

UTC = dt.timezone.utc


def main() -> int:
    cfg = load_config(); cal = SessionCalendar(cfg.sessions)
    m15 = resample_m1(open_store(cfg.data_root).read_bars_m1(cfg.symbol), timeframe="M15")
    src = TableSource(m15)
    print(f"M15 bars {len(src):,}  slippage/side {cfg.costs.slippage_per_side}")
    slip = SlippageModel(per_side=cfg.costs.slippage_per_side)
    out = {}

    # 1. Random, pooled over seeds. The control: must lose about the cost.
    nets, per_seed = [], []
    for seed in range(8):
        r = run_backtest(src, RandomStrategy(np.random.default_rng(seed),
                         entry_probability=0.02, max_bars=32),
                         calendar=cal, period_seconds=900, slippage=slip,
                         swap=SwapModel(), columns=COLUMNS)
        n = np.array([t.net_pnl / t.size for t in r.trades])
        nets.append(n); per_seed.append(float(n.mean()))
        if seed == 0:
            m0, r0 = compute_metrics(r.trades, r.equity_ts, r.equity, r.starting_cash), r
    allnet = np.concatenate(nets)
    se = allnet.std(ddof=1) / np.sqrt(len(allnet))
    print()
    print("=" * 66)
    print("BASELINE 1 — RandomStrategy (the control)")
    print("=" * 66)
    print(f"  seeds 8, pooled trades {len(allnet):,}")
    print(f"  expectancy/unit {allnet.mean():+.4f}  95% CI "
          f"[{allnet.mean()-1.96*se:+.4f}, {allnet.mean()+1.96*se:+.4f}]")
    print(f"  per-seed spread {min(per_seed):+.4f} .. {max(per_seed):+.4f}")
    print()
    print(render_metrics(m0, "RandomStrategy, seed 0"))
    out["random"] = {"pooled_trades": len(allnet), "expectancy_per_unit": float(allnet.mean()),
                     "ci": [float(allnet.mean()-1.96*se), float(allnet.mean()+1.96*se)],
                     "per_seed": per_seed}

    # 2. London breakout
    r = run_backtest(src, LondonBreakoutStrategy(), calendar=cal,
                     period_seconds=900, slippage=slip, swap=SwapModel(),
                     columns=COLUMNS)
    m = compute_metrics(r.trades, r.equity_ts, r.equity, r.starting_cash)
    print()
    print("=" * 66)
    print("BASELINE 2 — LondonBreakoutStrategy")
    print("=" * 66)
    print(render_metrics(m, "LondonBreakout, full history"))
    print()
    print("  by year:")
    for k, v in by_year(r.trades).items():
        print(f"    {k}  trades {v['trades']:>5}  expectancy {v['expectancy']:+.4f}  "
              f"net {v['net_pnl']:+10.2f}  win {v['win_rate']*100:4.1f}%")
    print("  by session:")
    for k, v in by_session(r.trades).items():
        print(f"    {k:<8} trades {v['trades']:>5}  expectancy {v['expectancy']:+.4f}")
    out["london"] = m.as_dict()
    out["london_by_year"] = by_year(r.trades)

    (cfg.data_root / "_reports").mkdir(parents=True, exist_ok=True)
    (cfg.data_root / "_reports" / "baselines.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
