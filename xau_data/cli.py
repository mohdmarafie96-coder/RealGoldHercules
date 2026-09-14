"""Command line entrypoints: ingest, build, views, validate."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .config import Config, load_config
from .ingest.dukascopy import ingest_range
from .storage.duckdb_views import build_views
from .storage.parquet_store import write_table
from .storage.paths import bars_partition, ticks_partition
from .storage.schemas import BARS, TIMEFRAME_SECONDS
from .timeutils import UTC, month_range
from .transform.bars import bars_from_table
from .transform.features import add_cost_features, FEATURES
from .quality.validate import validate_bars, write_report


def _month_bounds(y: int, m: int) -> tuple[datetime, datetime]:
    start = datetime(y, m, 1, tzinfo=UTC)
    end = datetime(y + (m == 12), (m % 12) + 1, 1, tzinfo=UTC)
    return start, end


def cmd_ingest(cfg: Config, args: argparse.Namespace) -> int:
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    s = ingest_range(cfg, start, end)
    print(f"hours={s.hours_considered} fetched={s.fetched} skipped={s.skipped} "
          f"empty={s.empty} missing={s.missing} failed={s.failed} "
          f"ticks={s.ticks:,} months_written={s.months_written}")
    for f in s.failures[:20]:
        print(f"  FAILED {f}")
    return 1 if s.failed else 0


def cmd_build(cfg: Config, args: argparse.Namespace) -> int:
    """Ticks to bars for every configured timeframe, per month partition."""
    months = sorted({
        (int(p.parent.name.split("=")[1]), int(p.name.split("=")[1]))
        for p in (cfg.ticks_dir / f"symbol={cfg.symbol}").glob("year=*/month=*")
    })
    if not months:
        print("no tick partitions found", file=sys.stderr)
        return 1
    total = 0
    for y, m in months:
        part = ticks_partition(cfg.ticks_dir, cfg.symbol, y, m)
        ticks = pq.read_table(part)
        for tf in cfg.timeframes:
            bars = bars_from_table(ticks, timeframe=tf)
            if bars.num_rows == 0:
                continue
            write_table(bars, bars_partition(cfg.bars_dir, cfg.symbol, tf, y, m),
                        BARS, sort_by="ts_open")
            total += bars.num_rows
            print(f"  {y}-{m:02d} {tf:>3}: {bars.num_rows:,} bars "
                  f"from {ticks.num_rows:,} ticks")
    print(f"wrote {total:,} bars across {len(months)} month(s)")
    return 0


def cmd_features(cfg: Config, args: argparse.Namespace) -> int:
    """Bars -> feature-ready dataset with causal cost and volatility columns."""
    import pyarrow.dataset as ds
    base = cfg.bars_dir / f"symbol={cfg.symbol}" / f"timeframe={args.timeframe}"
    if not base.exists():
        print(f"no bars for {args.timeframe}; run `build` first", file=sys.stderr)
        return 1
    tbl = ds.dataset(base, format="parquet", partitioning="hive").to_table()
    tbl = tbl.select([f.name for f in BARS]).cast(BARS).sort_by([("ts_open", "ascending")])
    slip = args.slippage if args.slippage is not None else cfg.costs.slippage_per_side
    feats = add_cost_features(tbl, slippage_per_side=slip,
                              atr_period=args.atr_period)
    out_base = cfg.data_root / "features" / f"symbol={cfg.symbol}" / f"timeframe={args.timeframe}"
    months = sorted({(d.year, d.month) for d in feats["ts_open"].to_pylist()})
    import pyarrow.compute as pcx
    n = 0
    for y, m in months:
        lo = datetime(y, m, 1, tzinfo=UTC)
        hi = datetime(y + (m == 12), (m % 12) + 1, 1, tzinfo=UTC)
        mask = pcx.and_(pcx.greater_equal(feats["ts_open"], pa.scalar(lo, type=feats.schema.field("ts_open").type)),
                        pcx.less(feats["ts_open"], pa.scalar(hi, type=feats.schema.field("ts_open").type)))
        part = feats.filter(mask)
        if part.num_rows == 0:
            continue
        write_table(part, out_base / f"year={y:04d}" / f"month={m:02d}",
                    FEATURES, sort_by="ts_open")
        n += part.num_rows
    print(f"wrote {n:,} feature rows to {out_base}")
    return 0


def cmd_views(cfg: Config, args: argparse.Namespace) -> int:
    made = build_views(cfg.duckdb_path, cfg.data_root, cfg.timeframes)
    print(f"duckdb: {cfg.duckdb_path}")
    print(f"views: {', '.join(made) if made else '(none — no parquet found)'}")
    return 0


def cmd_validate(cfg: Config, args: argparse.Namespace) -> int:
    import pyarrow.dataset as ds
    tables: dict[str, pa.Table] = {}
    for tf in cfg.timeframes:
        base = cfg.bars_dir / f"symbol={cfg.symbol}" / f"timeframe={tf}"
        if not base.exists():
            continue
        tables[tf] = ds.dataset(base, format="parquet", partitioning="hive").to_table()
    if not tables:
        print("no bar partitions found; run `build` first", file=sys.stderr)
        return 1
    ref = tables[next(iter(tables))]
    cov = {
        "timeframes": ", ".join(tables),
        "bars per timeframe": ", ".join(f"{k}={v.num_rows:,}" for k, v in tables.items()),
        "first bar open (UTC)": str(pa.compute.min(ref["ts_open"]).as_py()),
        "last bar open (UTC)": str(pa.compute.max(ref["ts_open"]).as_py()),
        "total ticks in bars": f"{pa.compute.sum(ref['tick_count']).as_py():,}",
    }
    rep = validate_bars(cfg, tables, coverage=cov)
    md, js = write_report(cfg, rep, args.label)
    print(f"errors={rep.errors} warnings={rep.warnings}")
    print(f"markdown: {md}")
    print(f"json:     {js}")
    fail_on = cfg.quality.get("fail_on", "error")
    return 1 if (fail_on == "error" and rep.errors) else 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="xau-data")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--sessions", default="configs/sessions.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="download and persist ticks")
    pi.add_argument("--start", required=True, help="ISO datetime, UTC")
    pi.add_argument("--end", required=True, help="ISO datetime, UTC, exclusive")
    pi.set_defaults(fn=cmd_ingest)

    pb = sub.add_parser("build", help="ticks -> bars")
    pb.set_defaults(fn=cmd_build)

    pf = sub.add_parser("features", help="bars -> feature-ready dataset")
    pf.add_argument("--timeframe", default="M15")
    pf.add_argument("--slippage", type=float, default=None,
                    help="per side USD/oz; default comes from configs costs.slippage_per_side")
    pf.add_argument("--atr-period", type=int, default=14, dest="atr_period")
    pf.set_defaults(fn=cmd_features)

    pv = sub.add_parser("views", help="create DuckDB views")
    pv.set_defaults(fn=cmd_views)

    pq_ = sub.add_parser("validate", help="run quality checks and write the report")
    pq_.add_argument("--label", default="report")
    pq_.set_defaults(fn=cmd_validate)

    args = p.parse_args(argv)
    cfg = load_config(args.config, args.sessions)
    return int(args.fn(cfg, args))


if __name__ == "__main__":
    raise SystemExit(main())
