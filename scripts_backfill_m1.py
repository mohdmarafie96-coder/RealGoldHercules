"""Unattended M1 candle backfill. Safe to kill and restart: resume is decided
by the store, so nothing already persisted is refetched."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime, timezone

from xau_data.config import load_config
from xau_data.ingest.dukascopy_candles import ingest_days
from xau_data.storage.store import open_store

START = date.fromisoformat(os.environ.get("XAU_START", "2021-01-01"))
END = date.fromisoformat(os.environ.get("XAU_END", datetime.now(timezone.utc).date().isoformat()))

def main() -> int:
    cfg = load_config()
    store = open_store(cfg.data_root)
    print(f"store={store.uri}", flush=True)
    print(f"range={START}..{END}", flush=True)
    done = store.completed_days(cfg.symbol)
    print(f"already complete: {len(done)} days", flush=True)
    t0 = time.time()
    s = ingest_days(cfg, START, END, store=store, progress_every=10, flush_every_days=10)
    el = time.time() - t0
    out = {
        "considered": s.days_considered, "written": s.days_written,
        "skipped": s.days_skipped, "failed": s.days_failed,
        "rows": s.rows, "stripped": s.stripped,
        "elapsed_s": round(el), "days_per_hour": round(s.days_written / max(el, 1) * 3600, 1),
    }
    print("BACKFILL_M1_DONE " + json.dumps(out), flush=True)
    prog = cfg.data_root / "_m1_backfill_result.json"
    prog.parent.mkdir(parents=True, exist_ok=True)
    prog.write_text(json.dumps({**out, "failures": s.failures[:50]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
