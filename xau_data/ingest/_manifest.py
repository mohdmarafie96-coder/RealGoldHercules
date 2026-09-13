"""Resume-safety ledger: one row per source file."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Iterable, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from ..storage.parquet_store import write_table
from ..storage.schemas import MANIFEST
from ..timeutils import UTC, require_utc

__all__ = ["Manifest", "ManifestRow", "Status", "SKIP_STATUSES"]

Status = Literal["ok", "empty", "missing", "failed"]

#: Statuses that mean "already handled, do not refetch".
SKIP_STATUSES: Final[frozenset[str]] = frozenset({"ok", "empty"})


@dataclass(frozen=True, slots=True)
class ManifestRow:
    symbol: str
    hour_utc: datetime
    status: Status
    bytes: int
    sha256: str | None
    tick_count: int
    fetched_at: datetime
    error: str | None


class Manifest:
    """Append-mostly ledger persisted as a single Parquet file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._rows: dict[tuple[str, int], ManifestRow] = {}
        if path.exists():
            tbl = pq.read_table(path)
            for r in tbl.to_pylist():
                hour = r["hour_utc"]
                if hour.tzinfo is None:
                    hour = hour.replace(tzinfo=UTC)
                key = (r["symbol"], int(hour.timestamp()))
                self._rows[key] = ManifestRow(
                    r["symbol"], hour, r["status"], r["bytes"], r["sha256"],
                    r["tick_count"], r["fetched_at"], r["error"],
                )

    def __len__(self) -> int:
        return len(self._rows)

    @staticmethod
    def _key(symbol: str, hour: datetime) -> tuple[str, int]:
        require_utc(hour, name="hour_utc")
        return symbol, int(hour.timestamp())

    def should_skip(self, symbol: str, hour: datetime) -> bool:
        row = self._rows.get(self._key(symbol, hour))
        return row is not None and row.status in SKIP_STATUSES

    def record(self, row: ManifestRow) -> None:
        self._rows[self._key(row.symbol, row.hour_utc)] = row

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self._rows.values():
            out[r.status] = out.get(r.status, 0) + 1
        return out

    def rows(self) -> Iterable[ManifestRow]:
        return sorted(self._rows.values(), key=lambda r: (r.symbol, r.hour_utc))

    def flush(self) -> Path:
        rows = list(self.rows())
        tbl = pa.table({
            "symbol": [r.symbol for r in rows],
            "hour_utc": pa.array([r.hour_utc for r in rows], type=MANIFEST.field("hour_utc").type),
            "status": [r.status for r in rows],
            "bytes": pa.array([r.bytes for r in rows], type=pa.int64()),
            "sha256": [r.sha256 for r in rows],
            "tick_count": pa.array([r.tick_count for r in rows], type=pa.int32()),
            "fetched_at": pa.array([r.fetched_at for r in rows], type=MANIFEST.field("fetched_at").type),
            "error": [r.error for r in rows],
        })
        return write_table(tbl, self._path.parent, MANIFEST,
                           filename=self._path.name, sort_by="hour_utc")
