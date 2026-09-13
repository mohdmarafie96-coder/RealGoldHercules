"""Persistence that works identically against local disk and Google Cloud Storage.

The destination is a single URI, taken from configuration or the environment:

    /var/data/xau                 local directory
    gs://my-bucket/xau            Google Cloud Storage

Resume is decided by asking the STORE what already exists, never by looking at
local disk. That is what lets the job be killed, moved to another machine, and
restarted without refetching.

Credentials are never hardcoded. GCS uses Application Default Credentials, so
the service-account key is located by GOOGLE_APPLICATION_CREDENTIALS, or by the
VM's attached service account when running on Compute Engine.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Final, Iterable, Protocol

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from ..errors import ConfigError
from .schemas import BARS_M1, CANDLE_DAY

__all__ = ["Store", "open_store", "GCS_URI_ENV", "CREDENTIALS_ENV"]

GCS_URI_ENV: Final[str] = "XAU_DATA_URI"
CREDENTIALS_ENV: Final[str] = "GOOGLE_APPLICATION_CREDENTIALS"
_COMPRESSION: Final[str] = "zstd"


def _split(uri: str) -> tuple[pafs.FileSystem, str]:
    """Return (filesystem, path) for a local path or a gs:// URI."""
    if uri.startswith(("gs://", "gcs://")):
        bucket_and_path = uri.split("://", 1)[1]
        if not bucket_and_path.strip("/"):
            raise ConfigError(f"{uri!r} names no bucket")
        # Application Default Credentials; never an inline key.
        fs = pafs.GcsFileSystem(anonymous=False)
        return fs, bucket_and_path.rstrip("/")
    p = Path(uri).expanduser().resolve()
    return pafs.LocalFileSystem(), str(p)


class Store:
    """Parquet store partitioned year/month, on local disk or GCS."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._fs, self._root = _split(uri)

    # ---- layout ------------------------------------------------------

    def bars_m1_dir(self, symbol: str) -> str:
        return f"{self._root}/bars_m1/symbol={symbol}"

    def bars_m1_partition(self, symbol: str, year: int, month: int) -> str:
        return f"{self.bars_m1_dir(symbol)}/year={year:04d}/month={month:02d}"

    def candle_days_path(self, symbol: str) -> str:
        return f"{self._root}/_manifests/candle_days/symbol={symbol}"

    # ---- helpers -----------------------------------------------------

    def _ls(self, path: str) -> list[pafs.FileInfo]:
        try:
            sel = pafs.FileSelector(path, recursive=True, allow_not_found=True)
            return self._fs.get_file_info(sel)
        except (FileNotFoundError, OSError):
            return []

    def exists(self, path: str) -> bool:
        info = self._fs.get_file_info(path)
        return info.type != pafs.FileType.NotFound

    def _write(self, table: pa.Table, path: str, schema: pa.Schema,
               sort_by: str) -> str:
        table = table.select(schema.names).cast(schema)
        if sort_by:
            table = table.sort_by([(sort_by, "ascending")])
        parent = path.rsplit("/", 1)[0]
        self._fs.create_dir(parent, recursive=True)
        tmp = f"{path}.tmp"
        with self._fs.open_output_stream(tmp) as sink:
            pq.write_table(table, sink, compression=_COMPRESSION, version="2.6",
                           coerce_timestamps="us", allow_truncated_timestamps=False)
        # object stores have no atomic rename; delete-then-move is the best
        # available, and the manifest is what makes resume correct anyway
        if self.exists(path):
            self._fs.delete_file(path)
        self._fs.move(tmp, path)
        return path

    # ---- reads -------------------------------------------------------

    def completed_days(self, symbol: str) -> set[date]:
        """Days already ingested, according to the manifest in the store."""
        path = self.candle_days_path(symbol)
        files = [f.path for f in self._ls(path) if f.path.endswith(".parquet")]
        if not files:
            return set()
        out: set[date] = set()
        for f in files:
            try:
                tbl = pq.read_table(f, filesystem=self._fs, columns=["day_utc"])
            except (OSError, pa.ArrowInvalid):
                continue
            for v in tbl["day_utc"].to_pylist():
                if v is not None:
                    out.add(v.date() if isinstance(v, datetime) else v)
        return out

    def read_bars_m1(self, symbol: str) -> pa.Table:
        path = self.bars_m1_dir(symbol)
        if not self._ls(path):
            return BARS_M1.empty_table()
        dataset = ds.dataset(path, filesystem=self._fs, format="parquet",
                             partitioning="hive")
        tbl = dataset.to_table()
        keep = [n for n in BARS_M1.names if n in tbl.column_names]
        return tbl.select(keep).cast(pa.schema([BARS_M1.field(n) for n in keep]))

    def read_candle_days(self, symbol: str) -> pa.Table:
        path = self.candle_days_path(symbol)
        if not self._ls(path):
            return CANDLE_DAY.empty_table()
        return ds.dataset(path, filesystem=self._fs, format="parquet").to_table()

    # ---- writes ------------------------------------------------------

    def append_bars_m1(self, symbol: str, year: int, month: int,
                       table: pa.Table) -> str:
        """Merge `table` into the month partition, de-duplicating on ts_open."""
        part = self.bars_m1_partition(symbol, year, month)
        dest = f"{part}/part-0.parquet"
        if self.exists(dest):
            existing = pq.read_table(dest, filesystem=self._fs)
            table = pa.concat_tables([existing.cast(BARS_M1), table.cast(BARS_M1)])
            seen: set = set()
            keep: list[int] = []
            for i, ts in enumerate(table["ts_open"].to_pylist()):
                if ts not in seen:
                    seen.add(ts)
                    keep.append(i)
            table = table.take(keep)
        return self._write(table, dest, BARS_M1, "ts_open")

    def append_candle_days(self, symbol: str, table: pa.Table) -> str:
        dest = f"{self.candle_days_path(symbol)}/part-0.parquet"
        if self.exists(dest):
            existing = pq.read_table(dest, filesystem=self._fs)
            table = pa.concat_tables([existing.cast(CANDLE_DAY), table.cast(CANDLE_DAY)])
            seen: set = set()
            keep: list[int] = []
            for i, d in enumerate(table["day_utc"].to_pylist()):
                if d not in seen:
                    seen.add(d)
                    keep.append(i)
            table = table.take(keep)
        return self._write(table, dest, CANDLE_DAY, "day_utc")


def open_store(default_uri: str | os.PathLike[str]) -> Store:
    """Open the store named by XAU_DATA_URI, falling back to ``default_uri``.

    Setting XAU_DATA_URI to a gs:// URI is the only change needed to persist
    outside the machine. Credentials come from GOOGLE_APPLICATION_CREDENTIALS
    or the VM's attached service account.
    """
    uri = os.environ.get(GCS_URI_ENV) or str(default_uri)
    if uri.startswith(("gs://", "gcs://")):
        if not os.environ.get(CREDENTIALS_ENV) and not os.environ.get(
            "GCE_METADATA_HOST", ""
        ):
            # not fatal: on GCE the attached service account is used instead
            pass
    return Store(uri)
