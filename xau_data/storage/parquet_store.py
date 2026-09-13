"""Parquet IO with the declared schema enforced on every write.

Writes are atomic: build to a temporary file in the destination directory, then
rename. An interrupted run never leaves a half-written partition that later
reads as valid.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..errors import SchemaError

__all__ = ["write_table", "read_dataset", "enforce_schema"]

_COMPRESSION = "zstd"


def enforce_schema(table: pa.Table, schema: pa.Schema, *, where: str) -> pa.Table:
    """Cast a table to the declared schema, failing loudly on any mismatch."""
    missing = [f.name for f in schema if f.name not in table.column_names]
    if missing:
        raise SchemaError(f"{where}: missing columns {missing}")
    extra = [n for n in table.column_names if n not in schema.names]
    if extra:
        raise SchemaError(f"{where}: unexpected columns {extra}")

    # Arrow will happily cast a NAIVE timestamp column to timestamp[us, tz=UTC],
    # silently reinterpreting local wall-clock as UTC. That is precisely the bug
    # this package exists to prevent, so it is rejected rather than cast.
    for f in schema:
        if not pa.types.is_timestamp(f.type):
            continue
        got = table.schema.field(f.name).type
        if pa.types.is_timestamp(got) and got.tz is None:
            raise SchemaError(
                f"{where}: column {f.name!r} is a NAIVE timestamp. Localise it to "
                "UTC explicitly; implicit reinterpretation is not allowed."
            )
        if not pa.types.is_timestamp(got) and not pa.types.is_integer(got):
            raise SchemaError(
                f"{where}: column {f.name!r} must be a UTC timestamp, got {got}"
            )
    try:
        return table.select(schema.names).cast(schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
        raise SchemaError(f"{where}: cannot cast to declared schema: {exc}") from exc


def write_table(
    table: pa.Table,
    directory: Path,
    schema: pa.Schema,
    *,
    filename: str = "part-0.parquet",
    sort_by: str | None = None,
) -> Path:
    """Write one partition file atomically, with the schema enforced."""
    table = enforce_schema(table, schema, where=str(directory))
    if sort_by is not None:
        table = table.sort_by([(sort_by, "ascending")])
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / filename
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".parquet.tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression=_COMPRESSION, version="2.6",
                       coerce_timestamps="us", allow_truncated_timestamps=False)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def read_dataset(base: Path, schema: pa.Schema | None = None) -> pa.Table:
    """Read a partitioned tree. Returns an empty table if nothing is there."""
    if not base.exists():
        return pa.table({f.name: pa.array([], type=f.type) for f in schema}) if schema \
            else pa.table({})
    dataset = ds.dataset(base, format="parquet", partitioning="hive")
    return dataset.to_table()
