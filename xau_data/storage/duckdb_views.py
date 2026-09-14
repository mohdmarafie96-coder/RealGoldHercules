"""DuckDB views over the Parquet tree.

DuckDB preserves TIMESTAMP WITH TIME ZONE from Parquet, so the UTC invariant
survives into SQL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import duckdb

__all__ = ["build_views", "connect"]


def connect(db_path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)


def build_views(
    db_path: Path,
    data_root: Path,
    timeframes: Sequence[str],
) -> list[str]:
    """(Re)create every view. Returns the view names created."""
    con = connect(db_path)
    made: list[str] = []
    try:
        con.execute("SET TimeZone='UTC'")

        def has(glob: str) -> bool:
            return any(Path(data_root).glob(glob))

        if has("ticks/**/*.parquet"):
            con.execute(f"""
                CREATE OR REPLACE VIEW ticks AS
                SELECT *, ask - bid AS spread
                FROM read_parquet('{data_root}/ticks/**/*.parquet',
                                  hive_partitioning := true)
            """)
            made.append("ticks")

        for tf in timeframes:
            if not has(f"bars/**/timeframe={tf}/**/*.parquet"):
                continue
            name = f"bars_{tf.lower()}"
            con.execute(f"""
                CREATE OR REPLACE VIEW {name} AS
                SELECT * FROM read_parquet(
                    '{data_root}/bars/symbol=*/timeframe={tf}/**/*.parquet',
                    hive_partitioning := true)
            """)
            made.append(name)

        for tf in timeframes:
            if not has(f"features/**/timeframe={tf}/**/*.parquet"):
                continue
            name = f"features_{tf.lower()}"
            con.execute(f"""
                CREATE OR REPLACE VIEW {name} AS
                SELECT * FROM read_parquet(
                    '{data_root}/features/symbol=*/timeframe={tf}/**/*.parquet',
                    hive_partitioning := true)
            """)
            made.append(name)

        if has("features_m15/**/*.parquet"):
            con.execute(f"""
                CREATE OR REPLACE VIEW ml_features_m15 AS
                SELECT * FROM read_parquet(
                    '{data_root}/features_m15/symbol=*/**/*.parquet',
                    hive_partitioning := true)
            """)
            made.append("ml_features_m15")

        if has("context/**/*.parquet"):
            con.execute(f"""
                CREATE OR REPLACE VIEW context AS
                SELECT * FROM read_parquet('{data_root}/context/**/*.parquet',
                                           hive_partitioning := true)
            """)
            made.append("context")

        if has("calendar/**/*.parquet"):
            con.execute(f"""
                CREATE OR REPLACE VIEW calendar AS
                SELECT * FROM read_parquet('{data_root}/calendar/**/*.parquet',
                                           hive_partitioning := true)
            """)
            made.append("calendar")

        con.commit()
    finally:
        con.close()
    return made
