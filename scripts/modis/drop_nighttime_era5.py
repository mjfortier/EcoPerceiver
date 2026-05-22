#!/usr/bin/env python3
"""Drop nighttime rows from the ERA5/ec_data SQLite table.

Rows with incoming shortwave radiation below the requested W m-2 threshold are
treated as nighttime rows where GPP would be forced to zero. This script assumes
the radiation column has already been min/max normalized with EcoPerceiver's
DEFAULT_NORM values, which is how the local ERA5 SQLite databases are built.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecoperceiver.constants import DEFAULT_NORM

DEFAULT_DB_PATH = Path("/home/l/luislara/links/projects/aip-pal/luislara/ep/data/era5.db")
DEFAULT_TABLE = "ec_data"
DEFAULT_RADIATION_COLUMN = "SW_IN"
DEFAULT_THRESHOLD_W_M2 = 2.0


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop rows from ec_data where solar radiation is below a daytime "
            "threshold. By default this removes nighttime rows with "
            "SW_IN < 2 W m-2."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database to edit. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Table to filter. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "--radiation-column",
        default=DEFAULT_RADIATION_COLUMN,
        help=f"Solar radiation column. Default: {DEFAULT_RADIATION_COLUMN}",
    )
    parser.add_argument(
        "--threshold-w-m2",
        type=float,
        default=DEFAULT_THRESHOLD_W_M2,
        help=f"Daytime threshold in W m-2. Default: {DEFAULT_THRESHOLD_W_M2}",
    )
    parser.add_argument(
        "--drop-missing-radiation",
        action="store_true",
        help="Also drop rows where the radiation column is NULL.",
    )
    parser.add_argument(
        "--no-rebuild-ids",
        action="store_true",
        help=(
            "Do not rebuild ec_data ids after deletion. Rebuilding is the "
            "default because ERA5Dataset expects contiguous id windows."
        ),
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM after deleting rows to reclaim disk space.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print how many rows would be dropped without changing the DB.",
    )
    return parser.parse_args()


def ensure_valid_args(args: argparse.Namespace) -> None:
    if args.threshold_w_m2 < 0:
        raise SystemExit("--threshold-w-m2 must be non-negative.")
    if not args.db_path.exists():
        raise SystemExit(f"Database does not exist: {args.db_path}")


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    return [row[1] for row in rows]


def require_table_and_column(
    conn: sqlite3.Connection,
    table: str,
    radiation_column: str,
) -> None:
    columns = table_columns(conn, table)
    if not columns:
        raise RuntimeError(f"Table does not exist or has no columns: {table}")
    if radiation_column not in columns:
        raise RuntimeError(
            f"Column {radiation_column!r} does not exist in table {table!r}. "
            f"Available columns: {', '.join(columns)}"
        )


def sample_radiation_values(
    conn: sqlite3.Connection,
    table: str,
    radiation_column: str,
    sample_size: int = 100_000,
) -> tuple[float | None, float | None]:
    table_sql = quote_identifier(table)
    column_sql = quote_identifier(radiation_column)
    min_value, max_value = conn.execute(
        f"""
        SELECT MIN({column_sql}), MAX({column_sql})
        FROM (
            SELECT {column_sql}
            FROM {table_sql}
            WHERE {column_sql} IS NOT NULL
            LIMIT ?
        )
        """,
        (sample_size,),
    ).fetchone()
    return min_value, max_value


def normalized_threshold(column: str, threshold_w_m2: float) -> float:
    if column not in DEFAULT_NORM:
        raise RuntimeError(
            f"Cannot convert W m-2 threshold to normalized units for {column!r}; "
            "the column is not present in DEFAULT_NORM."
        )

    norm = DEFAULT_NORM[column]
    value_min = norm["norm_min"]
    value_max = norm["norm_max"]
    value_mid = (value_max + value_min) / 2.0
    value_range = value_max - value_min
    if norm["cyclic"]:
        value_range /= 2.0
    return (threshold_w_m2 - value_mid) / value_range


def normalized_bounds(column: str) -> tuple[float, float]:
    norm = DEFAULT_NORM[column]
    value_min = norm["norm_min"]
    value_max = norm["norm_max"]
    value_mid = (value_max + value_min) / 2.0
    value_range = value_max - value_min
    if norm["cyclic"]:
        value_range /= 2.0
    return (
        (value_min - value_mid) / value_range,
        (value_max - value_mid) / value_range,
    )


def validate_normalized_radiation_values(
    conn: sqlite3.Connection,
    table: str,
    radiation_column: str,
) -> None:
    min_value, max_value = sample_radiation_values(conn, table, radiation_column)
    if min_value is None or max_value is None:
        return

    expected_min, expected_max = normalized_bounds(radiation_column)
    tolerance = 1e-6
    if min_value < expected_min - tolerance or max_value > expected_max + tolerance:
        raise RuntimeError(
            f"{radiation_column!r} does not look normalized in {table!r}: "
            f"sample range is [{min_value:g}, {max_value:g}], expected roughly "
            f"[{expected_min:g}, {expected_max:g}] from DEFAULT_NORM."
        )


def count_rows(
    conn: sqlite3.Connection,
    table: str,
    where_sql: str | None = None,
    params: Iterable[object] = (),
) -> int:
    table_sql = quote_identifier(table)
    sql = f"SELECT COUNT(*) FROM {table_sql}"
    if where_sql:
        sql += f" WHERE {where_sql}"
    return int(conn.execute(sql, tuple(params)).fetchone()[0])


def drop_condition_sql(radiation_column: str, drop_missing_radiation: bool) -> str:
    column_sql = quote_identifier(radiation_column)
    condition = f"{column_sql} < ?"
    if drop_missing_radiation:
        condition = f"({condition} OR {column_sql} IS NULL)"
    return condition


def create_reindexed_table_sql(conn: sqlite3.Connection, table: str, tmp_table: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"Could not read CREATE TABLE SQL for {table!r}.")

    create_sql = row[0].strip()
    match = re.match(
        r"^(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?)(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)",
        create_sql,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError(f"Could not rewrite CREATE TABLE SQL for {table!r}.")

    return f"{match.group(1)}{quote_identifier(tmp_table)}{create_sql[match.end():]}"


def rebuild_ids(conn: sqlite3.Connection, table: str) -> None:
    columns = table_columns(conn, table)
    if "id" not in columns:
        raise RuntimeError(f"Cannot rebuild ids because {table!r} has no id column.")

    tmp_table = f"{table}__reindexed"
    table_sql = quote_identifier(table)
    tmp_table_sql = quote_identifier(tmp_table)
    copy_columns = [column for column in columns if column != "id"]
    copy_columns_sql = ", ".join(quote_identifier(column) for column in copy_columns)

    order_columns = [
        column for column in ("coord_id", "timestamp", "id") if column in columns
    ]
    order_sql = ", ".join(quote_identifier(column) for column in order_columns)
    if not order_sql:
        order_sql = quote_identifier("id")

    index_sqls = [
        row[0]
        for row in conn.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index'
              AND tbl_name = ?
              AND sql IS NOT NULL
            ORDER BY name
            """,
            (table,),
        ).fetchall()
    ]

    conn.execute(f"DROP TABLE IF EXISTS {tmp_table_sql}")
    conn.execute(create_reindexed_table_sql(conn, table, tmp_table))
    conn.execute(
        f"""
        INSERT INTO {tmp_table_sql} ({copy_columns_sql})
        SELECT {copy_columns_sql}
        FROM {table_sql}
        ORDER BY {order_sql}
        """
    )
    conn.execute(f"DROP TABLE {table_sql}")
    conn.execute(f"ALTER TABLE {tmp_table_sql} RENAME TO {table_sql}")
    for index_sql in index_sqls:
        conn.execute(index_sql)


def main() -> int:
    args = parse_args()
    ensure_valid_args(args)

    db_path = args.db_path.expanduser().resolve()
    with sqlite3.connect(db_path) as conn:
        require_table_and_column(conn, args.table, args.radiation_column)
        validate_normalized_radiation_values(
            conn,
            args.table,
            args.radiation_column,
        )
        threshold_in_storage_units = normalized_threshold(
            args.radiation_column,
            args.threshold_w_m2,
        )
        condition_sql = drop_condition_sql(
            args.radiation_column,
            args.drop_missing_radiation,
        )

        print(f"DB: {db_path}")
        print(f"Table: {args.table}")
        print(f"Radiation column: {args.radiation_column}")
        print("Radiation units: normalized")
        print(
            f"Threshold: {args.threshold_w_m2:g} W m-2 "
            f"({threshold_in_storage_units:g} in stored units)"
        )

        if args.dry_run:
            total_rows = count_rows(conn, args.table)
            rows_to_drop = count_rows(
                conn,
                args.table,
                condition_sql,
                (threshold_in_storage_units,),
            )
            rows_to_keep = total_rows - rows_to_drop
            print(f"Rows before: {total_rows}")
            print(f"Rows to drop: {rows_to_drop}")
            print(f"Rows to keep: {rows_to_keep}")
            print("Dry run only; no rows were deleted.")
            return 0

        conn.execute("BEGIN")
        try:
            cursor = conn.execute(
                f"DELETE FROM {quote_identifier(args.table)} WHERE {condition_sql}",
                (threshold_in_storage_units,),
            )
            rows_dropped = cursor.rowcount
            if rows_dropped > 0 and not args.no_rebuild_ids:
                rebuild_ids(conn, args.table)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    if args.vacuum:
        with sqlite3.connect(db_path) as conn:
            conn.execute("VACUUM")

    print(f"Dropped {rows_dropped} nighttime rows from {args.table}.")
    if rows_dropped > 0 and not args.no_rebuild_ids:
        print("Rebuilt ec_data ids ordered by coord_id, timestamp, id.")
    if args.vacuum:
        print("Vacuumed database.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
