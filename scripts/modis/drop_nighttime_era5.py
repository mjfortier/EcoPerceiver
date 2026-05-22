#!/usr/bin/env python3
"""Drop nighttime rows from the ERA5/ec_data SQLite table.

Rows with incoming shortwave radiation below the requested W m-2 threshold are
treated as nighttime rows where GPP would be forced to zero. This script assumes
the radiation column has already been min/max normalized with EcoPerceiver's
DEFAULT_NORM values, which is how the local ERA5 SQLite databases are built.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecoperceiver.constants import DEFAULT_NORM

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

DEFAULT_DB_PATH = Path("/home/l/luislara/links/projects/aip-pal/luislara/ep/data/era5.db")
DEFAULT_TABLE = "ec_data"
DEFAULT_RADIATION_COLUMN = "SW_IN"
DEFAULT_THRESHOLD_W_M2 = 2.0
DEFAULT_DELETE_CHUNK_SIZE = 250_000


class NullProgress:
    def update(self, _: int = 1) -> None:
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass


def scan_progress(desc: str, total: int, *, unit: str = "row-id"):
    if tqdm is None:
        return nullcontext(NullProgress())
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


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
        "--vacuum",
        action="store_true",
        help="Run VACUUM after deleting rows to reclaim disk space.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print how many rows would be dropped without changing the DB.",
    )
    parser.add_argument(
        "--delete-chunk-size",
        type=int,
        default=DEFAULT_DELETE_CHUNK_SIZE,
        help=(
            "Number of id values to scan per delete chunk. "
            f"Default: {DEFAULT_DELETE_CHUNK_SIZE}."
        ),
    )
    return parser.parse_args()


def ensure_valid_args(args: argparse.Namespace) -> None:
    if args.threshold_w_m2 < 0:
        raise SystemExit("--threshold-w-m2 must be non-negative.")
    if args.delete_chunk_size < 1:
        raise SystemExit("--delete-chunk-size must be at least 1.")
    if not args.db_path.exists():
        raise SystemExit(f"Database does not exist: {args.db_path}")


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


def drop_condition_sql(radiation_column: str, drop_missing_radiation: bool) -> str:
    column_sql = quote_identifier(radiation_column)
    condition = f"{column_sql} < ?"
    if drop_missing_radiation:
        condition = f"({condition} OR {column_sql} IS NULL)"
    return condition


def id_scan_bounds(conn: sqlite3.Connection, table: str) -> tuple[int, int] | None:
    table_sql = quote_identifier(table)
    row = conn.execute(
        f"SELECT id FROM {table_sql} LIMIT 1"
    ).fetchone()
    if row is None:
        return None

    seq_row = conn.execute(
        "SELECT seq FROM sqlite_sequence WHERE name = ?",
        (table,),
    ).fetchone()
    if seq_row is not None and seq_row[0] is not None:
        return 1, int(seq_row[0])

    max_row = conn.execute(
        f"SELECT id FROM {table_sql} ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if max_row is None:
        return None
    return 1, int(max_row[0])


def id_windows(min_id: int, max_id: int, chunk_size: int):
    start_id = min_id
    while start_id <= max_id:
        end_id = min(start_id + chunk_size - 1, max_id)
        yield start_id, end_id
        start_id = end_id + 1


def delete_nighttime_rows(
    conn: sqlite3.Connection,
    table: str,
    condition_sql: str,
    threshold_in_storage_units: float,
    chunk_size: int,
) -> int:
    bounds = id_scan_bounds(conn, table)
    if bounds is None:
        return 0

    min_id, max_id = bounds
    total_id_slots = max_id - min_id + 1
    table_sql = quote_identifier(table)
    rows_dropped = 0
    with scan_progress("drop_nighttime scan/delete", total_id_slots) as progress:
        for start_id, end_id in id_windows(min_id, max_id, chunk_size):
            progress.set_postfix(id=f"{start_id}-{end_id}", dropped=rows_dropped)
            progress.refresh()
            cursor = conn.execute(
                f"""
                DELETE FROM {table_sql}
                WHERE id >= ?
                  AND id <= ?
                  AND {condition_sql}
                """,
                (start_id, end_id, threshold_in_storage_units),
            )
            rows_dropped += cursor.rowcount
            progress.update(end_id - start_id + 1)
            progress.set_postfix(id=f"{start_id}-{end_id}", dropped=rows_dropped)
    return rows_dropped


def count_nighttime_rows_by_id_scan(
    conn: sqlite3.Connection,
    table: str,
    condition_sql: str,
    threshold_in_storage_units: float,
    chunk_size: int,
) -> tuple[int, int]:
    bounds = id_scan_bounds(conn, table)
    if bounds is None:
        return 0, 0

    min_id, max_id = bounds
    total_id_slots = max_id - min_id + 1
    table_sql = quote_identifier(table)
    rows_to_drop = 0
    with scan_progress("drop_nighttime scan/count", total_id_slots) as progress:
        for start_id, end_id in id_windows(min_id, max_id, chunk_size):
            progress.set_postfix(id=f"{start_id}-{end_id}", matched=rows_to_drop)
            progress.refresh()
            rows_to_drop += int(
                conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {table_sql}
                    WHERE id >= ?
                      AND id <= ?
                      AND {condition_sql}
                    """,
                    (start_id, end_id, threshold_in_storage_units),
                ).fetchone()[0]
            )
            progress.update(end_id - start_id + 1)
            progress.set_postfix(id=f"{start_id}-{end_id}", matched=rows_to_drop)
    return total_id_slots, rows_to_drop


def vacuum_database(conn: sqlite3.Connection) -> None:
    row = conn.execute("PRAGMA page_count").fetchone()
    total_pages = int(row[0]) if row else 0
    with scan_progress("drop_nighttime vacuum", total_pages, unit="page") as progress:
        conn.execute("VACUUM")
        progress.update(total_pages)


def main() -> int:
    args = parse_args()
    ensure_valid_args(args)

    db_path = args.db_path.expanduser().resolve()
    with sqlite3.connect(db_path) as conn:
        threshold_in_storage_units = normalized_threshold(
            args.radiation_column,
            args.threshold_w_m2,
        )
        condition_sql = drop_condition_sql(
            args.radiation_column,
            args.drop_missing_radiation,
        )

        print(f"DB: {db_path}", flush=True)
        print(f"Table: {args.table}", flush=True)
        print(f"Radiation column: {args.radiation_column}", flush=True)
        print("Radiation units: normalized", flush=True)
        print(
            f"Threshold: {args.threshold_w_m2:g} W m-2 "
            f"({threshold_in_storage_units:g} in stored units)",
            flush=True,
        )
        bounds = id_scan_bounds(conn, args.table)
        if bounds is None:
            print("ID scan range: empty table", flush=True)
        else:
            min_id, max_id = bounds
            print(
                f"ID scan range: {min_id}-{max_id} "
                f"({max_id - min_id + 1} id values)",
                flush=True,
            )
            print(f"Delete chunk size: {args.delete_chunk_size} id values", flush=True)

        if args.dry_run:
            total_id_slots, rows_to_drop = count_nighttime_rows_by_id_scan(
                conn,
                args.table,
                condition_sql,
                threshold_in_storage_units,
                args.delete_chunk_size,
            )
            print(f"ID values scanned: {total_id_slots}", flush=True)
            print(f"Rows to drop: {rows_to_drop}", flush=True)
            print("Dry run only; no rows were deleted.", flush=True)
            return 0

        conn.execute("BEGIN")
        try:
            rows_dropped = delete_nighttime_rows(
                conn,
                args.table,
                condition_sql,
                threshold_in_storage_units,
                args.delete_chunk_size,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    if args.vacuum:
        with sqlite3.connect(db_path) as conn:
            vacuum_database(conn)

    print(f"Dropped {rows_dropped} nighttime rows from {args.table}.", flush=True)
    print("Skipped id rebuild; run rebuild_ids_era5 after all filters.", flush=True)
    if args.vacuum:
        print("Vacuumed database.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
