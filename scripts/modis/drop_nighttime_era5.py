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
from dataclasses import dataclass
import re
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
DEFAULT_CHUNK_SIZE = 250_000


@dataclass(frozen=True)
class RewriteResult:
    id_slots_scanned: int
    rows_kept: int


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
            "Rewrite ec_data with only daytime rows where solar radiation is "
            "at or above the requested threshold. By default this removes "
            "nighttime rows with SW_IN < 2 W m-2."
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
        help="Run VACUUM after rewriting rows to reclaim disk space.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print how many rows would be kept without changing the DB.",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Number of id values to scan per rewrite chunk. "
            f"Default: {DEFAULT_CHUNK_SIZE}."
        ),
    )
    parser.add_argument(
        "--delete-chunk-size",
        dest="chunk_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def ensure_valid_args(args: argparse.Namespace) -> None:
    if args.threshold_w_m2 < 0:
        raise SystemExit("--threshold-w-m2 must be non-negative.")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1.")
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


def keep_condition_sql(radiation_column: str, drop_missing_radiation: bool) -> str:
    column_sql = quote_identifier(radiation_column)
    if drop_missing_radiation:
        return f"{column_sql} >= ?"
    return f"({column_sql} >= ? OR {column_sql} IS NULL)"


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    if not rows:
        raise RuntimeError(f"Could not read columns for table {table!r}.")
    return [row[1] for row in rows]


def table_index_sqls(conn: sqlite3.Connection, table: str) -> list[str]:
    return [
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


def create_rewrite_table_sql(
    conn: sqlite3.Connection,
    table: str,
    tmp_table: str,
) -> str:
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


def count_rewrite_rows_by_id_scan(
    conn: sqlite3.Connection,
    table: str,
    keep_sql: str,
    threshold_in_storage_units: float,
    chunk_size: int,
) -> tuple[int, int]:
    bounds = id_scan_bounds(conn, table)
    if bounds is None:
        return 0, 0

    min_id, max_id = bounds
    total_id_slots = max_id - min_id + 1
    table_sql = quote_identifier(table)
    rows_to_keep = 0
    with scan_progress("drop_nighttime rewrite/count", total_id_slots) as progress:
        for start_id, end_id in id_windows(min_id, max_id, chunk_size):
            progress.set_postfix(id=f"{start_id}-{end_id}", kept=rows_to_keep)
            progress.refresh()
            rows_to_keep += int(
                conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {table_sql}
                    WHERE id >= ?
                      AND id <= ?
                      AND {keep_sql}
                    """,
                    (start_id, end_id, threshold_in_storage_units),
                ).fetchone()[0]
            )
            progress.update(end_id - start_id + 1)
            progress.set_postfix(id=f"{start_id}-{end_id}", kept=rows_to_keep)
    return total_id_slots, rows_to_keep


def rewrite_without_nighttime_rows(
    conn: sqlite3.Connection,
    table: str,
    keep_sql: str,
    threshold_in_storage_units: float,
    chunk_size: int,
) -> RewriteResult:
    bounds = id_scan_bounds(conn, table)
    if bounds is None:
        return RewriteResult(id_slots_scanned=0, rows_kept=0)

    columns = table_columns(conn, table)
    if "id" not in columns:
        raise RuntimeError(f"Cannot rewrite {table!r} because it has no id column.")

    min_id, max_id = bounds
    total_id_slots = max_id - min_id + 1
    tmp_table = f"{table}__drop_nighttime_rewrite"
    table_sql = quote_identifier(table)
    tmp_table_sql = quote_identifier(tmp_table)
    columns_sql = ", ".join(quote_identifier(column) for column in columns)
    index_sqls = table_index_sqls(conn, table)

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(f"DROP TABLE IF EXISTS {tmp_table_sql}")
        conn.execute(create_rewrite_table_sql(conn, table, tmp_table))
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    rows_kept = 0
    insert_select_sql = f"""
        INSERT INTO {tmp_table_sql} ({columns_sql})
        SELECT {columns_sql}
        FROM {table_sql}
        WHERE id >= ?
          AND id <= ?
          AND {keep_sql}
        ORDER BY id
    """
    with scan_progress("drop_nighttime rewrite/keep", total_id_slots) as progress:
        for start_id, end_id in id_windows(min_id, max_id, chunk_size):
            progress.set_postfix(id=f"{start_id}-{end_id}", kept=rows_kept)
            progress.refresh()
            before_changes = conn.total_changes
            conn.execute("BEGIN")
            try:
                conn.execute(
                    insert_select_sql,
                    (start_id, end_id, threshold_in_storage_units),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            rows_kept += conn.total_changes - before_changes
            progress.update(end_id - start_id + 1)
            progress.set_postfix(id=f"{start_id}-{end_id}", kept=rows_kept)

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(f"DROP TABLE {table_sql}")
        conn.execute(f"ALTER TABLE {tmp_table_sql} RENAME TO {table_sql}")
        for index_sql in index_sqls:
            conn.execute(index_sql)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return RewriteResult(id_slots_scanned=total_id_slots, rows_kept=rows_kept)


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
        keep_sql = keep_condition_sql(
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
            print(f"Rewrite chunk size: {args.chunk_size} id values", flush=True)

        if args.dry_run:
            total_id_slots, rows_to_keep = count_rewrite_rows_by_id_scan(
                conn,
                args.table,
                keep_sql,
                threshold_in_storage_units,
                args.chunk_size,
            )
            print(f"ID values scanned: {total_id_slots}", flush=True)
            print(f"Rows to keep: {rows_to_keep}", flush=True)
            print("Dry run only; no rows were rewritten.", flush=True)
            return 0

        rewrite_result = rewrite_without_nighttime_rows(
            conn,
            args.table,
            keep_sql,
            threshold_in_storage_units,
            args.chunk_size,
        )
        result_message = (
            f"Rewrote {args.table}: scanned {rewrite_result.id_slots_scanned} "
            f"id slots and kept {rewrite_result.rows_kept} daytime rows."
        )
        id_rebuild_message = (
            "Preserved existing ids; run rebuild_ids_era5 after all filters."
        )

    if args.vacuum:
        with sqlite3.connect(db_path) as conn:
            vacuum_database(conn)

    print(result_message, flush=True)
    print(id_rebuild_message, flush=True)
    if args.vacuum:
        print("Vacuumed database.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
