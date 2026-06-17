#!/usr/bin/env python3
"""Rebuild ERA5 ec_data ids once after filtering rows."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import re
import sqlite3
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


DEFAULT_DB_PATH = Path("/home/l/luislara/links/projects/aip-pal/luislara/ep/data/era5.db")
DEFAULT_TABLE = "ec_data"
DEFAULT_COPY_CHUNK_SIZE = 100_000


class NullProgress:
    def update(self, _: int = 1) -> None:
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass


def progress_bar(desc: str, total: int, *, unit: str):
    if tqdm is None:
        return nullcontext(NullProgress())
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite ec_data once so id values are contiguous after all row "
            "filtering steps have finished."
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
        help=f"Table whose ids should be rebuilt. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM after rebuilding ids.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print how many rows would be copied without changing the DB.",
    )
    parser.add_argument(
        "--copy-chunk-size",
        type=int,
        default=DEFAULT_COPY_CHUNK_SIZE,
        help=(
            "Number of rows to copy per id-rebuild chunk. "
            f"Default: {DEFAULT_COPY_CHUNK_SIZE}."
        ),
    )
    return parser.parse_args()


def ensure_valid_args(args: argparse.Namespace) -> None:
    if not args.db_path.exists():
        raise SystemExit(f"Database does not exist: {args.db_path}")
    if args.copy_chunk_size < 1:
        raise SystemExit("--copy-chunk-size must be at least 1.")


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    return [row[1] for row in rows]


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    return int(
        conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()[0]
    )


def create_rebuilt_ids_table_sql(
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


def copy_and_order_columns(
    columns: list[str],
    table: str,
) -> tuple[list[str], list[str]]:
    if "id" not in columns:
        raise RuntimeError(f"Cannot rebuild ids because {table!r} has no id column.")

    copy_columns = [column for column in columns if column != "id"]
    if not copy_columns:
        raise RuntimeError(f"Cannot rebuild ids for {table!r}; it has no non-id columns.")

    order_columns = [
        column for column in ("coord_id", "timestamp", "id") if column in columns
    ]
    if not order_columns:
        order_columns = ["id"]

    return copy_columns, order_columns


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


def copied_position(order_columns: list[str], last_order_values: tuple[object, ...]) -> str:
    values_by_column = dict(zip(order_columns, last_order_values))
    parts = []
    for column in ("coord_id", "timestamp", "id"):
        if column in values_by_column:
            parts.append(f"{column}={values_by_column[column]}")
    return ",".join(parts) if parts else str(last_order_values[-1])


def copy_rows_with_progress(
    conn: sqlite3.Connection,
    table: str,
    tmp_table: str,
    copy_columns: list[str],
    order_columns: list[str],
    total_rows: int,
    chunk_size: int,
) -> int:
    table_sql = quote_identifier(table)
    tmp_table_sql = quote_identifier(tmp_table)
    copy_columns_sql = ", ".join(quote_identifier(column) for column in copy_columns)
    select_columns = copy_columns + [
        column for column in order_columns if column not in copy_columns
    ]
    select_columns_sql = ", ".join(quote_identifier(column) for column in select_columns)
    order_sql = ", ".join(quote_identifier(column) for column in order_columns)
    placeholders = ", ".join("?" for _ in copy_columns)
    insert_sql = f"INSERT INTO {tmp_table_sql} ({copy_columns_sql}) VALUES ({placeholders})"
    order_indexes = [select_columns.index(column) for column in order_columns]
    copied_rows = 0
    cursor = conn.execute(
        f"""
        SELECT {select_columns_sql}
        FROM {table_sql}
        ORDER BY {order_sql}
        """
    )

    with progress_bar("rebuild_ids_era5 copy", total_rows, unit="row") as progress:
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break

            conn.executemany(
                insert_sql,
                [row[: len(copy_columns)] for row in rows],
            )
            copied_rows += len(rows)
            last_order_values = tuple(rows[-1][index] for index in order_indexes)
            progress.update(len(rows))
            progress.set_postfix(
                copied=copied_rows,
                at=copied_position(order_columns, last_order_values),
            )

    return copied_rows


def rebuild_ids(conn: sqlite3.Connection, table: str, chunk_size: int) -> int:
    columns = table_columns(conn, table)
    if "id" not in columns:
        raise RuntimeError(f"Cannot rebuild ids because {table!r} has no id column.")

    tmp_table = f"{table}__rebuilt_ids"
    table_sql = quote_identifier(table)
    tmp_table_sql = quote_identifier(tmp_table)
    copy_columns, order_columns = copy_and_order_columns(columns, table)
    index_sqls = table_index_sqls(conn, table)

    conn.execute(f"DROP TABLE IF EXISTS {tmp_table_sql}")
    conn.execute(create_rebuilt_ids_table_sql(conn, table, tmp_table))
    total_rows = count_rows(conn, table)
    copied_rows = copy_rows_with_progress(
        conn,
        table,
        tmp_table,
        copy_columns,
        order_columns,
        total_rows,
        chunk_size,
    )
    conn.execute(f"DROP TABLE {table_sql}")
    conn.execute(f"ALTER TABLE {tmp_table_sql} RENAME TO {table_sql}")
    for index_sql in index_sqls:
        conn.execute(index_sql)
    return copied_rows


def vacuum_database(conn: sqlite3.Connection) -> None:
    row = conn.execute("PRAGMA page_count").fetchone()
    total_pages = int(row[0]) if row else 0
    with progress_bar("rebuild_ids_era5 vacuum", total_pages, unit="page") as progress:
        conn.execute("VACUUM")
        progress.update(total_pages)


def main() -> int:
    args = parse_args()
    args.db_path = args.db_path.expanduser().resolve()
    ensure_valid_args(args)

    with sqlite3.connect(args.db_path) as conn:
        print(f"DB: {args.db_path}")
        print(f"Table: {args.table}")
        print(f"Copy chunk size: {args.copy_chunk_size} rows")

        if args.dry_run:
            total_rows = count_rows(conn, args.table)
            print(f"Rows to copy into rebuilt-id table: {total_rows}")
            print("Dry run only; no ids were rebuilt.")
            return 0

        conn.execute("BEGIN")
        try:
            copied_rows = rebuild_ids(conn, args.table, args.copy_chunk_size)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    if args.vacuum:
        with sqlite3.connect(args.db_path) as conn:
            vacuum_database(conn)

    print(
        f"Rebuilt ids for {args.table}: copied {copied_rows} rows and recreated indexes."
    )
    if args.vacuum:
        print("Vacuumed database.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
