#!/usr/bin/env python3
"""Create persistent SQLite indexes for evaluation and MODIS lookup."""

from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from urllib.parse import quote

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


DEFAULT_DB_PATH = Path("experiments/data/era5.db")
DEFAULT_TABLE = "ec_data"
DEFAULT_INDEX_NAME = "idx_ec_data_coord_id_timestamp_id"
DEFAULT_INDEX_COLUMNS = ("coord_id", "timestamp", "id")
SQLITE_PROGRESS_OPCODES = 100_000
HEARTBEAT_SECONDS = 15.0
PROGRESS_STATUS_SECONDS = 2.0
SQLITE_TIMEOUT_SECONDS = 60.0
ESTIMATED_INDEX_BYTES_PER_ROW = 32.0
ESTIMATED_TEMP_SORT_MULTIPLIER = 1.25
MIN_FREE_SPACE_MARGIN_BYTES = 10 * 1024**3
DEFAULT_BUILD_JOURNAL_MODE = "DELETE"
DEFAULT_PROGRESS_MODE = "auto"
DEFAULT_SQLITE_THREADS = 8


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def format_bytes(num_bytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def process_temp_file_bytes(temp_dir: Path) -> int:
    """Return bytes held by SQLite temp files, including unlinked Linux files."""
    fd_dir = Path("/proc") / str(os.getpid()) / "fd"
    temp_prefix = f"{temp_dir}{os.sep}"
    total = 0
    seen: set[tuple[int, int]] = set()

    if fd_dir.exists():
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            fds = []
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if not target.startswith(temp_prefix):
                continue
            try:
                stat_result = fd.stat()
            except OSError:
                continue
            key = (stat_result.st_dev, stat_result.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += stat_result.st_size

    try:
        paths = list(temp_dir.iterdir())
    except OSError:
        paths = []
    for path in paths:
        try:
            stat_result = path.stat()
        except OSError:
            continue
        key = (stat_result.st_dev, stat_result.st_ino)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            total += stat_result.st_size

    return total


def progress_status_text(
    *,
    opcodes: int,
    started_at: float,
    db_path: Path | None,
    temp_dir: Path | None,
) -> str:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    parts = [
        f"opcodes={opcodes:,}",
        f"elapsed={elapsed / 60.0:.1f} min",
        f"rate={opcodes / elapsed:,.0f} opcode/s",
    ]

    if db_path is not None:
        db_bytes = safe_file_size(db_path)
        wal_bytes = safe_file_size(db_path.with_name(f"{db_path.name}-wal"))
        journal_bytes = safe_file_size(db_path.with_name(f"{db_path.name}-journal"))
        parts.append(f"db={format_bytes(db_bytes)}")
        if wal_bytes:
            parts.append(f"wal={format_bytes(wal_bytes)}")
        if journal_bytes:
            parts.append(f"journal={format_bytes(journal_bytes)}")

    if temp_dir is not None:
        temp_bytes = process_temp_file_bytes(temp_dir)
        parts.append(f"temp={format_bytes(temp_bytes)}")
        try:
            free_bytes = shutil.disk_usage(existing_path_for_disk_check(temp_dir)).free
            parts.append(f"free={format_bytes(free_bytes)}")
        except RuntimeError:
            pass

    return ", ".join(parts)


class SqliteProgress:
    def __init__(
        self,
        desc: str,
        *,
        db_path: Path | None,
        temp_dir: Path | None,
        progress_mode: str,
    ):
        self.desc = desc
        self.db_path = db_path
        self.temp_dir = temp_dir
        self.progress_mode = self.resolve_progress_mode(progress_mode)
        self.opcodes = 0
        self.started_at = time.monotonic()
        self.last_heartbeat = self.started_at
        self.last_status = self.started_at
        self.progress = (
            tqdm(
                total=None,
                desc=desc,
                unit="opcode",
                unit_scale=True,
                dynamic_ncols=True,
                mininterval=1.0,
            )
            if self.progress_mode == "tqdm"
            else None
        )

    @staticmethod
    def resolve_progress_mode(progress_mode: str) -> str:
        if progress_mode == "none":
            return "none"
        if progress_mode == "heartbeat":
            return "heartbeat"
        if progress_mode == "tqdm":
            if tqdm is None:
                print("tqdm is not installed; falling back to heartbeat progress.", flush=True)
                return "heartbeat"
            return "tqdm"
        if tqdm is not None and sys.stderr.isatty():
            return "tqdm"
        return "heartbeat"

    def status_text(self) -> str:
        try:
            return progress_status_text(
                opcodes=self.opcodes,
                started_at=self.started_at,
                db_path=self.db_path,
                temp_dir=self.temp_dir,
            )
        except OSError as exc:
            return f"opcodes={self.opcodes:,}, status unavailable: {exc}"

    def update(self) -> int:
        self.opcodes += SQLITE_PROGRESS_OPCODES
        if self.progress_mode == "none":
            return 0

        now = time.monotonic()
        if self.progress is not None:
            self.progress.update(SQLITE_PROGRESS_OPCODES)
            if now - self.last_status >= PROGRESS_STATUS_SECONDS:
                self.progress.set_postfix_str(self.status_text(), refresh=True)
                self.last_status = now
            return 0

        if now - self.last_heartbeat >= HEARTBEAT_SECONDS:
            print(f"{self.desc}: {self.status_text()}", flush=True)
            self.last_heartbeat = now
        return 0

    def close(self) -> None:
        if self.progress is not None:
            self.progress.set_postfix_str(self.status_text(), refresh=True)
            self.progress.close()


@contextmanager
def sqlite_progress(
    conn: sqlite3.Connection,
    desc: str,
    *,
    db_path: Path | None,
    temp_dir: Path | None,
    progress_mode: str,
):
    progress = SqliteProgress(
        desc,
        db_path=db_path,
        temp_dir=temp_dir,
        progress_mode=progress_mode,
    )
    conn.set_progress_handler(progress.update, SQLITE_PROGRESS_OPCODES)
    try:
        yield
    finally:
        conn.set_progress_handler(None, 0)
        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the composite ec_data index used by eval startup and "
            "active MODIS coordinate lookup."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database to index. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Table to index. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help=f"Composite index name. Default: {DEFAULT_INDEX_NAME}",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_INDEX_COLUMNS),
        help=(
            "Columns for the composite index. "
            f"Default: {' '.join(DEFAULT_INDEX_COLUMNS)}"
        ),
    )
    parser.add_argument(
        "--analyze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ANALYZE after index changes so SQLite picks the new index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned index changes without modifying the database.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help=(
            "Directory for SQLite temporary sorter files. Default: a sqlite-tmp "
            "directory next to the database."
        ),
    )
    parser.add_argument(
        "--journal-mode",
        choices=("DELETE", "TRUNCATE", "PERSIST", "WAL"),
        default=DEFAULT_BUILD_JOURNAL_MODE,
        help=(
            "Journal mode to use while building the index. DELETE avoids writing "
            "the full index through a large WAL file. Default: DELETE."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip disk/page-count estimates before creating a missing index.",
    )
    parser.add_argument(
        "--estimate-index-bytes-per-row",
        type=float,
        default=ESTIMATED_INDEX_BYTES_PER_ROW,
        help=(
            "Preflight estimate for final index bytes per table row. "
            f"Default: {ESTIMATED_INDEX_BYTES_PER_ROW:g}."
        ),
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "tqdm", "heartbeat", "none"),
        default=DEFAULT_PROGRESS_MODE,
        help=(
            "Progress display. auto uses tqdm on an interactive terminal and "
            "heartbeat log lines otherwise. tqdm is indeterminate because "
            "SQLite does not expose total CREATE INDEX work. Default: auto."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_SQLITE_THREADS,
        help=(
            "SQLite worker threads for sort operations such as CREATE INDEX. "
            f"Default: {DEFAULT_SQLITE_THREADS}."
        ),
    )
    return parser.parse_args()


def ensure_valid_args(args: argparse.Namespace) -> None:
    args.db_path = args.db_path.expanduser().resolve()
    if not args.db_path.exists():
        raise SystemExit(f"Database does not exist: {args.db_path}")
    if not args.columns:
        raise SystemExit("--columns must contain at least one column.")
    if args.temp_dir is None:
        args.temp_dir = args.db_path.parent / "sqlite-tmp"
    else:
        args.temp_dir = args.temp_dir.expanduser().resolve()
    if args.estimate_index_bytes_per_row <= 0:
        raise SystemExit("--estimate-index-bytes-per-row must be positive.")
    if args.threads < 0:
        raise SystemExit("--threads must be zero or greater.")


def connect_database(db_path: Path, *, readonly: bool) -> sqlite3.Connection:
    if readonly:
        db_uri = f"file:{quote(str(db_path), safe='/')}?mode=ro"
        return sqlite3.connect(db_uri, timeout=SQLITE_TIMEOUT_SECONDS, uri=True)
    return sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_SECONDS)


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    if not rows:
        raise RuntimeError(f"Table does not exist or has no columns: {table}")
    return {row[1] for row in rows}


def ensure_table_has_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
) -> None:
    available_columns = table_columns(conn, table)
    missing_columns = [column for column in columns if column not in available_columns]
    if missing_columns:
        raise RuntimeError(
            f"Cannot index {table}; missing column(s): {', '.join(missing_columns)}"
        )


def index_columns(conn: sqlite3.Connection, index_name: str) -> tuple[str, ...] | None:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?",
        (index_name,),
    ).fetchall()
    if not rows:
        return None

    return tuple(
        column_row[2]
        for column_row in conn.execute(
            f"PRAGMA index_info({quote_identifier(index_name)})"
        )
    )


def table_row_estimate(conn: sqlite3.Connection, table: str) -> int:
    sequence_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'"
    ).fetchone()
    if sequence_exists is not None:
        row = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = ?",
            (table,),
        ).fetchone()
        if row is not None and row[0] is not None:
            return int(row[0])

    try:
        row = conn.execute(f"SELECT MAX(rowid) FROM {quote_identifier(table)}").fetchone()
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Cannot cheaply estimate row count for {table!r}; "
            "rerun with --skip-preflight if you want to proceed."
        ) from exc
    return int(row[0] or 0)


def sqlite_pragma_int(conn: sqlite3.Connection, pragma_name: str) -> int:
    row = conn.execute(f"PRAGMA {pragma_name}").fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"Could not read PRAGMA {pragma_name}.")
    return int(row[0])


def existing_path_for_disk_check(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise RuntimeError(f"No existing parent directory found for {path}")
        current = parent
    return current


def same_filesystem(path_a: Path, path_b: Path) -> bool:
    return os.stat(path_a).st_dev == os.stat(path_b).st_dev


def ensure_enough_space(path: Path, required_bytes: float, label: str) -> None:
    free_bytes = shutil.disk_usage(path).free
    print(
        f"{label} free space: {format_bytes(free_bytes)} "
        f"(estimated need: {format_bytes(required_bytes)})",
        flush=True,
    )
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Not enough free space on {path} for {label}: "
            f"need about {format_bytes(required_bytes)}, "
            f"available {format_bytes(free_bytes)}."
        )


def preflight_index_build(
    conn: sqlite3.Connection,
    *,
    table: str,
    db_path: Path,
    temp_dir: Path,
    estimate_index_bytes_per_row: float,
) -> None:
    page_size = sqlite_pragma_int(conn, "page_size")
    page_count = sqlite_pragma_int(conn, "page_count")
    max_page_count = sqlite_pragma_int(conn, "max_page_count")
    row_count = table_row_estimate(conn, table)
    estimated_index_bytes = row_count * estimate_index_bytes_per_row
    estimated_index_pages = int((estimated_index_bytes + page_size - 1) // page_size)
    estimated_final_pages = page_count + estimated_index_pages
    estimated_temp_bytes = estimated_index_bytes * ESTIMATED_TEMP_SORT_MULTIPLIER

    print("Preflight estimate:", flush=True)
    print(f"  table rows: {row_count:,}", flush=True)
    print(f"  current DB size: {format_bytes(page_count * page_size)}", flush=True)
    print(f"  estimated final index size: {format_bytes(estimated_index_bytes)}", flush=True)
    print(f"  estimated SQLite temp sort space: {format_bytes(estimated_temp_bytes)}", flush=True)
    print(
        f"  estimated final page count: {estimated_final_pages:,} "
        f"of {max_page_count:,}",
        flush=True,
    )

    if estimated_final_pages >= max_page_count:
        raise RuntimeError(
            "The estimated final database page count exceeds SQLite's configured "
            f"max_page_count ({max_page_count:,})."
        )

    db_parent = db_path.parent
    temp_space_path = existing_path_for_disk_check(temp_dir)
    if same_filesystem(db_parent, temp_space_path):
        required_bytes = (
            estimated_index_bytes
            + estimated_temp_bytes
            + MIN_FREE_SPACE_MARGIN_BYTES
        )
        ensure_enough_space(db_parent, required_bytes, "DB/temp filesystem")
    else:
        ensure_enough_space(
            db_parent,
            estimated_index_bytes + MIN_FREE_SPACE_MARGIN_BYTES,
            "DB filesystem",
        )
        ensure_enough_space(
            temp_space_path,
            estimated_temp_bytes + MIN_FREE_SPACE_MARGIN_BYTES,
            "SQLite temp filesystem",
        )


def prepare_temp_dir(temp_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"SQLite temp dir: {temp_dir}")
        return

    temp_dir.mkdir(parents=True, exist_ok=True)
    if not temp_dir.is_dir():
        raise RuntimeError(f"SQLite temp path is not a directory: {temp_dir}")

    os.environ["SQLITE_TMPDIR"] = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)
    print(f"SQLite temp dir: {temp_dir}", flush=True)


def configure_sqlite_runtime(conn: sqlite3.Connection, *, threads: int) -> None:
    conn.execute("PRAGMA temp_store = FILE")
    row = conn.execute(f"PRAGMA threads = {threads}").fetchone()
    actual_threads = int(row[0]) if row is not None else threads
    print(f"SQLite worker threads: {actual_threads}", flush=True)
    if actual_threads != threads:
        print(
            f"Requested {threads} SQLite worker threads, "
            f"but SQLite accepted {actual_threads}.",
            flush=True,
        )


@contextmanager
def sqlite_index_build_mode(
    conn: sqlite3.Connection,
    *,
    journal_mode: str,
    dry_run: bool,
):
    original_row = conn.execute("PRAGMA journal_mode").fetchone()
    original_mode = str(original_row[0]).upper() if original_row else "UNKNOWN"
    requested_mode = journal_mode.upper()
    active_mode = original_mode
    print(f"Original journal_mode: {original_mode}", flush=True)

    if not dry_run and requested_mode != original_mode:
        row = conn.execute(f"PRAGMA journal_mode = {requested_mode}").fetchone()
        active_mode = str(row[0]).upper() if row else requested_mode
        if active_mode != requested_mode:
            raise RuntimeError(
                f"Could not switch SQLite journal_mode to {requested_mode}; "
                f"SQLite reported {active_mode}."
            )
        print(f"Build journal_mode: {active_mode}", flush=True)

    try:
        yield
    except BaseException:
        if not dry_run:
            conn.rollback()
        raise
    finally:
        if not dry_run and original_mode != "UNKNOWN" and active_mode != original_mode:
            row = conn.execute(f"PRAGMA journal_mode = {original_mode}").fetchone()
            restored_mode = str(row[0]).upper() if row else "UNKNOWN"
            print(f"Restored journal_mode: {restored_mode}", flush=True)


def create_composite_index(
    conn: sqlite3.Connection,
    *,
    table: str,
    index_name: str,
    columns: list[str],
    db_path: Path,
    temp_dir: Path,
    progress_mode: str,
    dry_run: bool,
) -> None:
    existing_columns = index_columns(conn, index_name)
    requested_columns = tuple(columns)
    if existing_columns is not None:
        if existing_columns != requested_columns:
            raise RuntimeError(
                f"Index {index_name!r} already exists on columns "
                f"{existing_columns}, expected {requested_columns}."
            )
        print(f"Composite index already exists: {index_name}({', '.join(columns)})")
        return

    columns_sql = ", ".join(quote_identifier(column) for column in columns)
    create_sql = (
        f"CREATE INDEX {quote_identifier(index_name)} "
        f"ON {quote_identifier(table)}({columns_sql})"
    )
    print(f"Creating composite index: {index_name}({', '.join(columns)})", flush=True)
    if dry_run:
        print(f"Dry run SQL: {create_sql}")
        return

    with sqlite_progress(
        conn,
        f"create {index_name}",
        db_path=db_path,
        temp_dir=temp_dir,
        progress_mode=progress_mode,
    ):
        conn.execute(create_sql)
    conn.commit()
    print(f"Created composite index: {index_name}", flush=True)


def analyze_table(
    conn: sqlite3.Connection,
    *,
    table: str,
    db_path: Path,
    temp_dir: Path,
    progress_mode: str,
    dry_run: bool,
) -> None:
    analyze_sql = f"ANALYZE {quote_identifier(table)}"
    print(f"Running ANALYZE for {table}", flush=True)
    if dry_run:
        print(f"Dry run SQL: {analyze_sql}")
        return

    with sqlite_progress(
        conn,
        f"analyze {table}",
        db_path=db_path,
        temp_dir=temp_dir,
        progress_mode=progress_mode,
    ):
        conn.execute(analyze_sql)
    conn.commit()
    print(f"Analyzed table: {table}", flush=True)


def main() -> int:
    args = parse_args()
    ensure_valid_args(args)

    print(f"DB: {args.db_path}")
    print(f"Table: {args.table}")
    print(f"Composite index: {args.index_name}({', '.join(args.columns)})")

    prepare_temp_dir(args.temp_dir, dry_run=args.dry_run)
    with closing(connect_database(args.db_path, readonly=args.dry_run)) as conn:
        ensure_table_has_columns(conn, args.table, args.columns)
        missing_index = index_columns(conn, args.index_name) is None
        if missing_index and not args.skip_preflight:
            preflight_index_build(
                conn,
                table=args.table,
                db_path=args.db_path,
                temp_dir=args.temp_dir,
                estimate_index_bytes_per_row=args.estimate_index_bytes_per_row,
            )
        if args.dry_run:
            print(f"SQLite worker threads: {args.threads} (planned)", flush=True)
        else:
            configure_sqlite_runtime(conn, threads=args.threads)
        with sqlite_index_build_mode(
            conn,
            journal_mode=args.journal_mode,
            dry_run=args.dry_run,
        ):
            create_composite_index(
                conn,
                table=args.table,
                index_name=args.index_name,
                columns=args.columns,
                db_path=args.db_path,
                temp_dir=args.temp_dir,
                progress_mode=args.progress,
                dry_run=args.dry_run,
            )
            if args.analyze:
                analyze_table(
                    conn,
                    table=args.table,
                    db_path=args.db_path,
                    temp_dir=args.temp_dir,
                    progress_mode=args.progress,
                    dry_run=args.dry_run,
                )

    if args.dry_run:
        print("Dry run only; no indexes were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
