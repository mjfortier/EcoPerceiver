#!/usr/bin/env python3
"""Assign coord_data.igbp from a MODIS MCD12C1 GeoTIFF."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import sys
import time
from urllib.parse import quote

import numpy as np
import rasterio
from rasterio.transform import rowcol
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ecoperceiver.constants import IGBP_ACRONYMS_MODIS

DEFAULT_DB_PATH = Path("/home/l/luislara/links/projects/aip-pal/luislara/ep/data/era5.db")
DEFAULT_C1_PATH = REPO_ROOT / "experiments" / "data" / "raw_modis" / "201801011200C1.tiff"
DEFAULT_TABLE = "coord_data"
DEFAULT_BATCH_SIZE = 10_000
SQLITE_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class RasterGrid:
    values: np.ndarray
    transform: object

    @property
    def height(self) -> int:
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class PlannedAssignment:
    rowid: int
    new_igbp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample MODIS MCD12C1 Majority_Land_Cover_Type_1 values and assign "
            "coord_data.igbp labels in an ERA5 SQLite database. The default is "
            "a dry run; pass --write to modify the database."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database to modify. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--c1-path",
        type=Path,
        default=DEFAULT_C1_PATH,
        help=f"MODIS MCD12C1 GeoTIFF. Default: {DEFAULT_C1_PATH}",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Coordinate table to assign. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "--only-null",
        action="store_true",
        help="Only assign rows where coord_data.igbp is NULL.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply the planned assignments. Without this flag, only a dry run is printed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"SQLite assignment batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def load_c1_raster(path: Path) -> RasterGrid:
    if not path.exists():
        raise SystemExit(f"C1 GeoTIFF does not exist: {path}")

    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise SystemExit(f"Expected a single-band C1 raster, found {dataset.count} bands.")
        if dataset.crs is not None and dataset.crs.to_epsg() != 4326:
            raise SystemExit(f"Expected C1 raster in EPSG:4326, found {dataset.crs}.")

        values = dataset.read(1)
        transform = dataset.transform

    if values.ndim != 2:
        raise SystemExit(f"Expected a single-band C1 raster, found shape {values.shape}.")

    unknown_codes = sorted(set(np.unique(values).tolist()) - set(IGBP_ACRONYMS_MODIS))
    if unknown_codes:
        raise SystemExit(
            "C1 raster contains unknown IGBP code(s): "
            + ", ".join(str(code) for code in unknown_codes)
        )

    if transform.a <= 0 or transform.e >= 0:
        raise SystemExit(f"Expected north-up C1 raster transform, found {transform}.")

    return RasterGrid(
        values=values,
        transform=transform,
    )


def clamp_index(value: int, upper: int) -> int:
    return max(0, min(upper - 1, value))


def sample_nearest(grid: RasterGrid, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    rows, cols = rowcol(grid.transform, lons, lats)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    rows = np.clip(rows, 0, grid.height - 1)
    cols = np.clip(cols, 0, grid.width - 1)
    return grid.values[rows, cols]


def connect_database(db_path: Path, *, readonly: bool) -> sqlite3.Connection:
    if readonly:
        db_uri = f"file:{quote(str(db_path), safe='/')}?mode=ro"
        return sqlite3.connect(db_uri, timeout=SQLITE_TIMEOUT_SECONDS, uri=True)
    return sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_SECONDS)


def ensure_coord_table(conn: sqlite3.Connection, table: str) -> None:
    columns = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    }
    if not columns:
        raise SystemExit(f"Table does not exist or has no columns: {table}")

    missing = {"lat", "lon", "igbp"} - columns
    if missing:
        raise SystemExit(
            f"Table {table} is missing required column(s): {', '.join(sorted(missing))}"
        )


def count_coord_rows(
    conn: sqlite3.Connection,
    table: str,
    *,
    only_null: bool,
) -> int:
    table_sql = quote_identifier(table)
    where_sql = "WHERE igbp IS NULL" if only_null else ""
    return int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {table_sql}
            {where_sql}
            """
        ).fetchone()[0]
    )


def iter_coord_row_batches(
    conn: sqlite3.Connection,
    table: str,
    *,
    only_null: bool,
    batch_size: int,
):
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    table_sql = quote_identifier(table)
    where_sql = "WHERE igbp IS NULL" if only_null else ""
    cursor = conn.execute(
        f"""
        SELECT rowid, coord_id, lat, lon, igbp
        FROM {table_sql}
        {where_sql}
        """
    )

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def plan_assignments(
    conn: sqlite3.Connection,
    table: str,
    grid: RasterGrid,
    *,
    only_null: bool,
    batch_size: int,
) -> tuple[list[PlannedAssignment], Counter[str], Counter[str], Counter[tuple[str, str]], int, int]:
    assignments: list[PlannedAssignment] = []
    old_counts: Counter[str] = Counter()
    new_counts: Counter[str] = Counter()
    transitions: Counter[tuple[str, str]] = Counter()
    skipped_missing_coords = 0
    row_count = count_coord_rows(conn, table, only_null=only_null)

    with tqdm(
        total=row_count,
        desc="assign_igbp plan",
        unit="coord",
        dynamic_ncols=True,
    ) as progress:
        for batch in iter_coord_row_batches(
            conn,
            table,
            only_null=only_null,
            batch_size=batch_size,
        ):
            valid_positions = []
            lats = []
            lons = []
            for position, row in enumerate(batch):
                _, _, lat, lon, old_igbp = row
                old_label = "NULL" if old_igbp is None else str(old_igbp)
                old_counts[old_label] += 1

                if lat is None or lon is None:
                    skipped_missing_coords += 1
                    new_counts[old_label] += 1
                    continue

                valid_positions.append(position)
                lats.append(float(lat))
                lons.append(float(lon))

            if valid_positions:
                codes = sample_nearest(
                    grid,
                    lats=np.asarray(lats, dtype=np.float64),
                    lons=np.asarray(lons, dtype=np.float64),
                )
                for position, code in zip(valid_positions, codes):
                    rowid, _, _, _, old_igbp = batch[position]
                    old_label = "NULL" if old_igbp is None else str(old_igbp)
                    new_igbp = IGBP_ACRONYMS_MODIS[int(code)]
                    new_counts[new_igbp] += 1
                    transitions[(old_label, new_igbp)] += 1

                    if old_igbp != new_igbp:
                        assignments.append(
                            PlannedAssignment(
                                rowid=int(rowid),
                                new_igbp=new_igbp,
                            )
                        )

            progress.update(len(batch))

    return assignments, old_counts, new_counts, transitions, skipped_missing_coords, row_count


def print_counts(title: str, counts: Counter[str]) -> None:
    print(title)
    for label, count in counts.most_common():
        print(f"  {label}: {count:,}")


def print_summary(
    *,
    db_path: Path,
    c1_path: Path,
    table: str,
    only_null: bool,
    row_count: int,
    assignments: list[PlannedAssignment],
    old_counts: Counter[str],
    new_counts: Counter[str],
    transitions: Counter[tuple[str, str]],
    skipped_missing_coords: int,
) -> None:
    print(f"DB: {db_path}")
    print(f"C1: {c1_path}")
    print(f"Table: {table}")
    print("Sampling: nearest coord_data lat/lon point")
    print(f"Rows scanned: {row_count:,}")
    if only_null:
        print("Scope: igbp IS NULL rows only")
    else:
        print("Scope: all coord_data rows")
    print(f"Rows with missing lat/lon skipped: {skipped_missing_coords:,}")
    print(f"Rows that would be assigned/changed: {len(assignments):,}")
    print()
    print_counts("Current labels in scan scope:", old_counts)
    print()
    print_counts("Labels after planned update in scan scope:", new_counts)
    print()
    print("Largest label transitions:")
    for (old_label, new_label), count in transitions.most_common(20):
        if old_label != new_label:
            print(f"  {old_label} -> {new_label}: {count:,}")


def apply_assignments(
    conn: sqlite3.Connection,
    table: str,
    assignments: list[PlannedAssignment],
    *,
    batch_size: int,
) -> None:
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    table_sql = quote_identifier(table)
    assignment_sql = f"UPDATE {table_sql} SET igbp = ? WHERE rowid = ?"
    started_at = time.monotonic()

    with conn:
        with tqdm(
            total=len(assignments),
            desc="assign_igbp write",
            unit="row",
            dynamic_ncols=True,
        ) as progress:
            for offset in range(0, len(assignments), batch_size):
                batch = assignments[offset : offset + batch_size]
                conn.executemany(
                    assignment_sql,
                    [(assignment.new_igbp, assignment.rowid) for assignment in batch],
                )
                progress.update(len(batch))

    print(f"Applied assignments in {format_duration(time.monotonic() - started_at)}")


def main() -> int:
    args = parse_args()
    db_path = resolve_path(args.db_path)
    c1_path = resolve_path(args.c1_path)

    if not db_path.exists():
        raise SystemExit(f"Database does not exist: {db_path}")

    grid = load_c1_raster(c1_path)
    with connect_database(db_path, readonly=not args.write) as conn:
        ensure_coord_table(conn, args.table)
        assignments, old_counts, new_counts, transitions, skipped_missing_coords, row_count = plan_assignments(
            conn,
            args.table,
            grid,
            only_null=args.only_null,
            batch_size=args.batch_size,
        )
        print_summary(
            db_path=db_path,
            c1_path=c1_path,
            table=args.table,
            only_null=args.only_null,
            row_count=row_count,
            assignments=assignments,
            old_counts=old_counts,
            new_counts=new_counts,
            transitions=transitions,
            skipped_missing_coords=skipped_missing_coords,
        )

        if not args.write:
            print()
            print("Dry run only; rerun with --write to assign IGBP in the database.")
            return 0

        if not assignments:
            print("No assignments needed.")
            return 0

        apply_assignments(conn, args.table, assignments, batch_size=args.batch_size)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
