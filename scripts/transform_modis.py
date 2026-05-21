#!/usr/bin/env python3
"""Transform raw global MODIS rasters into 9x8x8 arrays per 0.25 degree cell.

This script follows the cleaning logic used in CarbonSense/EcoPerceiver stage 3:

- MCD43A4 bands 0-6:
  - treat values < 0 or > 30000 as fill
  - truncate valid values at 10000
  - divide by 10000
- MCD43A2:
  - snow: null/fill (> 1) -> -1, else keep 0/1
  - water: map land/water classes to a binary band

Unlike the original site-level stage 3 pipeline, this script does not crop
pixels `[1:9, 1:9]`. It chunks the full global rasters directly into the
720 x 1440 ERA5-style grid, yielding one 9 x 8 x 8 tensor per cell.

Each transformed cell is inserted into `modis_data(coord_id, modis_date, data)`
in the target SQLite database.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

MISSING_DEPENDENCIES: list[str] = []

try:
    import rasterio
    from rasterio.windows import Window
except ModuleNotFoundError:
    MISSING_DEPENDENCIES.append("rasterio")
    rasterio = None
    Window = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "experiments" / "data" / "poc_modis"
DEFAULT_DB_PATH = REPO_ROOT / "experiments" / "data" / "poc_era5.db"
CELL_SIZE_DEGREES = 0.25
CELL_SIZE_PIXELS = 8
EXPECTED_GRID_WIDTH = 1440
EXPECTED_GRID_HEIGHT = 720
EXPECTED_RASTER_WIDTH = EXPECTED_GRID_WIDTH * CELL_SIZE_PIXELS
EXPECTED_RASTER_HEIGHT = EXPECTED_GRID_HEIGHT * CELL_SIZE_PIXELS
MODIS_FILE_RE = re.compile(r"^(?P<timestamp>\d{12})(?P<product>A2|A4)\.tiff$")
DEFAULT_HEAD_ELEMENTS = 24
GRID_KEY_SCALE = 4
COORD_ID_UNIQUE_INDEX = "idx_coord_data_coord_id"

# Same remapping as stage_3.py.
WATER_DICT = {
    0: 1,  # shallow ocean
    1: 0,  # land
    2: 0,  # ocean coastlines and lake shorelines
    3: 1,  # shallow inland water
    4: 1,  # ephemeral water
    5: 1,  # deep inland water
    6: 1,  # moderate or continental ocean
    7: 1,  # deep ocean
}
WATER_LOOKUP = np.array([WATER_DICT[i] for i in range(8)], dtype=np.float32)


@dataclass(frozen=True)
class FilePair:
    timestamp: str
    a4_path: Path
    a2_path: Path


@dataclass(frozen=True)
class ExampleCell:
    timestamp: str
    grid_row: int
    grid_col: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    values: np.ndarray


@dataclass
class ExampleReservoir:
    size: int
    rng: np.random.Generator
    seen: int = 0
    examples: list[ExampleCell] = field(default_factory=list)

    def add(self, example: ExampleCell) -> None:
        if self.size <= 0:
            return

        self.seen += 1
        if len(self.examples) < self.size:
            self.examples.append(example)
            return

        replacement_index = int(self.rng.integers(0, self.seen))
        if replacement_index < self.size:
            self.examples[replacement_index] = example


@dataclass(frozen=True)
class DbInsertStats:
    matched_coords: int = 0
    inserted_or_updated_rows: int = 0


def ensure_dependencies() -> None:
    if not MISSING_DEPENDENCIES:
        return

    missing = ", ".join(sorted(MISSING_DEPENDENCIES))
    raise SystemExit(
        "Missing Python dependencies: "
        f"{missing}. Activate the project environment or install them first."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transform paired raw MODIS A4/A2 GeoTIFFs into one 9x8x8 tensor "
            "per 0.25 degree grid cell and insert them into SQLite."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing raw MODIS GeoTIFFs. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=(
            "SQLite database to populate with `modis_data`. "
            f"Default: {DEFAULT_DB_PATH}"
        ),
    )
    parser.add_argument(
        "--date",
        action="append",
        default=[],
        help=(
            "Specific timestamp(s) to transform, e.g. 201712031200. "
            "Repeat for multiple dates."
        ),
    )
    parser.add_argument(
        "--limit-dates",
        type=int,
        default=None,
        help="Limit the number of matched timestamps processed.",
    )
    parser.add_argument(
        "--example-count",
        type=int,
        default=10,
        help="Number of random transformed example cells to print. Default: 10.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Update existing `modis_data` rows for the same "
            "`(coord_id, modis_date)`."
        ),
    )
    return parser.parse_args()


def discover_file_pairs(input_dir: Path, wanted_dates: set[str]) -> list[FilePair]:
    grouped: dict[str, dict[str, Path]] = {}

    for path in sorted(input_dir.iterdir()):
        match = MODIS_FILE_RE.match(path.name)
        if match is None:
            continue

        timestamp = match.group("timestamp")
        if wanted_dates and timestamp not in wanted_dates:
            continue

        product = match.group("product")
        grouped.setdefault(timestamp, {})[product] = path

    pairs: list[FilePair] = []
    for timestamp in sorted(grouped):
        files = grouped[timestamp]
        if "A4" in files and "A2" in files:
            pairs.append(FilePair(timestamp, files["A4"], files["A2"]))

    return pairs


def validate_raster(src, *, expected_band_count: int | tuple[int, ...], product: str) -> None:
    if isinstance(expected_band_count, int):
        valid_counts = (expected_band_count,)
    else:
        valid_counts = expected_band_count

    if src.count not in valid_counts:
        raise ValueError(
            f"{product} expected band count {valid_counts}, got {src.count} "
            f"for {src.name}."
        )

    if src.width != EXPECTED_RASTER_WIDTH or src.height != EXPECTED_RASTER_HEIGHT:
        raise ValueError(
            f"{product} expected raster size "
            f"{EXPECTED_RASTER_WIDTH}x{EXPECTED_RASTER_HEIGHT}, got "
            f"{src.width}x{src.height} for {src.name}."
        )


def clean_a4_data(arr: np.ndarray) -> np.ndarray:
    arr = np.where((arr > 30000) | (arr < 0), -10000, arr)
    arr = np.where(arr > 10000, 10000, arr)
    return (arr / 10000.0).astype(np.float32)


def clean_a2_data(arr: np.ndarray) -> np.ndarray:
    snow_arr = arr[0].astype(np.float32)
    snow_arr = np.where(snow_arr > 1, -1, snow_arr)
    snow_arr = np.nan_to_num(snow_arr, nan=-1.0)

    # The old stage_3.py used arr[2] because the site-level pull kept extra
    # A2 bands. The raw GeoTIFF downloader selects only snow + land/water,
    # so the water band is the second band when count == 2.
    water_band_index = 2 if arr.shape[0] > 2 else 1
    water_arr = arr[water_band_index].astype(np.float32)
    water_arr = np.where(water_arr > 7, 0, water_arr)
    water_arr = np.nan_to_num(water_arr, nan=0.0).astype(np.int16, copy=False)
    water_arr = WATER_LOOKUP[water_arr].astype(np.float32, copy=False)
    water_arr = np.nan_to_num(water_arr, nan=0.0)

    return np.stack((snow_arr, water_arr), axis=0).astype(np.float32, copy=False)


def row_bounds(grid_row: int) -> tuple[float, float]:
    lat_max = 90.0 - (grid_row * CELL_SIZE_DEGREES)
    lat_min = lat_max - CELL_SIZE_DEGREES
    return lat_min, lat_max


def col_bounds(grid_col: int) -> tuple[float, float]:
    lon_min = -180.0 + (grid_col * CELL_SIZE_DEGREES)
    lon_max = lon_min + CELL_SIZE_DEGREES
    return lon_min, lon_max


def grid_key(value: float) -> int:
    return int(round(value * GRID_KEY_SCALE))


def modis_date_from_timestamp(timestamp: str) -> int:
    return int(timestamp[:8])


def load_coord_lookup(db_path: Path) -> dict[int, dict[int, int]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT coord_id, lat, lon FROM coord_data").fetchall()

    if not rows:
        raise RuntimeError(f"`coord_data` is empty in {db_path}.")

    lookup: dict[int, dict[int, int]] = {}
    for coord_id, lat, lon in rows:
        lookup.setdefault(grid_key(lat), {})[grid_key(lon)] = coord_id
    return lookup


def ensure_coord_table_key(conn: sqlite3.Connection) -> None:
    try:
        null_coord_ids = conn.execute(
            "SELECT COUNT(*) FROM coord_data WHERE coord_id IS NULL"
        ).fetchone()[0]
    except sqlite3.OperationalError as exc:
        if "no such table: coord_data" in str(exc):
            raise RuntimeError(
                "The target database does not contain `coord_data`. "
                "Initialize the ERA5 database first or use `--db-path` to "
                "point at the correct SQLite file."
            ) from exc
        raise
    if null_coord_ids:
        raise RuntimeError(
            "`coord_data.coord_id` contains NULL values, so it cannot be used "
            "as a foreign-key target."
        )

    duplicate = conn.execute(
        """
        SELECT coord_id, COUNT(*)
        FROM coord_data
        GROUP BY coord_id
        HAVING COUNT(*) > 1
        LIMIT 1
        """
    ).fetchone()
    if duplicate is not None:
        coord_id, count = duplicate
        raise RuntimeError(
            "`coord_data.coord_id` is not unique. "
            f"coord_id={coord_id} appears {count} times."
        )

    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS {COORD_ID_UNIQUE_INDEX} "
        "ON coord_data(coord_id)"
    )
    conn.commit()


def ensure_modis_table(conn: sqlite3.Connection) -> None:
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS modis_data (
            modis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            coord_id INTEGER NOT NULL,
            modis_date INTEGER NOT NULL,
            data BLOB NOT NULL,
            FOREIGN KEY(coord_id) REFERENCES coord_data(coord_id),
            UNIQUE(coord_id, modis_date)
        );
        CREATE INDEX IF NOT EXISTS idx_modis_coord_date
        ON modis_data(coord_id, modis_date);
    """
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='modis_data'"
    ).fetchone()

    if table_exists is None:
        conn.executescript(create_table_sql)
        conn.commit()
        return

    foreign_keys = conn.execute("PRAGMA foreign_key_list(modis_data)").fetchall()
    if foreign_keys:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_modis_coord_date "
            "ON modis_data(coord_id, modis_date)"
        )
        conn.commit()
        return

    conn.executescript(
        """
        ALTER TABLE modis_data RENAME TO modis_data_old;
        CREATE TABLE modis_data (
            modis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            coord_id INTEGER NOT NULL,
            modis_date INTEGER NOT NULL,
            data BLOB NOT NULL,
            FOREIGN KEY(coord_id) REFERENCES coord_data(coord_id),
            UNIQUE(coord_id, modis_date)
        );
        INSERT INTO modis_data (modis_id, coord_id, modis_date, data)
        SELECT modis_id, coord_id, modis_date, data
        FROM modis_data_old;
        DROP TABLE modis_data_old;
        CREATE INDEX IF NOT EXISTS idx_modis_coord_date
        ON modis_data(coord_id, modis_date);
        """
    )
    conn.commit()


def insert_pair_into_database(
    conn: sqlite3.Connection,
    pair: FilePair,
    coord_lookup: dict[int, dict[int, int]],
    overwrite: bool,
) -> DbInsertStats:
    matched_coords = 0
    before_changes = conn.total_changes
    modis_date = modis_date_from_timestamp(pair.timestamp)

    if overwrite:
        insert_sql = (
            "INSERT INTO modis_data (coord_id, modis_date, data) VALUES (?, ?, ?) "
            "ON CONFLICT(coord_id, modis_date) DO UPDATE SET data=excluded.data"
        )
    else:
        insert_sql = (
            "INSERT INTO modis_data (coord_id, modis_date, data) VALUES (?, ?, ?) "
            "ON CONFLICT(coord_id, modis_date) DO NOTHING"
        )

    row_iter: Iterator[tuple[int, np.ndarray]] = iter_cell_rows(pair)
    if tqdm is not None:
        row_iter = tqdm(
            row_iter,
            total=EXPECTED_GRID_HEIGHT,
            desc=f"DB {pair.timestamp}",
            unit="row",
            dynamic_ncols=True,
        )

    for grid_row, cell_row in row_iter:
        lat_min, _ = row_bounds(grid_row)
        lon_lookup = coord_lookup.get(grid_key(lat_min))
        if lon_lookup is None:
            continue

        row_records: list[tuple[int, int, memoryview]] = []
        for grid_col, cell in enumerate(cell_row):
            lon_min, _ = col_bounds(grid_col)
            coord_id = lon_lookup.get(grid_key(lon_min))
            if coord_id is None:
                continue

            matched_coords += 1
            row_records.append(
                (
                    coord_id,
                    modis_date,
                    memoryview(np.ascontiguousarray(cell, dtype=np.float32)),
                )
            )

        if row_records:
            conn.executemany(insert_sql, row_records)

    conn.commit()
    return DbInsertStats(
        matched_coords=matched_coords,
        inserted_or_updated_rows=conn.total_changes - before_changes,
    )


def iter_cell_rows(pair: FilePair) -> Iterator[tuple[int, np.ndarray]]:
    with rasterio.open(pair.a4_path) as a4_src, rasterio.open(pair.a2_path) as a2_src:
        validate_raster(a4_src, expected_band_count=7, product="A4")
        validate_raster(a2_src, expected_band_count=(2, 3), product="A2")

        if a4_src.bounds != a2_src.bounds:
            raise ValueError(
                f"Bounds differ for timestamp {pair.timestamp}: "
                f"A4={a4_src.bounds}, A2={a2_src.bounds}"
            )

        for grid_row in range(EXPECTED_GRID_HEIGHT):
            pixel_row = grid_row * CELL_SIZE_PIXELS
            window = Window(
                col_off=0,
                row_off=pixel_row,
                width=EXPECTED_RASTER_WIDTH,
                height=CELL_SIZE_PIXELS,
            )

            a4_strip = clean_a4_data(a4_src.read(window=window))
            a2_strip = clean_a2_data(a2_src.read(window=window))
            combined = np.concatenate((a4_strip, a2_strip), axis=0)

            cell_row = combined.reshape(
                9,
                CELL_SIZE_PIXELS,
                EXPECTED_GRID_WIDTH,
                CELL_SIZE_PIXELS,
            ).transpose(2, 0, 1, 3)

            yield grid_row, np.ascontiguousarray(cell_row)


def collect_examples(
    reservoir: ExampleReservoir,
    timestamp: str,
    grid_row: int,
    cell_row: np.ndarray,
) -> None:
    for grid_col, cell in enumerate(cell_row):
        lat_min, lat_max = row_bounds(grid_row)
        lon_min, lon_max = col_bounds(grid_col)
        example = ExampleCell(
            timestamp=timestamp,
            grid_row=grid_row,
            grid_col=grid_col,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            values=cell.copy(),
        )
        reservoir.add(example)


def transform_pair(
    pair: FilePair,
    reservoir: ExampleReservoir,
) -> None:
    row_iter: Iterator[tuple[int, np.ndarray]] = iter_cell_rows(pair)
    if tqdm is not None:
        row_iter = tqdm(
            row_iter,
            total=EXPECTED_GRID_HEIGHT,
            desc=f"{pair.timestamp}",
            unit="row",
            dynamic_ncols=True,
        )

    for grid_row, cell_row in row_iter:
        collect_examples(reservoir, pair.timestamp, grid_row, cell_row)


def print_examples(examples: list[ExampleCell], requested_count: int) -> None:
    if not examples:
        print("No example cells were collected.")
        return

    print()
    print(f"Printed {len(examples)} example cells (requested {requested_count}).")
    for idx, example in enumerate(examples, start=1):
        print()
        print(
            f"Example {idx}: timestamp={example.timestamp} "
            f"grid_row={example.grid_row} grid_col={example.grid_col} "
            f"lat=[{example.lat_min:.2f}, {example.lat_max:.2f}) "
            f"lon=[{example.lon_min:.2f}, {example.lon_max:.2f})"
        )
        head = example.values.reshape(-1)[:DEFAULT_HEAD_ELEMENTS]
        print(
            np.array2string(
                head,
                precision=4,
                floatmode="fixed",
                suppress_small=False,
            )
        )
        print(f"shape={example.values.shape}")


def main() -> int:
    args = parse_args()
    ensure_dependencies()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if not args.db_path.exists():
        raise SystemExit(f"Database does not exist: {args.db_path}")

    pairs = discover_file_pairs(args.input_dir, set(args.date))
    if args.limit_dates is not None:
        if args.limit_dates < 1:
            raise SystemExit("--limit-dates must be >= 1.")
        pairs = pairs[: args.limit_dates]

    if not pairs:
        raise SystemExit(
            "No paired A4/A2 files were found. "
            "Check --input-dir or provide explicit --date values."
        )

    print(
        f"Found {len(pairs)} paired timestamp(s) in {args.input_dir}. "
        f"Transforming into {EXPECTED_GRID_HEIGHT}x{EXPECTED_GRID_WIDTH} cells "
        f"of shape 9x8x8."
    )

    reservoir = ExampleReservoir(
        size=args.example_count,
        rng=np.random.default_rng(),
    )
    with sqlite3.connect(args.db_path) as conn:
        ensure_coord_table_key(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_modis_table(conn)
        coord_lookup = load_coord_lookup(args.db_path)
        print(
            f"Loaded coord lookup from {args.db_path} with "
            f"{sum(len(v) for v in coord_lookup.values())} coord_id entries."
        )

        for pair in pairs:
            transform_pair(pair=pair, reservoir=reservoir)

            stats = insert_pair_into_database(
                conn=conn,
                pair=pair,
                coord_lookup=coord_lookup,
                overwrite=args.overwrite,
            )
            print(
                f"Inserted MODIS for {pair.timestamp}: "
                f"matched {stats.matched_coords} coord rows, "
                f"wrote {stats.inserted_or_updated_rows} rows into modis_data."
            )

    print_examples(reservoir.examples, args.example_count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
