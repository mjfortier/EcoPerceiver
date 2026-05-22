#!/usr/bin/env python3
"""Drop full geographic areas from the ERA5/ec_data SQLite table."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = Path("/home/l/luislara/links/projects/aip-pal/luislara/ep/data/era5.db")
DEFAULT_AREAS_CONFIG_PATH = SCRIPT_DIR / "drop_areas_config.yml"
DEFAULT_TABLE = "ec_data"
DEFAULT_COORD_TABLE = "coord_data"
DEFAULT_DELETE_CHUNK_SIZE = 2_000_000
ERA5_GRID_DEGREES = 0.25
T = TypeVar("T")


@dataclass(frozen=True)
class AreaBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class Area:
    name: str
    description: str
    enabled: bool
    bounds: AreaBounds
    polygon: tuple[tuple[float, float], ...] | None = None


class NullProgress:
    def update(self, _: int = 1) -> None:
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass


def step_progress(desc: str, total: int):
    if tqdm is None:
        return nullcontext(NullProgress())
    return tqdm(total=total, desc=desc, unit="step", dynamic_ncols=True)


def progress_bar(desc: str, total: int, *, unit: str):
    if tqdm is None:
        return nullcontext(NullProgress())
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str,
    leave: bool = False,
) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
    )


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop complete geographic areas from ec_data by matching coord_data "
            "lat/lon values against configured ERA5-aligned bounding boxes."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database to edit. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--areas-config",
        type=Path,
        default=DEFAULT_AREAS_CONFIG_PATH,
        help=f"YAML file with area definitions. Default: {DEFAULT_AREAS_CONFIG_PATH}",
    )
    parser.add_argument(
        "--area",
        action="append",
        default=[],
        help=(
            "Area name to drop. Repeat for multiple areas. "
            "Default: all enabled areas in --areas-config."
        ),
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Table to filter. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "--coord-table",
        default=DEFAULT_COORD_TABLE,
        help=f"Coordinate table. Default: {DEFAULT_COORD_TABLE}",
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
        "--delete-coord-chunk-size",
        dest="delete_chunk_size",
        type=int,
        default=DEFAULT_DELETE_CHUNK_SIZE,
        help=(
            "Number of ec_data id values to scan per area-delete chunk. "
            f"Default: {DEFAULT_DELETE_CHUNK_SIZE}."
        ),
    )
    return parser.parse_args()


def ensure_valid_args(args: argparse.Namespace) -> None:
    if not args.db_path.exists():
        raise SystemExit(f"Database does not exist: {args.db_path}")
    if not args.areas_config.exists():
        raise SystemExit(f"Areas config does not exist: {args.areas_config}")
    if args.delete_chunk_size < 1:
        raise SystemExit("--delete-chunk-size must be at least 1.")


def is_grid_aligned(value: float) -> bool:
    scaled = value / ERA5_GRID_DEGREES
    return abs(scaled - round(scaled)) < 1e-9


def require_grid_aligned(name: str, value: float, area_name: str) -> None:
    if not is_grid_aligned(value):
        raise RuntimeError(
            f"Area {area_name!r} bound {name}={value:g} is not aligned to the "
            f"{ERA5_GRID_DEGREES:g} degree ERA5 grid."
        )


def load_areas(config_path: Path) -> list[Area]:
    if yaml is None:
        raise SystemExit(
            "Missing Python dependency: PyYAML. Install it or run inside the "
            "project environment."
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise RuntimeError(f"Areas config must be a YAML mapping: {config_path}")

    raw_areas = config.get("areas", [])
    if not isinstance(raw_areas, list):
        raise RuntimeError("Areas config field 'areas' must be a list.")

    areas: list[Area] = []
    seen_names: set[str] = set()
    for raw_area in raw_areas:
        if not isinstance(raw_area, dict):
            raise RuntimeError("Each area entry must be a YAML mapping.")

        name = raw_area.get("name")
        if not name:
            raise RuntimeError("Each area entry must define a non-empty name.")
        name = str(name)
        if name in seen_names:
            raise RuntimeError(f"Duplicate area name in config: {name}")
        seen_names.add(name)

        polygon = load_area_polygon(name, raw_area)
        bounds = load_area_bounds(name, raw_area, polygon)
        validate_bounds(name, bounds)
        areas.append(
            Area(
                name=name,
                description=str(raw_area.get("description", "")),
                enabled=bool(raw_area.get("enabled", True)),
                bounds=bounds,
                polygon=polygon,
            )
        )

    return areas


def load_area_bounds(
    area_name: str,
    raw_area: dict[str, Any],
    polygon: tuple[tuple[float, float], ...] | None,
) -> AreaBounds:
    raw_bounds = raw_area.get("bounds")
    if raw_bounds is not None:
        if not isinstance(raw_bounds, dict):
            raise RuntimeError(f"Area {area_name!r} bounds must be a mapping.")
        return AreaBounds(
            lat_min=float(raw_bounds["lat_min"]),
            lat_max=float(raw_bounds["lat_max"]),
            lon_min=float(raw_bounds["lon_min"]),
            lon_max=float(raw_bounds["lon_max"]),
        )

    if not polygon:
        raise RuntimeError(f"Area {area_name!r} must define bounds or polygon.points.")

    lon_values = [lon for lon, _ in polygon]
    lat_values = [lat for _, lat in polygon]
    return AreaBounds(
        lat_min=min(lat_values),
        lat_max=max(lat_values),
        lon_min=min(lon_values),
        lon_max=max(lon_values),
    )


def load_area_polygon(
    area_name: str,
    raw_area: dict[str, Any],
) -> tuple[tuple[float, float], ...] | None:
    raw_polygon = raw_area.get("polygon")
    if raw_polygon is None:
        return None
    if not isinstance(raw_polygon, dict):
        raise RuntimeError(f"Area {area_name!r} polygon must be a mapping.")

    raw_points = raw_polygon.get("points")
    if not isinstance(raw_points, list) or len(raw_points) < 4:
        raise RuntimeError(
            f"Area {area_name!r} polygon.points must contain at least 4 points."
        )

    points: list[tuple[float, float]] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise RuntimeError(f"Area {area_name!r} polygon points must be [lon, lat].")
        lon = float(raw_point[0])
        lat = float(raw_point[1])
        require_grid_aligned("polygon lon", lon, area_name)
        require_grid_aligned("polygon lat", lat, area_name)
        points.append((lon, lat))

    if points[0] != points[-1]:
        points.append(points[0])
    return tuple(points)


def validate_bounds(area_name: str, bounds: AreaBounds) -> None:
    if not -90.0 <= bounds.lat_min <= 90.0:
        raise RuntimeError(f"Area {area_name!r} lat_min is outside [-90, 90].")
    if not -90.0 <= bounds.lat_max <= 90.0:
        raise RuntimeError(f"Area {area_name!r} lat_max is outside [-90, 90].")
    if not -180.0 <= bounds.lon_min <= 180.0:
        raise RuntimeError(f"Area {area_name!r} lon_min is outside [-180, 180].")
    if not -180.0 <= bounds.lon_max <= 180.0:
        raise RuntimeError(f"Area {area_name!r} lon_max is outside [-180, 180].")
    if bounds.lat_min > bounds.lat_max:
        raise RuntimeError(f"Area {area_name!r} lat_min must be <= lat_max.")
    if bounds.lon_min > bounds.lon_max:
        raise RuntimeError(f"Area {area_name!r} lon_min must be <= lon_max.")

    require_grid_aligned("lat_min", bounds.lat_min, area_name)
    require_grid_aligned("lat_max", bounds.lat_max, area_name)
    require_grid_aligned("lon_min", bounds.lon_min, area_name)
    require_grid_aligned("lon_max", bounds.lon_max, area_name)


def select_areas(areas: list[Area], selected_names: list[str]) -> list[Area]:
    if selected_names:
        area_by_name = {area.name: area for area in areas}
        unknown = [name for name in selected_names if name not in area_by_name]
        if unknown:
            raise RuntimeError(
                f"Unknown area(s): {', '.join(unknown)}. "
                f"Available areas: {', '.join(area_by_name)}"
            )
        return [area_by_name[name] for name in selected_names]

    return [area for area in areas if area.enabled]


def ensure_coord_id_index(conn: sqlite3.Connection, table: str) -> None:
    index_name = "idx_" + re.sub(r"[^A-Za-z0-9_]+", "_", table).strip("_") + "_coord_id"
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {quote_identifier(index_name)} "
        f"ON {quote_identifier(table)}(coord_id)"
    )
    conn.commit()


def has_coord_id_index(conn: sqlite3.Connection, table: str) -> bool:
    indexes = conn.execute(f"PRAGMA index_list({quote_identifier(table)})").fetchall()
    for index in indexes:
        index_name = index[1]
        columns = [
            row[2]
            for row in conn.execute(
                f"PRAGMA index_info({quote_identifier(index_name)})"
            ).fetchall()
        ]
        if "coord_id" in columns:
            return True
    return False


def point_on_segment(
    lon: float,
    lat: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    x1, y1 = start
    x2, y2 = end
    cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
    if abs(cross) > 1e-10:
        return False

    return (
        min(x1, x2) - 1e-10 <= lon <= max(x1, x2) + 1e-10
        and min(y1, y2) - 1e-10 <= lat <= max(y1, y2) + 1e-10
    )


def point_in_polygon(
    lon: float,
    lat: float,
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    inside = False
    for start, end in zip(polygon, polygon[1:]):
        if point_on_segment(lon, lat, start, end):
            return True

        x1, y1 = start
        x2, y2 = end
        crosses_lat = (y1 > lat) != (y2 > lat)
        if crosses_lat:
            crossing_lon = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < crossing_lon:
                inside = not inside
    return inside


def point_in_area(area: Area, lon: float, lat: float) -> bool:
    bounds = area.bounds
    if not (
        bounds.lat_min <= lat <= bounds.lat_max
        and bounds.lon_min <= lon <= bounds.lon_max
    ):
        return False

    if area.polygon is None:
        return True

    return point_in_polygon(lon, lat, area.polygon)


def area_candidate_rows(
    conn: sqlite3.Connection,
    coord_table: str,
    areas: list[Area],
) -> list[tuple[int, float, float]]:
    coord_table_sql = quote_identifier(coord_table)
    rows_by_coord_id: dict[int, tuple[int, float, float]] = {}
    for area in progress_iter(
        areas,
        desc="drop_areas read coord_data",
        total=len(areas),
        unit="area",
    ):
        bounds = area.bounds
        rows = conn.execute(
            f"""
            SELECT coord_id, lat, lon
            FROM {coord_table_sql}
            WHERE lat >= ?
              AND lat <= ?
              AND lon >= ?
              AND lon <= ?
            """,
            (bounds.lat_min, bounds.lat_max, bounds.lon_min, bounds.lon_max),
        ).fetchall()
        for coord_id, lat, lon in rows:
            rows_by_coord_id[int(coord_id)] = (int(coord_id), float(lat), float(lon))
    return list(rows_by_coord_id.values())


def matching_coord_ids(
    conn: sqlite3.Connection,
    coord_table: str,
    areas: list[Area],
) -> list[int]:
    coord_ids = []
    candidate_rows = area_candidate_rows(conn, coord_table, areas)
    for coord_id, lat, lon in progress_iter(
        candidate_rows,
        desc="drop_areas test coords",
        total=len(candidate_rows),
        unit="coord",
    ):
        if any(point_in_area(area, lon=lon, lat=lat) for area in areas):
            coord_ids.append(coord_id)
    return sorted(set(coord_ids))


def create_matching_coord_table(
    conn: sqlite3.Connection,
    coord_table: str,
    areas: list[Area],
) -> tuple[str, list[int]]:
    temp_table = "drop_areas_matching_coord_ids"
    temp_table_sql = quote_identifier(temp_table)
    coord_ids = matching_coord_ids(conn, coord_table, areas)
    conn.execute(f"DROP TABLE IF EXISTS temp.{temp_table_sql}")
    conn.execute(f"CREATE TEMP TABLE {temp_table_sql} (coord_id INTEGER PRIMARY KEY)")
    if coord_ids:
        conn.executemany(
            f"INSERT INTO {temp_table_sql} (coord_id) VALUES (?)",
            [(coord_id,) for coord_id in coord_ids],
        )
    return temp_table, coord_ids


def count_matching_rows(
    conn: sqlite3.Connection,
    table: str,
    temp_coord_table: str,
) -> int:
    table_sql = quote_identifier(table)
    temp_coord_table_sql = quote_identifier(temp_coord_table)
    return int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {table_sql}
            WHERE coord_id IN (SELECT coord_id FROM {temp_coord_table_sql})
            """
        ).fetchone()[0]
    )


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    return int(
        conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()[0]
    )


def id_scan_bounds(conn: sqlite3.Connection, table: str) -> tuple[int, int] | None:
    table_sql = quote_identifier(table)
    row = conn.execute(f"SELECT id FROM {table_sql} LIMIT 1").fetchone()
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


def delete_matching_ec_rows_by_id_scan(
    conn: sqlite3.Connection,
    table: str,
    temp_coord_table: str,
    coords_to_drop: int,
    chunk_size: int,
) -> int:
    if coords_to_drop == 0:
        return 0

    bounds = id_scan_bounds(conn, table)
    if bounds is None:
        return 0

    min_id, max_id = bounds
    total_id_slots = max_id - min_id + 1
    table_sql = quote_identifier(table)
    temp_coord_table_sql = quote_identifier(temp_coord_table)
    rows_dropped = 0
    with progress_bar("drop_areas ec_data scan/delete", total_id_slots, unit="row-id") as progress:
        for start_id, end_id in id_windows(min_id, max_id, chunk_size):
            progress.set_postfix(id=f"{start_id}-{end_id}", dropped=rows_dropped)
            progress.refresh()
            cursor = conn.execute(
                f"""
                DELETE FROM {table_sql}
                WHERE id >= ?
                  AND id <= ?
                  AND coord_id IN (SELECT coord_id FROM {temp_coord_table_sql})
                """,
                (start_id, end_id),
            )
            rows_dropped += cursor.rowcount
            progress.update(end_id - start_id + 1)
            progress.set_postfix(id=f"{start_id}-{end_id}", dropped=rows_dropped)
    return rows_dropped


def describe_area(area: Area) -> str:
    bounds = area.bounds
    return (
        f"{area.name}: lat [{bounds.lat_min:g}, {bounds.lat_max:g}], "
        f"lon [{bounds.lon_min:g}, {bounds.lon_max:g}]"
    )


def main() -> int:
    args = parse_args()
    args.db_path = resolve_path(args.db_path)
    args.areas_config = resolve_path(args.areas_config)
    ensure_valid_args(args)

    total_steps = 4 if args.dry_run else 4 + int(args.vacuum)
    with step_progress("drop_areas", total_steps) as progress:
        areas = select_areas(load_areas(args.areas_config), args.area)
        progress.update()
        if not areas:
            raise SystemExit("No enabled or selected areas to drop.")

        with sqlite3.connect(args.db_path) as conn:
            temp_coord_table, coord_ids_to_drop = create_matching_coord_table(
                conn, args.coord_table, areas
            )
            coords_to_drop = len(coord_ids_to_drop)
            progress.set_postfix(coords=coords_to_drop)
            progress.update()

            print(f"DB: {args.db_path}", flush=True)
            print(f"Table to delete from: {args.table}", flush=True)
            print(f"Coord table used for lookup only: {args.coord_table}", flush=True)
            print(f"Areas config: {args.areas_config}", flush=True)
            print("Selected areas:", flush=True)
            for area in areas:
                print(f"  - {describe_area(area)}", flush=True)
            print(f"Matching coord_data rows: {coords_to_drop}", flush=True)
            bounds = id_scan_bounds(conn, args.table)
            if bounds is None:
                print("ec_data id scan range: empty table", flush=True)
            else:
                min_id, max_id = bounds
                print(
                    f"ec_data id scan range: {min_id}-{max_id} "
                    f"({max_id - min_id + 1} id values)",
                    flush=True,
                )
                print(
                    f"Will scan/delete ec_data in chunks of "
                    f"{args.delete_chunk_size} id values.",
                    flush=True,
                )

            if args.dry_run:
                if has_coord_id_index(conn, args.table):
                    total_rows = count_rows(conn, args.table)
                    progress.update()
                    rows_to_drop = count_matching_rows(
                        conn,
                        args.table,
                        temp_coord_table,
                    )
                    progress.update()
                    rows_to_keep = total_rows - rows_to_drop
                    print(f"Rows before: {total_rows}", flush=True)
                    print(f"Rows to drop from ec_data: {rows_to_drop}", flush=True)
                    print(f"Rows to keep: {rows_to_keep}", flush=True)
                else:
                    progress.update(2)
                    print(
                        "Rows to drop: skipped because ec_data.coord_id is not indexed. "
                        "A real run will create the index before deleting rows.",
                        flush=True,
                    )
                print("Dry run only; no rows were deleted.", flush=True)
                return 0

            ensure_coord_id_index(conn, args.table)
            progress.update()
            conn.execute("BEGIN")
            try:
                rows_dropped = delete_matching_ec_rows_by_id_scan(
                    conn,
                    args.table,
                    temp_coord_table,
                    coords_to_drop,
                    args.delete_chunk_size,
                )
                progress.set_postfix(coords=coords_to_drop, dropped=rows_dropped)
                progress.update()
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        if args.vacuum:
            with sqlite3.connect(args.db_path) as conn:
                conn.execute("VACUUM")
            progress.update()

    print(f"Dropped {rows_dropped} rows from {args.table}.", flush=True)
    print("Skipped id rebuild; run rebuild_ids_era5 after all filters.", flush=True)
    if args.vacuum:
        print("Vacuumed database.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
