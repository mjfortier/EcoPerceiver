#!/usr/bin/env python3
"""Drop full geographic areas from the ERA5/ec_data SQLite table."""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = Path("/home/l/luislara/links/projects/aip-pal/luislara/ep/data/era5.db")
DEFAULT_AREAS_CONFIG_PATH = SCRIPT_DIR / "drop_areas_config.yml"
DEFAULT_TABLE = "ec_data"
DEFAULT_COORD_TABLE = "coord_data"
ERA5_GRID_DEGREES = 0.25


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
    if not args.db_path.exists():
        raise SystemExit(f"Database does not exist: {args.db_path}")
    if not args.areas_config.exists():
        raise SystemExit(f"Areas config does not exist: {args.areas_config}")


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


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    return [row[1] for row in rows]


def require_tables(conn: sqlite3.Connection, table: str, coord_table: str) -> None:
    table_cols = table_columns(conn, table)
    if not table_cols:
        raise RuntimeError(f"Table does not exist or has no columns: {table}")
    if "id" not in table_cols:
        raise RuntimeError(f"Table {table!r} must contain an id column.")
    if "coord_id" not in table_cols:
        raise RuntimeError(f"Table {table!r} must contain a coord_id column.")

    coord_cols = table_columns(conn, coord_table)
    if not coord_cols:
        raise RuntimeError(f"Table does not exist or has no columns: {coord_table}")
    for column in ("coord_id", "lat", "lon"):
        if column not in coord_cols:
            raise RuntimeError(f"Table {coord_table!r} must contain {column!r}.")


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
    for area in areas:
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
    for coord_id, lat, lon in area_candidate_rows(conn, coord_table, areas):
        if any(point_in_area(area, lon=lon, lat=lat) for area in areas):
            coord_ids.append(coord_id)
    return sorted(set(coord_ids))


def create_matching_coord_table(
    conn: sqlite3.Connection,
    coord_table: str,
    areas: list[Area],
) -> tuple[str, int]:
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
    return temp_table, len(coord_ids)


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


def delete_matching_rows(
    conn: sqlite3.Connection,
    table: str,
    temp_coord_table: str,
) -> int:
    table_sql = quote_identifier(table)
    temp_coord_table_sql = quote_identifier(temp_coord_table)
    cursor = conn.execute(
        f"""
        DELETE FROM {table_sql}
        WHERE coord_id IN (SELECT coord_id FROM {temp_coord_table_sql})
        """
    )
    return cursor.rowcount


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

    areas = select_areas(load_areas(args.areas_config), args.area)
    if not areas:
        raise SystemExit("No enabled or selected areas to drop.")

    with sqlite3.connect(args.db_path) as conn:
        require_tables(conn, args.table, args.coord_table)
        temp_coord_table, coords_to_drop = create_matching_coord_table(
            conn, args.coord_table, areas
        )

        print(f"DB: {args.db_path}")
        print(f"Table: {args.table}")
        print(f"Coord table: {args.coord_table}")
        print(f"Areas config: {args.areas_config}")
        print("Selected areas:")
        for area in areas:
            print(f"  - {describe_area(area)}")
        print(f"Matching coord rows: {coords_to_drop}")

        if args.dry_run:
            if has_coord_id_index(conn, args.table):
                total_rows = count_rows(conn, args.table)
                rows_to_drop = count_matching_rows(
                    conn,
                    args.table,
                    temp_coord_table,
                )
                rows_to_keep = total_rows - rows_to_drop
                print(f"Rows before: {total_rows}")
                print(f"Rows to drop: {rows_to_drop}")
                print(f"Rows to keep: {rows_to_keep}")
            else:
                print(
                    "Rows to drop: skipped because ec_data.coord_id is not indexed. "
                    "A real run will create the index before deleting rows."
                )
            print("Dry run only; no rows were deleted.")
            return 0

        ensure_coord_id_index(conn, args.table)
        conn.execute("BEGIN")
        try:
            rows_dropped = delete_matching_rows(
                conn,
                args.table,
                temp_coord_table,
            )
            if rows_dropped > 0 and not args.no_rebuild_ids:
                rebuild_ids(conn, args.table)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    if args.vacuum:
        with sqlite3.connect(args.db_path) as conn:
            conn.execute("VACUUM")

    print(f"Dropped {rows_dropped} rows from {args.table}.")
    if rows_dropped > 0 and not args.no_rebuild_ids:
        print("Rebuilt ec_data ids ordered by coord_id, timestamp, id.")
    if args.vacuum:
        print("Vacuumed database.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
