#!/usr/bin/env python3
"""Plot ERA5 cells that are present at one hour and missing at the next.

The raw ERA5 DB has no composite coord/timestamp index, so this script uses the
indexed processed DB only to locate candidate ec_data row ids quickly. It then
verifies those row ids against experiments/data/raw_era5/era5.db by primary key
before drawing the map.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DB = Path("/home/l/luislara/links/projects/aip-pal/luislara/ep/data/2016_2017/era5_2016_2017.db")
DEFAULT_INDEXED_DB = Path("/home/l/luislara/links/projects/aip-pal/luislara/ep/data/2016_2017/era5_2016_2017.db")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "era5_pipeline" / "plot" / "tmp"
DEFAULT_PRESENT_TIMESTAMP = 20170103080000
DEFAULT_MISSING_TIMESTAMP = 20170103090000
DEFAULT_EXCLUDED_IGBP = ("WAT", "SNO", "BSV", "URB", "CRO", "CVM")
RAW_VERIFY_MAX_ID_SPAN = 250_000


@dataclass(frozen=True)
class Coord:
    coord_id: int
    lat: float
    lon: float
    igbp: str


@dataclass(frozen=True)
class ClassifiedCoord:
    coord: Coord
    row_id: int
    row_timestamp: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a world map of ERA5 raw-verified cells that are present at "
            "one timestamp and missing at another."
        )
    )
    parser.add_argument("--raw-db", type=Path, default=DEFAULT_RAW_DB)
    parser.add_argument("--indexed-db", type=Path, default=DEFAULT_INDEXED_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--present-timestamp", type=int, default=DEFAULT_PRESENT_TIMESTAMP)
    parser.add_argument("--missing-timestamp", type=int, default=DEFAULT_MISSING_TIMESTAMP)
    parser.add_argument(
        "--exclude-igbp",
        nargs="*",
        default=DEFAULT_EXCLUDED_IGBP,
        help="IGBP classes excluded from the plotted coordinate set.",
    )
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--raw-neighborhood-samples",
        type=int,
        default=25,
        help="Number of hole cells to verify by raw primary-key neighborhood.",
    )
    parser.add_argument(
        "--verify-available-row-ids",
        action="store_true",
        help=(
            "Also verify every blue context row id against the raw DB. "
            "Default skips this because it causes many random reads."
        ),
    )
    parser.add_argument(
        "--skip-raw-verification",
        action="store_true",
        help="Write the PNG/report/cache without raw ec_data row-id verification.",
    )
    parser.add_argument(
        "--verify-cache",
        type=Path,
        default=None,
        help="Only verify a previously written *_holes.csv cache against the raw DB.",
    )
    parser.add_argument(
        "--verify-cache-start",
        type=int,
        default=0,
        help="Start offset for --verify-cache chunking.",
    )
    parser.add_argument(
        "--verify-cache-limit",
        type=int,
        default=None,
        help="Maximum number of cached rows to verify in this pass.",
    )
    return parser.parse_args()


def connect_readonly(path: Path) -> sqlite3.Connection:
    path = path.expanduser().resolve()
    uri = f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -131072")
    return conn


def attach_raw_coord_db(conn: sqlite3.Connection, raw_db: Path, schema: str = "rawdb") -> None:
    uri = f"file:{quote(str(raw_db.expanduser().resolve()), safe='/')}?mode=ro&immutable=1"
    escaped_uri = uri.replace("'", "''")
    escaped_schema = schema.replace('"', '""')
    conn.execute(f"ATTACH DATABASE '{escaped_uri}' AS \"{escaped_schema}\"")


def timestamp_label(timestamp: int) -> str:
    return datetime.strptime(str(timestamp), "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M")


def add_hours(timestamp: int, hours: int) -> int:
    dt = datetime.strptime(str(timestamp), "%Y%m%d%H%M%S") + timedelta(hours=hours)
    return int(dt.strftime("%Y%m%d%H%M%S"))


def output_stem(present_timestamp: int, missing_timestamp: int) -> str:
    return f"raw_era5_holes_{missing_timestamp}_vs_{present_timestamp}"


def load_coords(raw_conn: sqlite3.Connection, excluded_igbp: set[str]) -> list[Coord]:
    rows = raw_conn.execute(
        """
        SELECT coord_id, lat, lon, igbp
        FROM coord_data
        WHERE coord_id IS NOT NULL
          AND lat IS NOT NULL
          AND lon IS NOT NULL
        ORDER BY coord_id;
        """
    ).fetchall()
    coords = []
    for coord_id, lat, lon, igbp in rows:
        igbp_text = "" if igbp is None else str(igbp).upper()
        if igbp_text in excluded_igbp:
            continue
        coords.append(Coord(int(coord_id), float(lat), float(lon), igbp_text))
    return coords


def classify_raw_coords_with_indexed_db(
    indexed_conn: sqlite3.Connection,
    raw_db: Path,
    excluded_igbp: tuple[str, ...],
    present_timestamp: int,
    missing_timestamp: int,
) -> tuple[int, list[ClassifiedCoord], list[ClassifiedCoord]]:
    attach_raw_coord_db(indexed_conn, raw_db)
    excluded_sql = ""
    params: list[object] = [present_timestamp, missing_timestamp]
    if excluded_igbp:
        placeholders = ",".join("?" for _ in excluded_igbp)
        excluded_sql = f"AND UPPER(COALESCE(c.igbp, '')) NOT IN ({placeholders})"
        params.extend(excluded_igbp)

    rows = indexed_conn.execute(
        f"""
        SELECT
            c.coord_id,
            c.lat,
            c.lon,
            COALESCE(c.igbp, '') AS igbp,
            (
                SELECT MIN(ec_present.id)
                FROM main.ec_data ec_present
                WHERE ec_present.coord_id = c.coord_id
                  AND ec_present.timestamp = ?
            ) AS present_id,
            (
                SELECT MIN(ec_missing.id)
                FROM main.ec_data ec_missing
                WHERE ec_missing.coord_id = c.coord_id
                  AND ec_missing.timestamp = ?
            ) AS missing_id
        FROM rawdb.coord_data c
        WHERE c.coord_id IS NOT NULL
          AND c.lat IS NOT NULL
          AND c.lon IS NOT NULL
          {excluded_sql}
        ORDER BY c.coord_id;
        """,
        params,
    ).fetchall()

    holes: list[ClassifiedCoord] = []
    available_at_missing: list[ClassifiedCoord] = []
    for coord_id, lat, lon, igbp, present_id, missing_id in rows:
        coord = Coord(int(coord_id), float(lat), float(lon), str(igbp).upper())
        if missing_id is not None:
            available_at_missing.append(
                ClassifiedCoord(coord=coord, row_id=int(missing_id), row_timestamp=missing_timestamp)
            )
        elif present_id is not None:
            holes.append(
                ClassifiedCoord(coord=coord, row_id=int(present_id), row_timestamp=present_timestamp)
            )

    return len(rows), available_at_missing, holes


def classify_coords(
    indexed_conn: sqlite3.Connection,
    coords: list[Coord],
    present_timestamp: int,
    missing_timestamp: int,
) -> tuple[list[ClassifiedCoord], list[ClassifiedCoord]]:
    holes: list[ClassifiedCoord] = []
    available_at_missing: list[ClassifiedCoord] = []

    query = """
        SELECT timestamp, id
        FROM ec_data
        WHERE coord_id = ?
          AND timestamp IN (?, ?)
        ORDER BY timestamp, id;
    """
    for idx, coord in enumerate(coords, start=1):
        rows = indexed_conn.execute(
            query,
            (coord.coord_id, present_timestamp, missing_timestamp),
        ).fetchall()
        present_ids = [int(row_id) for ts, row_id in rows if int(ts) == present_timestamp]
        missing_ids = [int(row_id) for ts, row_id in rows if int(ts) == missing_timestamp]
        if missing_ids:
            available_at_missing.append(
                ClassifiedCoord(coord=coord, row_id=missing_ids[0], row_timestamp=missing_timestamp)
            )
        elif present_ids:
            holes.append(
                ClassifiedCoord(coord=coord, row_id=present_ids[0], row_timestamp=present_timestamp)
            )

        if idx % 25_000 == 0:
            print(
                f"classified {idx:,}/{len(coords):,} coords: "
                f"available={len(available_at_missing):,} holes={len(holes):,}",
                flush=True,
            )

    return available_at_missing, holes


def iter_rowid_ranges(records: list[ClassifiedCoord], max_span: int):
    sorted_records = sorted(records, key=lambda record: record.row_id)
    chunk: list[ClassifiedCoord] = []
    chunk_min = None
    for record in sorted_records:
        if chunk and chunk_min is not None and record.row_id - chunk_min > max_span:
            yield chunk
            chunk = []
            chunk_min = None
        if not chunk:
            chunk_min = record.row_id
        chunk.append(record)
    if chunk:
        yield chunk


def verify_raw_row_ids(
    raw_conn: sqlite3.Connection,
    records: list[ClassifiedCoord],
    label: str,
) -> int:
    expected = {
        record.row_id: (record.coord.coord_id, record.row_timestamp)
        for record in records
    }
    verified_ids: set[int] = set()
    for chunk in iter_rowid_ranges(records, RAW_VERIFY_MAX_ID_SPAN):
        chunk_ids = {record.row_id for record in chunk}
        row_id_min = min(chunk_ids)
        row_id_max = max(chunk_ids)
        raw_rows = raw_conn.execute(
            """
            SELECT id, coord_id, timestamp
            FROM ec_data
            WHERE id BETWEEN ? AND ?;
            """,
            (row_id_min, row_id_max),
        ).fetchall()
        for row_id, coord_id, timestamp in raw_rows:
            row_id = int(row_id)
            if row_id not in chunk_ids:
                continue
            expected_coord_id, expected_timestamp = expected[int(row_id)]
            if int(coord_id) != expected_coord_id or int(timestamp) != expected_timestamp:
                raise RuntimeError(
                    f"Raw DB mismatch for {label} row id {row_id}: "
                    f"expected coord_id={expected_coord_id}, timestamp={expected_timestamp}; "
                    f"got coord_id={coord_id}, timestamp={timestamp}"
                )
            verified_ids.add(row_id)
    verified = len(verified_ids)
    if verified != len(records):
        raise RuntimeError(
            f"Raw DB verification for {label} found {verified:,}/{len(records):,} row ids."
        )
    return verified


def verify_raw_hole_neighborhoods(
    raw_conn: sqlite3.Connection,
    _indexed_conn: sqlite3.Connection,
    holes: list[ClassifiedCoord],
    missing_timestamp: int,
    samples: int,
) -> tuple[int, list[str]]:
    if samples <= 0 or not holes:
        return 0, []

    checked = 0
    examples: list[str] = []
    for record in holes[:samples]:
        raw_rows = raw_conn.execute(
            """
            SELECT id, coord_id, timestamp
            FROM ec_data
            WHERE id BETWEEN ? AND ?
            ORDER BY id;
            """,
            (record.row_id - 12, record.row_id + 12),
        ).fetchall()
        same_coord_timestamps = [
            int(timestamp)
            for _row_id, coord_id, timestamp in raw_rows
            if int(coord_id) == record.coord.coord_id
        ]
        if missing_timestamp in same_coord_timestamps:
            raise RuntimeError(
                f"Raw DB neighborhood contains missing timestamp {missing_timestamp} "
                f"for coord_id={record.coord.coord_id}."
            )
        checked += 1
        if len(examples) < 5:
            examples.append(
                f"coord_id={record.coord.coord_id} lat={record.coord.lat:.2f} "
                f"lon={record.coord.lon:.2f} raw timestamps={same_coord_timestamps[:12]}"
            )
    return checked, examples


def lonlat_to_pixel(lon: float, lat: float, frame: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, right, bottom = frame
    x = left + (lon + 180.0) / 360.0 * (right - left)
    y = top + (90.0 - lat) / 180.0 * (bottom - top)
    return int(round(x)), int(round(y))


def draw_points(
    draw: ImageDraw.ImageDraw,
    records: list[ClassifiedCoord],
    frame: tuple[int, int, int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    for record in records:
        x, y = lonlat_to_pixel(record.coord.lon, record.coord.lat, frame)
        if radius <= 0:
            draw.point((x, y), fill=color)
        else:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_region_box(
    draw: ImageDraw.ImageDraw,
    frame: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    lat_min, lat_max, lon_min, lon_max = bounds
    x0, y0 = lonlat_to_pixel(lon_min, lat_max, frame)
    x1, y1 = lonlat_to_pixel(lon_max, lat_min, frame)
    draw.rectangle((x0, y0, x1, y1), outline=(35, 35, 35), width=2)
    draw.text((x0 + 5, max(y0 - 16, frame[1] + 2)), label, fill=(35, 35, 35), font=font)


def render_map(
    output_png: Path,
    available_at_missing: list[ClassifiedCoord],
    holes: list[ClassifiedCoord],
    present_timestamp: int,
    missing_timestamp: int,
    width: int,
    height: int,
) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    frame = (80, 70, width - 45, height - 95)
    left, top, right, bottom = frame

    draw.rectangle(frame, outline=(60, 60, 60), width=2)
    for lon in range(-180, 181, 30):
        x, _ = lonlat_to_pixel(lon, 0, frame)
        draw.line((x, top, x, bottom), fill=(225, 225, 225), width=1)
        draw.text((x - 13, bottom + 8), str(lon), fill=(70, 70, 70), font=font)
    for lat in range(-60, 91, 30):
        _, y = lonlat_to_pixel(0, lat, frame)
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((left - 42, y - 6), str(lat), fill=(70, 70, 70), font=font)

    draw_points(draw, available_at_missing, frame, color=(70, 112, 255), radius=0)
    draw_points(draw, holes, frame, color=(230, 35, 35), radius=1)

    region_boxes = {
        "India": (5.0, 35.0, 65.0, 95.0),
        "Australia": (-40.0, -15.0, 125.0, 145.0),
        "East Canada": (45.0, 65.0, -80.0, -50.0),
    }
    for label, bounds in region_boxes.items():
        draw_region_box(draw, frame, bounds, label, font)

    title = (
        "Raw ERA5 DB hole check: cells present at "
        f"{timestamp_label(present_timestamp)} but missing at {timestamp_label(missing_timestamp)}"
    )
    draw.text((left, 25), title, fill=(20, 20, 20), font=title_font)

    legend_y = bottom + 38
    draw.rectangle((left, legend_y, left + 16, legend_y + 16), fill=(70, 112, 255))
    draw.text(
        (left + 24, legend_y + 1),
        f"present at missing timestamp: {len(available_at_missing):,}",
        fill=(20, 20, 20),
        font=font,
    )
    draw.rectangle((left + 330, legend_y, left + 346, legend_y + 16), fill=(230, 35, 35))
    draw.text(
        (left + 354, legend_y + 1),
        f"holes: present previous hour, absent next hour: {len(holes):,}",
        fill=(20, 20, 20),
        font=font,
    )

    image.save(output_png)


def write_hole_cache(output_csv: Path, holes: list[ClassifiedCoord]) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("coord_id", "lat", "lon", "igbp", "row_id", "row_timestamp"))
        for record in holes:
            writer.writerow(
                (
                    record.coord.coord_id,
                    f"{record.coord.lat:.8f}",
                    f"{record.coord.lon:.8f}",
                    record.coord.igbp,
                    record.row_id,
                    record.row_timestamp,
                )
            )


def load_hole_cache(input_csv: Path) -> list[ClassifiedCoord]:
    holes: list[ClassifiedCoord] = []
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            coord = Coord(
                coord_id=int(row["coord_id"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                igbp=row["igbp"],
            )
            holes.append(
                ClassifiedCoord(
                    coord=coord,
                    row_id=int(row["row_id"]),
                    row_timestamp=int(row["row_timestamp"]),
                )
            )
    return holes


def write_report(
    output_report: Path,
    raw_db: Path,
    indexed_db: Path,
    coords_count: int,
    available_count: int,
    holes_count: int,
    raw_available_verified: int | None,
    raw_holes_verified: int | None,
    neighborhood_checked: int | None,
    neighborhood_examples: list[str],
    present_timestamp: int,
    missing_timestamp: int,
    excluded_igbp: tuple[str, ...],
) -> None:
    lines = [
        "Raw ERA5 hole map diagnostic",
        "",
        f"raw_db: {raw_db.resolve()}",
        f"indexed_db_used_for_fast_lookup: {indexed_db.resolve()}",
        f"present_timestamp: {present_timestamp} ({timestamp_label(present_timestamp)})",
        f"missing_timestamp: {missing_timestamp} ({timestamp_label(missing_timestamp)})",
        f"excluded_igbp: {', '.join(excluded_igbp) if excluded_igbp else '<none>'}",
        "",
        f"coords_considered_from_raw_coord_data: {coords_count:,}",
        f"coords_present_at_missing_timestamp: {available_count:,}",
        f"holes_present_at_previous_missing_at_next: {holes_count:,}",
        "",
        "raw_verified_present_at_missing_row_ids: "
        + (
            f"{raw_available_verified:,}"
            if raw_available_verified is not None
            else "skipped by default; pass --verify-available-row-ids"
        ),
        "raw_verified_hole_present_timestamp_row_ids: "
        + (
            f"{raw_holes_verified:,}"
            if raw_holes_verified is not None
            else "not run in this pass"
        ),
        "raw_neighborhood_absence_checks: "
        + (
            f"{neighborhood_checked:,}"
            if neighborhood_checked is not None
            else "not run in this pass"
        ),
        "",
        "raw_neighborhood_examples:",
    ]
    lines.extend(f"  - {example}" for example in neighborhood_examples)
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_verify_report(
    output_report: Path,
    raw_db: Path,
    cache_path: Path,
    cache_start: int,
    cache_limit: int | None,
    cache_total: int,
    raw_holes_verified: int,
    neighborhood_checked: int,
    neighborhood_examples: list[str],
) -> None:
    lines = [
        "Raw ERA5 hole cache verification",
        "",
        f"raw_db: {raw_db.resolve()}",
        f"hole_cache: {cache_path.resolve()}",
        f"cache_total_rows: {cache_total:,}",
        f"cache_start: {cache_start:,}",
        "cache_limit: " + (f"{cache_limit:,}" if cache_limit is not None else "all remaining"),
        f"raw_verified_hole_present_timestamp_row_ids: {raw_holes_verified:,}",
        f"raw_neighborhood_absence_checks: {neighborhood_checked:,}",
        "",
        "raw_neighborhood_examples:",
    ]
    lines.extend(f"  - {example}" for example in neighborhood_examples)
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_db = args.raw_db.expanduser().resolve()
    indexed_db = args.indexed_db.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_db.exists():
        raise FileNotFoundError(f"Raw DB not found: {raw_db}")
    if not indexed_db.exists():
        raise FileNotFoundError(f"Indexed DB not found: {indexed_db}")

    if args.verify_cache is not None:
        cache_path = args.verify_cache.expanduser().resolve()
        all_holes = load_hole_cache(cache_path)
        if args.verify_cache_start < 0:
            raise ValueError("--verify-cache-start must be >= 0")
        if args.verify_cache_limit is not None and args.verify_cache_limit <= 0:
            raise ValueError("--verify-cache-limit must be positive when provided")
        end = (
            len(all_holes)
            if args.verify_cache_limit is None
            else min(len(all_holes), args.verify_cache_start + args.verify_cache_limit)
        )
        holes = all_holes[args.verify_cache_start : end]
        suffix = (
            "_raw_verify"
            if args.verify_cache_start == 0 and args.verify_cache_limit is None
            else f"_raw_verify_{args.verify_cache_start:05d}_{end:05d}"
        )
        verify_report = cache_path.with_name(cache_path.stem + f"{suffix}.txt")
        print(
            f"verifying cached holes {args.verify_cache_start:,}:{end:,} "
            f"of {len(all_holes):,} against raw db: {raw_db}",
            flush=True,
        )
        with connect_readonly(raw_db) as raw_conn, connect_readonly(indexed_db) as indexed_conn:
            raw_holes_verified = verify_raw_row_ids(raw_conn, holes, label="holes")
            neighborhood_checked, neighborhood_examples = verify_raw_hole_neighborhoods(
                raw_conn,
                indexed_conn,
                holes,
                args.missing_timestamp,
                args.raw_neighborhood_samples,
            )
        write_verify_report(
            verify_report,
            raw_db,
            cache_path,
            args.verify_cache_start,
            args.verify_cache_limit,
            len(all_holes),
            raw_holes_verified,
            neighborhood_checked,
            neighborhood_examples,
        )
        print(f"wrote {verify_report}", flush=True)
        return 0

    excluded_igbp = tuple(code.upper() for code in args.exclude_igbp)
    stem = output_stem(args.present_timestamp, args.missing_timestamp)
    output_png = output_dir / f"{stem}.png"
    output_report = output_dir / f"{stem}.txt"
    output_holes_csv = output_dir / f"{stem}_holes.csv"

    print(f"raw db: {raw_db}", flush=True)
    print(f"indexed db for fast row-id lookup: {indexed_db}", flush=True)
    print(f"output png: {output_png}", flush=True)

    with connect_readonly(raw_db) as raw_conn, connect_readonly(indexed_db) as indexed_conn:
        coords_count, available_at_missing, holes = classify_raw_coords_with_indexed_db(
            indexed_conn,
            raw_db,
            excluded_igbp,
            args.present_timestamp,
            args.missing_timestamp,
        )
        print(f"loaded {coords_count:,} coords from raw coord_data after exclusions", flush=True)
        print(
            f"classification done: available={len(available_at_missing):,}, "
            f"holes={len(holes):,}",
            flush=True,
        )

        raw_holes_verified = None
        raw_available_verified = None
        neighborhood_checked = None
        neighborhood_examples: list[str] = []
        if not args.skip_raw_verification:
            raw_holes_verified = verify_raw_row_ids(raw_conn, holes, label="holes")
            raw_available_verified = (
                verify_raw_row_ids(raw_conn, available_at_missing, label="available")
                if args.verify_available_row_ids
                else None
            )
            neighborhood_checked, neighborhood_examples = verify_raw_hole_neighborhoods(
                raw_conn,
                indexed_conn,
                holes,
                args.missing_timestamp,
                args.raw_neighborhood_samples,
            )

    render_map(
        output_png,
        available_at_missing,
        holes,
        args.present_timestamp,
        args.missing_timestamp,
        args.width,
        args.height,
    )
    write_hole_cache(output_holes_csv, holes)
    write_report(
        output_report,
        raw_db,
        indexed_db,
        coords_count,
        len(available_at_missing),
        len(holes),
        raw_available_verified,
        raw_holes_verified,
        neighborhood_checked,
        neighborhood_examples,
        args.present_timestamp,
        args.missing_timestamp,
        excluded_igbp,
    )

    print(f"wrote {output_png}", flush=True)
    print(f"wrote {output_holes_csv}", flush=True)
    print(f"wrote {output_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
