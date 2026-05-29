#!/usr/bin/env python3
"""Render an IGBP map from coord_data in an ERA5 SQLite database."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Iterable
from urllib.parse import quote
from urllib.request import urlretrieve

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python dependency: Pillow. Install pillow or activate the "
        "project environment before running this script."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = Path("/home/l/luislara/links/projects/aip-pal/luislara/ep/data/era5.db")
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "igbp_coord_map.png"
DEFAULT_TABLE = "coord_data"
DEFAULT_COASTLINE_CACHE = (
    Path.home() / ".cache" / "ecoperceiver" / "ne_110m_admin_0_countries.geojson"
)
NATURAL_EARTH_COUNTRIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_110m_admin_0_countries.geojson"
)
NULL_LABEL = "NULL"
BACKGROUND_LABELS = {"WAT", "SNO", "BSV"}

IGBP_ORDER = (
    NULL_LABEL,
    "ENF",
    "EBF",
    "DNF",
    "DBF",
    "MF",
    "CSH",
    "OSH",
    "WSA",
    "SAV",
    "GRA",
    "WET",
    "CRO",
    "URB",
    "CVM",
    "WAT",
    "SNO",
    "BSV",
)

# Palette matched to the reference map style used for IGBP classes.
IGBP_COLORS = {
    NULL_LABEL: "#8F9693",
    "ENF": "#006837",
    "EBF": "#008B45",
    "DNF": "#A6CEE3",
    "DBF": "#1F78B4",
    "MF": "#B2DF8A",
    "CSH": "#6A3D9A",
    "OSH": "#FB9A99",
    "WSA": "#E31A1C",
    "SAV": "#FF7F00",
    "GRA": "#FDBF6F",
    "WET": "#CAB2D6",
    "CRO": "#8073AC",
    "URB": "#FFFF99",
    "CVM": "#B15928",
    "WAT": "#FFFFFF",
    "SNO": "#FFFFFF",
    "BSV": "#FFFFFF",
}
IGBP_FULL_NAMES = {
    NULL_LABEL: "Changed region",
    "ENF": "Evergreen needleleaf forest",
    "EBF": "Evergreen broadleaf forest",
    "DNF": "Deciduous needleleaf forest",
    "DBF": "Deciduous broadleaf forest",
    "MF": "Mixed forest",
    "CSH": "Closed shrublands",
    "OSH": "Open shrublands",
    "WSA": "Woody savannas",
    "SAV": "Savannas",
    "GRA": "Grasslands",
    "WET": "Permanent wetlands",
    "CRO": "Croplands",
    "URB": "Urban and built-up",
    "CVM": "Cropland/Natural vegetation mosaic",
    "WAT": "Water bodies",
    "SNO": "Snow and ice",
    "BSV": "Barren or sparsely vegetated",
}
FALLBACK_COLORS = (
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#FFFF33",
    "#A65628",
    "#F781BF",
    "#999999",
)

# Robinson projection coefficients at 5 degree latitude intervals.
ROBINSON_X = np.array(
    [
        1.0000,
        0.9986,
        0.9954,
        0.9900,
        0.9822,
        0.9730,
        0.9600,
        0.9427,
        0.9216,
        0.8962,
        0.8679,
        0.8350,
        0.7986,
        0.7597,
        0.7186,
        0.6732,
        0.6213,
        0.5722,
        0.5322,
    ],
    dtype=np.float64,
)
ROBINSON_Y = np.array(
    [
        0.0000,
        0.0620,
        0.1240,
        0.1860,
        0.2480,
        0.3100,
        0.3720,
        0.4340,
        0.4958,
        0.5571,
        0.6176,
        0.6769,
        0.7346,
        0.7903,
        0.8435,
        0.8936,
        0.9394,
        0.9761,
        1.0000,
    ],
    dtype=np.float64,
)
ROBINSON_X_SCALE = 0.8487
ROBINSON_Y_SCALE = 1.3523


@dataclass(frozen=True)
class PlotArea:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass(frozen=True)
class ProjectionFrame:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    area: PlotArea
    scale: float
    used_left: float
    used_top: float
    used_width: float
    used_height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a standalone map of IGBP classes from the SQLite coord_data "
            "table. IGBP NULL values are shown as Changed region."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database to read. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"PNG path to write. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Coordinate table with lat, lon, and igbp columns. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "--projection",
        choices=("robinson", "equirectangular"),
        default="equirectangular",
        help="Map projection for the output image. Default: equirectangular.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1000,
        help="Output image width in pixels. Default: 1000.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Output image height in pixels. Default: 600.",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=2,
        help="Rendered size for each database coordinate in pixels. Default: 2.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not draw the category legend in the PNG.",
    )
    parser.add_argument(
        "--no-coastlines",
        action="store_true",
        help="Do not draw Natural Earth country/coastline outlines.",
    )
    parser.add_argument(
        "--coastline-path",
        type=Path,
        default=DEFAULT_COASTLINE_CACHE,
        help=(
            "Path to a Natural Earth admin-0 countries GeoJSON. If missing, the "
            f"script downloads and caches it. Default: {DEFAULT_COASTLINE_CACHE}"
        ),
    )
    return parser.parse_args()


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    db_uri = f"file:{quote(str(db_path), safe='/')}?mode=ro"
    return sqlite3.connect(db_uri, uri=True)


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    if not rows:
        raise RuntimeError(f"Table does not exist or has no columns: {table}")
    return {row[1] for row in rows}


def load_coordinates(
    conn: sqlite3.Connection,
    table: str,
) -> tuple[np.ndarray, np.ndarray, list[str], Counter[str], int]:
    columns = table_columns(conn, table)
    missing = {"lat", "lon", "igbp"} - columns
    if missing:
        raise RuntimeError(
            f"{table} is missing required column(s): {', '.join(sorted(missing))}"
        )

    rows = conn.execute(
        f"""
        SELECT lat, lon, igbp
        FROM {quote_identifier(table)}
        """
    )

    lats: list[float] = []
    lons: list[float] = []
    labels: list[str] = []
    counts: Counter[str] = Counter()
    skipped_missing_coords = 0

    for lat, lon, igbp in rows:
        label = NULL_LABEL if igbp is None else str(igbp)
        counts[label] += 1
        if lat is None or lon is None:
            skipped_missing_coords += 1
            continue

        lats.append(float(lat))
        lons.append(float(lon))
        labels.append(label)

    if not lats:
        raise RuntimeError(f"No plottable rows found in {table}.")

    return (
        np.asarray(lats, dtype=np.float64),
        np.asarray(lons, dtype=np.float64),
        labels,
        counts,
        skipped_missing_coords,
    )


def robinson_project(
    lon: np.ndarray | Iterable[float],
    lat: np.ndarray | Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.clip(np.asarray(lat, dtype=np.float64), -90.0, 90.0)
    abs_lat = np.abs(lat_arr)
    index = np.minimum(np.floor(abs_lat / 5.0).astype(np.int64), 17)
    fraction = (abs_lat - (index * 5.0)) / 5.0

    x_coeff = ROBINSON_X[index] + fraction * (ROBINSON_X[index + 1] - ROBINSON_X[index])
    y_coeff = ROBINSON_Y[index] + fraction * (ROBINSON_Y[index + 1] - ROBINSON_Y[index])

    x = ROBINSON_X_SCALE * x_coeff * np.radians(lon_arr)
    y = ROBINSON_Y_SCALE * y_coeff * np.sign(lat_arr)
    return x, y


def project(
    lon: np.ndarray | Iterable[float],
    lat: np.ndarray | Iterable[float],
    projection: str,
) -> tuple[np.ndarray, np.ndarray]:
    if projection == "robinson":
        return robinson_project(lon, lat)

    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    return lon_arr, lat_arr


def make_projection_frame(projection: str, area: PlotArea) -> ProjectionFrame:
    if projection == "robinson":
        xmin = -ROBINSON_X_SCALE * math.pi
        xmax = ROBINSON_X_SCALE * math.pi
        ymin = -ROBINSON_Y_SCALE
        ymax = ROBINSON_Y_SCALE
    else:
        xmin, xmax = -180.0, 180.0
        ymin, ymax = -90.0, 90.0

    proj_width = xmax - xmin
    proj_height = ymax - ymin
    scale = min(area.width / proj_width, area.height / proj_height)
    used_width = proj_width * scale
    used_height = proj_height * scale
    used_left = area.left + ((area.width - used_width) / 2.0)
    used_top = area.top + ((area.height - used_height) / 2.0)

    return ProjectionFrame(
        name=projection,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        area=area,
        scale=scale,
        used_left=used_left,
        used_top=used_top,
        used_width=used_width,
        used_height=used_height,
    )


def to_pixels(
    lon: np.ndarray | Iterable[float],
    lat: np.ndarray | Iterable[float],
    frame: ProjectionFrame,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = project(lon, lat, frame.name)
    px = frame.used_left + ((x - frame.xmin) * frame.scale)
    py = frame.used_top + ((frame.ymax - y) * frame.scale)
    return np.rint(px).astype(np.int64), np.rint(py).astype(np.int64)


def load_font(
    size: int,
    *,
    bold: bool = False,
    serif: bool = False,
) -> ImageFont.ImageFont:
    family = "DejaVuSerif" if serif else "DejaVuSans"
    suffix = "-Bold" if bold else ""
    font_name = f"{family}{suffix}.ttf"
    candidates = (
        font_name,
        f"/usr/share/fonts/truetype/dejavu/{font_name}",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def category_order(counts: Counter[str]) -> list[str]:
    known = [label for label in IGBP_ORDER if label in counts]
    extras = sorted(label for label in counts if label not in IGBP_ORDER)
    return known + extras


def legend_order(counts: Counter[str]) -> list[str]:
    return [label for label in category_order(counts) if label not in BACKGROUND_LABELS]


def color_for_label(label: str, fallback_index: int = 0) -> str:
    if label in IGBP_COLORS:
        return IGBP_COLORS[label]
    return FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: str,
    width: int = 1,
) -> None:
    if len(points) >= 2:
        draw.line(points, fill=fill, width=width)


def projected_line(
    frame: ProjectionFrame,
    *,
    lons: np.ndarray,
    lats: np.ndarray,
) -> list[tuple[int, int]]:
    px, py = to_pixels(lons, lats, frame)
    return list(zip(px.tolist(), py.tolist()))


def degree_label(value: int, *, is_lon: bool) -> str:
    absolute = abs(value)
    if value == 0:
        return "0°"
    if is_lon:
        if absolute == 180:
            return "180°"
        suffix = "W" if value < 0 else "E"
    else:
        suffix = "S" if value < 0 else "N"
    return f"{absolute}°{suffix}"


def draw_axes(
    draw: ImageDraw.ImageDraw,
    frame: ProjectionFrame,
    font: ImageFont.ImageFont,
) -> None:
    axis_color = "#111111"
    label_color = "#222222"
    left = int(round(frame.used_left))
    top = int(round(frame.used_top))
    right = int(round(frame.used_left + frame.used_width))
    bottom = int(round(frame.used_top + frame.used_height))

    draw.rectangle((left, top, right, bottom), outline=axis_color, width=1)

    tick_length = 9
    lon_ticks = (-180, -120, -60, 0, 60, 120, 180)
    lat_ticks = (-60, -40, -20, 0, 20, 40, 60)

    for lon in lon_ticks:
        x, _ = to_pixels([lon], [frame.ymin], frame)
        x_pos = int(x[0])
        draw.line((x_pos, bottom, x_pos, bottom - tick_length), fill=axis_color, width=1)
        draw.line((x_pos, top, x_pos, top + tick_length), fill=axis_color, width=1)
        label = degree_label(lon, is_lon=True)
        label_width, label_height = text_size(draw, label, font)
        draw.text(
            (x_pos - (label_width / 2), bottom + 9),
            label,
            fill=label_color,
            font=font,
        )

    for lat in lat_ticks:
        _, y = to_pixels([frame.xmin], [lat], frame)
        y_pos = int(y[0])
        draw.line((left, y_pos, left + tick_length, y_pos), fill=axis_color, width=1)
        draw.line((right, y_pos, right - tick_length, y_pos), fill=axis_color, width=1)
        label = degree_label(lat, is_lon=False)
        label_width, label_height = text_size(draw, label, font)
        draw.text(
            (left - label_width - 10, y_pos - (label_height / 2)),
            label,
            fill=label_color,
            font=font,
        )

    x_title = "Longitude"
    y_title = "Latitude"
    x_title_width, _ = text_size(draw, x_title, font)
    draw.text(
        (left + ((right - left - x_title_width) / 2), bottom + 34),
        x_title,
        fill=label_color,
        font=font,
    )

    y_title_width, y_title_height = text_size(draw, y_title, font)
    y_label = Image.new(
        "RGBA",
        (y_title_width + 4, y_title_height + 4),
        (255, 255, 255, 0),
    )
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((2, 2), y_title, fill=label_color, font=font)
    rotated = y_label.rotate(90, expand=True)
    draw.bitmap(
        (left - 55, top + ((bottom - top - rotated.height) / 2)),
        rotated,
        fill=label_color,
    )


def draw_graticule(draw: ImageDraw.ImageDraw, frame: ProjectionFrame) -> None:
    grid_color = "#C7C7C7"

    for lat in range(-60, 90, 30):
        lons = np.linspace(-180.0, 180.0, 361)
        lats = np.full_like(lons, float(lat))
        draw_polyline(draw, projected_line(frame, lons=lons, lats=lats), fill=grid_color)

    for lon in range(-150, 180, 30):
        lats = np.linspace(-90.0, 90.0, 361)
        lons = np.full_like(lats, float(lon))
        draw_polyline(draw, projected_line(frame, lons=lons, lats=lats), fill=grid_color)


def draw_map_outline(draw: ImageDraw.ImageDraw, frame: ProjectionFrame) -> None:
    outline_color = "#222222"
    lons = np.linspace(-180.0, 180.0, 721)
    top_lats = np.full_like(lons, 90.0)
    bottom_lats = np.full_like(lons, -90.0)
    draw_polyline(
        draw,
        projected_line(frame, lons=lons, lats=top_lats),
        fill=outline_color,
        width=2,
    )
    draw_polyline(
        draw,
        projected_line(frame, lons=lons, lats=bottom_lats),
        fill=outline_color,
        width=2,
    )

    lats = np.linspace(-90.0, 90.0, 721)
    left_lons = np.full_like(lats, -180.0)
    right_lons = np.full_like(lats, 180.0)
    draw_polyline(
        draw,
        projected_line(frame, lons=left_lons, lats=lats),
        fill=outline_color,
        width=2,
    )
    draw_polyline(
        draw,
        projected_line(frame, lons=right_lons, lats=lats),
        fill=outline_color,
        width=2,
    )


def ensure_coastline_geojson(path: Path) -> Path | None:
    if path.exists():
        return path

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Natural Earth coastlines to: {path}", file=sys.stderr)
        urlretrieve(NATURAL_EARTH_COUNTRIES_URL, path)
    except Exception as exc:  # noqa: BLE001 - plotting should still work without outlines.
        print(f"Warning: could not load coastlines ({exc}).", file=sys.stderr)
        return None

    return path


def load_coastline_geometries(path: Path) -> list[dict]:
    coastline_path = ensure_coastline_geojson(path)
    if coastline_path is None:
        return []

    try:
        with coastline_path.open("r", encoding="utf-8") as file:
            geojson = json.load(file)
    except Exception as exc:  # noqa: BLE001 - plotting should still work without outlines.
        print(f"Warning: could not read coastlines ({exc}).", file=sys.stderr)
        return []

    geometries: list[dict] = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        if properties.get("ADMIN") == "Antarctica":
            continue
        geometry = feature.get("geometry")
        if geometry:
            geometries.append(geometry)
    return geometries


def iter_geometry_rings(geometry: dict) -> Iterable[list]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        for ring in coordinates:
            yield ring
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            for ring in polygon:
                yield ring


def split_antimeridian(ring: list) -> Iterable[list[tuple[float, float]]]:
    segment: list[tuple[float, float]] = []
    previous_lon: float | None = None

    for coordinate in ring:
        lon = float(coordinate[0])
        lat = float(coordinate[1])
        if previous_lon is not None and abs(lon - previous_lon) > 180.0:
            if len(segment) >= 2:
                yield segment
            segment = []
        segment.append((lon, lat))
        previous_lon = lon

    if len(segment) >= 2:
        yield segment


def draw_coastlines(
    draw: ImageDraw.ImageDraw,
    frame: ProjectionFrame,
    geometries: list[dict],
) -> None:
    if not geometries:
        return

    line_color = "#111111"
    for geometry in geometries:
        for ring in iter_geometry_rings(geometry):
            for segment in split_antimeridian(ring):
                lons = np.fromiter((point[0] for point in segment), dtype=np.float64)
                lats = np.fromiter((point[1] for point in segment), dtype=np.float64)
                draw_polyline(
                    draw,
                    projected_line(frame, lons=lons, lats=lats),
                    fill=line_color,
                    width=1,
                )


def draw_points(
    draw: ImageDraw.ImageDraw,
    lats: np.ndarray,
    lons: np.ndarray,
    labels: list[str],
    counts: Counter[str],
    frame: ProjectionFrame,
    point_size: int,
) -> None:
    px, py = to_pixels(lons, lats, frame)
    points_by_label: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for x, y, label in zip(px.tolist(), py.tolist(), labels):
        points_by_label[label].append((x, y))

    draw_order = []
    for label in IGBP_ORDER:
        if label in BACKGROUND_LABELS and label in points_by_label:
            draw_order.append(label)
    draw_order.extend(
        label
        for label in category_order(counts)
        if label not in BACKGROUND_LABELS | {NULL_LABEL} and label in points_by_label
    )
    if NULL_LABEL in points_by_label:
        draw_order.append(NULL_LABEL)

    fallback_index = 0
    half_size = point_size // 2
    for label in draw_order:
        color = color_for_label(label, fallback_index)
        fallback_index += 1
        label_points = points_by_label[label]
        if point_size <= 1:
            draw.point(label_points, fill=color)
            continue

        for x, y in label_points:
            draw.rectangle(
                (
                    x - half_size,
                    y - half_size,
                    x + point_size - half_size - 1,
                    y + point_size - half_size - 1,
                ),
                fill=color,
            )


def draw_legend(
    draw: ImageDraw.ImageDraw,
    counts: Counter[str],
    *,
    frame: ProjectionFrame,
    font: ImageFont.ImageFont,
) -> None:
    labels = legend_order(counts)
    if not labels:
        return

    row_height = 17
    swatch_size = 14
    legend_height = len(labels) * row_height
    left = int(round(frame.used_left + 10))
    top = int(
        round(
            min(
                frame.used_top + (frame.used_height * 0.42),
                frame.used_top + frame.used_height - legend_height - 8,
            )
        )
    )
    top = max(int(round(frame.used_top + 8)), top)

    y = top
    fallback_index = 0
    for label in labels:
        color = color_for_label(label, fallback_index)
        fallback_index += 1
        draw.rectangle(
            (left, y + 2, left + swatch_size, y + swatch_size + 2),
            fill=color,
            outline="#333333",
        )
        draw.text(
            (left + 21, y),
            IGBP_FULL_NAMES.get(label, label),
            fill="#111111",
            font=font,
            stroke_width=1,
            stroke_fill="#FFFFFF",
        )
        y += row_height


def render_map(
    lats: np.ndarray,
    lons: np.ndarray,
    labels: list[str],
    counts: Counter[str],
    coastline_geometries: list[dict],
    *,
    projection: str,
    width: int,
    height: int,
    point_size: int,
    legend: bool,
) -> Image.Image:
    if width < 500 or height < 320:
        raise RuntimeError("--width and --height are too small for the map layout.")

    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    axis_font = load_font(18, serif=True)
    legend_font = load_font(12, serif=True)

    area = PlotArea(
        left=72.0,
        top=20.0,
        right=float(width - 25),
        bottom=float(height - 70),
    )
    if area.width < 200 or area.height < 120:
        raise RuntimeError("Image is too small for the requested legend/layout.")

    frame = make_projection_frame(projection, area)

    draw_points(draw, lats, lons, labels, counts, frame, max(1, point_size))
    draw_coastlines(draw, frame, coastline_geometries)

    if legend:
        draw_legend(
            draw,
            counts,
            frame=frame,
            font=legend_font,
        )

    if projection == "equirectangular":
        draw_axes(draw, frame, axis_font)
    else:
        draw_map_outline(draw, frame)

    return image


def print_counts(counts: Counter[str], skipped_missing_coords: int) -> None:
    print("IGBP counts:")
    for label in category_order(counts):
        print(f"  {label}: {counts[label]:,}")
    print(f"Total coord_data rows: {sum(counts.values()):,}")
    if skipped_missing_coords:
        print(f"Rows skipped because lat/lon is NULL: {skipped_missing_coords:,}")


def main() -> int:
    args = parse_args()
    db_path = resolve_path(args.db_path)
    output_path = resolve_path(args.output_path)
    coastline_path = resolve_path(args.coastline_path)

    if not db_path.exists():
        raise SystemExit(f"Database does not exist: {db_path}")

    with connect_readonly(db_path) as conn:
        lats, lons, labels, counts, skipped_missing_coords = load_coordinates(
            conn,
            args.table,
        )

    coastline_geometries = []
    if not args.no_coastlines:
        coastline_geometries = load_coastline_geometries(coastline_path)

    image = render_map(
        lats,
        lons,
        labels,
        counts,
        coastline_geometries,
        projection=args.projection,
        width=args.width,
        height=args.height,
        point_size=args.point_size,
        legend=not args.no_legend,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print_counts(counts, skipped_missing_coords)
    print(f"Map written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
