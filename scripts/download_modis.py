#!/usr/bin/env python3
"""Download global MODIS rasters from Google Earth Engine.

This script is based on `MODIS_global_pull.ipynb` and preserves the
same output naming convention:

- `YYYYMMDD1200A4.tiff` for `MODIS/061/MCD43A4`
- `YYYYMMDD1200A2.tiff` for `MODIS/061/MCD43A2`
- `YYYYMMDD1200C1.tiff` for `MODIS/061/MCD12C1`

Outputs default to `experiments/data/raw_modis`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, timedelta
import os
from pathlib import Path
import tempfile
import time

MISSING_DEPENDENCIES: list[str] = []

try:
    import ee
except ModuleNotFoundError:
    MISSING_DEPENDENCIES.append("earthengine-api")
    ee = None

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
except ModuleNotFoundError:
    MISSING_DEPENDENCIES.append("rasterio")
    rasterio = None
    MemoryFile = None
    merge = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    MISSING_DEPENDENCIES.append("tqdm")
    tqdm = None

HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "data" / "raw_modis"
DEFAULT_PROJECT = "modis-488716"
DEFAULT_START_DATE = date(2016, 12, 31)
DEFAULT_END_DATE = date(2018, 1, 1)
DEFAULT_TARGET_SCALE = 1 / 32
DEFAULT_TILE_SIZE_DEG = 30
DEFAULT_MAX_WORKERS = 64


@dataclass(frozen=True)
class ProductSpec:
    suffix: str
    collection_id: str
    bands: tuple[str, ...]
    description: str
    cadence: str


PRODUCTS: dict[str, ProductSpec] = {
    "A4": ProductSpec(
        suffix="A4",
        collection_id="MODIS/061/MCD43A4",
        bands=(
            "Nadir_Reflectance_Band1",
            "Nadir_Reflectance_Band2",
            "Nadir_Reflectance_Band3",
            "Nadir_Reflectance_Band4",
            "Nadir_Reflectance_Band5",
            "Nadir_Reflectance_Band6",
            "Nadir_Reflectance_Band7",
        ),
        description="Daily NBAR reflectance",
        cadence="daily",
    ),
    "A2": ProductSpec(
        suffix="A2",
        collection_id="MODIS/061/MCD43A2",
        bands=(
            "Snow_BRDF_Albedo",
            "BRDF_Albedo_LandWaterType",
        ),
        description="Daily QA and snow flags",
        cadence="daily",
    ),
    "C1": ProductSpec(
        suffix="C1",
        collection_id="MODIS/061/MCD12C1",
        bands=("Majority_Land_Cover_Type_1",),
        description="Annual land cover",
        cadence="annual",
    ),
}


def ensure_dependencies() -> None:
    if not MISSING_DEPENDENCIES:
        return

    missing = ", ".join(sorted(MISSING_DEPENDENCIES))
    raise SystemExit(
        "Missing Python dependencies: "
        f"{missing}. Install them with `pip install earthengine-api rasterio`."
    )


def authenticate_earth_engine() -> None:
    ee.Authenticate(auth_mode="notebook")


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download global MODIS rasters for an inclusive date range. "
            "Annual C1 land-cover files are downloaded once per year touched "
            "by the requested range. Defaults to 2016-12-31 through 2018-01-01."
        )
    )
    parser.add_argument(
        "start_date",
        nargs="?",
        type=parse_iso_date,
        default=DEFAULT_START_DATE,
        help=(
            "Inclusive start date in YYYY-MM-DD format. "
            f"Default: {DEFAULT_START_DATE.isoformat()}."
        ),
    )
    parser.add_argument(
        "end_date",
        nargs="?",
        type=parse_iso_date,
        default=DEFAULT_END_DATE,
        help=(
            "Inclusive end date in YYYY-MM-DD format. "
            f"Default: {DEFAULT_END_DATE.isoformat()}."
        ),
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("EARTHENGINE_PROJECT", DEFAULT_PROJECT),
        help=(
            "Google Earth Engine project name. "
            "Defaults to EARTHENGINE_PROJECT if set, otherwise modis-488716."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where GeoTIFFs will be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=sorted(PRODUCTS),
        default=["A4", "A2", "C1"],
        help="Products to download. Default: A4 A2 C1.",
    )
    parser.add_argument(
        "--tile-size-deg",
        type=int,
        default=DEFAULT_TILE_SIZE_DEG,
        help="Global download tile size in degrees. Default: 30.",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=DEFAULT_TARGET_SCALE,
        help=(
            "Output resolution in degrees per pixel. "
            "The notebook used 1/32 degrees."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            "Concurrent tile downloads per image. "
            f"Default: {DEFAULT_MAX_WORKERS}."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per tile download before failing. Default: 3.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=5.0,
        help="Base sleep in seconds between retries. Default: 5.",
    )
    parser.add_argument(
        "--authenticate",
        action="store_true",
        help=(
            "Run Earth Engine authentication in notebook mode before "
            "initializing the API."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in the output directory.",
    )
    args = parser.parse_args()

    if args.end_date < args.start_date:
        parser.error("end_date must be on or after start_date.")

    return args


def initialize_earth_engine(
    project: str,
    authenticate: bool,
) -> None:
    if authenticate:
        authenticate_earth_engine()

    try:
        ee.Initialize(project=project, opt_url=HIGH_VOLUME_URL)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Google Earth Engine. Provide a valid project "
            "with --project or EARTHENGINE_PROJECT, and authenticate with "
            "`earthengine authenticate --auth_mode=notebook`, or rerun this "
            "script with --authenticate."
        ) from exc


def parse_asset_date(asset_id: str) -> date:
    asset_date = asset_id.rsplit("/", 1)[-1]
    try:
        return date.fromisoformat(asset_date.replace("_", "-"))
    except ValueError as exc:
        raise RuntimeError(f"Unexpected MODIS asset date format: {asset_date}") from exc


def fetch_asset_ids(
    spec: ProductSpec,
    start_date: date,
    end_date: date,
) -> dict[str, str]:
    collection = ee.ImageCollection(spec.collection_id).select(list(spec.bands))

    if spec.cadence == "annual":
        filter_start = date(start_date.year, 1, 1)
        filter_end = date(end_date.year + 1, 1, 1)
    else:
        filter_start = start_date
        filter_end = end_date + timedelta(days=1)

    asset_ids = (
        collection
        .filterDate(filter_start.isoformat(), filter_end.isoformat())
        .aggregate_array("system:id")
        .getInfo()
    )

    if not asset_ids:
        raise RuntimeError(
            f"No {spec.collection_id} assets found for "
            f"{start_date.isoformat()} to {end_date.isoformat()}."
        )

    selected: dict[str, str] = {}
    for asset_id in sorted(asset_ids):
        asset_date_key = asset_id.rsplit("/", 1)[-1]
        asset_date = parse_asset_date(asset_id)

        if spec.cadence == "annual":
            if start_date.year <= asset_date.year <= end_date.year:
                selected[asset_date_key] = asset_id
        elif start_date <= asset_date <= end_date:
            selected[asset_date_key] = asset_id

    if not selected:
        raise RuntimeError(
            f"No {spec.collection_id} assets remained after applying the "
            f"requested date range {start_date.isoformat()} to {end_date.isoformat()}."
        )

    return selected


def make_grid_tiles(tile_size_deg: int) -> list[tuple[int, int, int, int]]:
    tiles: list[tuple[int, int, int, int]] = []
    for lon in range(-180, 180, tile_size_deg):
        for lat in range(-90, 90, tile_size_deg):
            tiles.append((lon, lat, lon + tile_size_deg, lat + tile_size_deg))
    return tiles


def clamp_max_workers(max_workers: int, total_tiles: int) -> int:
    return max(1, min(max_workers, total_tiles))


def get_tile_to_file(
    asset_id: str,
    bands: tuple[str, ...],
    bounds: tuple[int, int, int, int],
    index: int,
    tmp_dir: str,
    target_scale: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    lon_min, lat_min, lon_max, lat_max = bounds
    width = int((lon_max - lon_min) / target_scale)
    height = int((lat_max - lat_min) / target_scale)

    grid = {
        "crsCode": "EPSG:4326",
        "affineTransform": {
            "scaleX": target_scale,
            "shearX": 0,
            "translateX": lon_min,
            "shearY": 0,
            "scaleY": -target_scale,
            "translateY": lat_max,
        },
        "dimensions": {"width": width, "height": height},
    }

    request = {
        "fileFormat": "GEO_TIFF",
        "bandIds": list(bands),
        "assetId": asset_id,
        "grid": grid,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = ee.data.getPixels(request)
            tile_path = os.path.join(tmp_dir, f"tile_{index:03d}.tif")

            with MemoryFile(response) as memfile:
                with memfile.open() as src:
                    profile = src.profile.copy()
                    with rasterio.open(tile_path, "w", **profile) as dst:
                        dst.write(src.read())

            return tile_path
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Tile download failed for {asset_id} at {bounds} after "
                    f"{max_retries} attempts."
                ) from exc
            time.sleep(retry_sleep * attempt)

    raise AssertionError("unreachable")


def retrieve_tile_files(
    asset_id: str,
    bands: tuple[str, ...],
    tiles: list[tuple[int, int, int, int]],
    tmp_dir: str,
    target_scale: float,
    max_workers: int,
    max_retries: int,
    retry_sleep: float,
    progress_desc: str,
) -> list[str]:
    tile_paths: dict[int, str] = {}
    effective_workers = clamp_max_workers(max_workers, len(tiles))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=effective_workers,
        thread_name_prefix="modis-tile",
    ) as executor:
        future_map = {
            executor.submit(
                get_tile_to_file,
                asset_id,
                bands,
                bounds,
                index,
                tmp_dir,
                target_scale,
                max_retries,
                retry_sleep,
            ): index
            for index, bounds in enumerate(tiles)
        }

        with tqdm(
            total=len(future_map),
            desc=progress_desc,
            unit="tile",
            leave=False,
            dynamic_ncols=True,
        ) as progress:
            for future in concurrent.futures.as_completed(future_map):
                index = future_map[future]
                tile_paths[index] = future.result()
                progress.update(1)

    return [tile_paths[index] for index in sorted(tile_paths)]


def merge_tiles(file_paths: list[str], output_path: Path) -> None:
    if not file_paths:
        raise RuntimeError("No tile files were produced for the merge step.")

    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in file_paths]
        mosaic, transform = merge(sources)
        profile = sources[0].profile.copy()

    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=mosaic.shape[0],
        dtype=mosaic.dtype,
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )

    partial_output = output_path.with_name(
        f"{output_path.stem}.partial{output_path.suffix}"
    )

    try:
        with rasterio.open(partial_output, "w", **profile) as dst:
            dst.write(mosaic)
        partial_output.replace(output_path)
    finally:
        if partial_output.exists():
            partial_output.unlink()


def output_filename(asset_date: str, suffix: str) -> str:
    return f"{asset_date.replace('_', '')}1200{suffix}.tiff"


def download_asset(
    spec: ProductSpec,
    asset_date: str,
    asset_id: str,
    output_dir: Path,
    tiles: list[tuple[int, int, int, int]],
    target_scale: float,
    max_workers: int,
    max_retries: int,
    retry_sleep: float,
    overwrite: bool,
) -> None:
    output_path = output_dir / output_filename(asset_date, spec.suffix)
    effective_workers = clamp_max_workers(max_workers, len(tiles))

    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path.name}")
        return

    with tempfile.TemporaryDirectory(prefix=f"{spec.suffix.lower()}_{asset_date}_") as tmp_dir:
        print(
            f"[download] {output_path.name} with {effective_workers} workers "
            f"across {len(tiles)} tiles"
        )
        file_paths = retrieve_tile_files(
            asset_id=asset_id,
            bands=spec.bands,
            tiles=tiles,
            tmp_dir=tmp_dir,
            target_scale=target_scale,
            max_workers=effective_workers,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
            progress_desc=f"{spec.suffix} {asset_date}",
        )
        merge_tiles(file_paths, output_path)

    print(f"[done] {output_path.name}")


def main() -> int:
    args = parse_args()
    ensure_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    initialize_earth_engine(
        project=args.project,
        authenticate=args.authenticate,
    )

    tiles = make_grid_tiles(args.tile_size_deg)
    effective_workers = clamp_max_workers(args.max_workers, len(tiles))
    print(
        f"Using {len(tiles)} global tiles per asset with up to "
        f"{effective_workers} workers for "
        f"{args.start_date.isoformat()} to {args.end_date.isoformat()} "
        f"and writing into {args.output_dir}"
    )

    for product_name in args.products:
        spec = PRODUCTS[product_name]
        asset_ids = fetch_asset_ids(spec, args.start_date, args.end_date)
        asset_dates = sorted(asset_ids)
        print(
            f"{product_name}: found {len(asset_dates)} assets "
            f"({spec.description})"
        )

        for asset_date in tqdm(
            asset_dates,
            desc=f"{product_name} assets",
            unit="asset",
            dynamic_ncols=True,
        ):
            download_asset(
                spec=spec,
                asset_date=asset_date,
                asset_id=asset_ids[asset_date],
                output_dir=args.output_dir,
                tiles=tiles,
                target_scale=args.target_scale,
                max_workers=effective_workers,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                overwrite=args.overwrite,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
