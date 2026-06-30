import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RUN_PATH = (
    "experiments/runs/"
    "final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0"
)
DEFAULT_PREDICTION_TARGETS = "pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE"
DEFAULT_CHUNK_ROWS = 2_000_000
DEFAULT_WRITE_TIME_CHUNK = 186
DEFAULT_MAX_MEMORY_GB = 470.0
INTERNAL_ORDER_COLUMN = "__sample_order"
BASE_OUTPUT_COLUMNS = ("lat", "lon", "igbp", "timestamp")
REQUIRED_COORDINATE_COLUMNS = ("timestamp", "lat", "lon", "igbp")
DIRECT_DUPLICATE_POLICIES = ("error", "first", "last")
ERA5_TIME_DIM = "valid_time"
ERA5_SPATIAL_DIMS = ("latitude", "longitude")
ERA5_CUBE_DIMS = (ERA5_TIME_DIM, *ERA5_SPATIAL_DIMS)
ERA5_COORDINATES_ATTR = "number valid_time latitude longitude expver"
COORDINATE_DECIMALS = 6
REGULAR_GRID_MIN_COVERAGE = 0.9
ERA5_CHUNK_TARGETS = (186, 31, 360)
ERA5_GLOBAL_ATTRS = {
    "GRIB_centre": "ecmf",
    "GRIB_centreDescription": "European Centre for Medium-Range Weather Forecasts",
    "GRIB_subCentre": 0,
    "Conventions": "CF-1.7",
    "institution": "European Centre for Medium-Range Weather Forecasts",
}
PREDICTION_TARGET_METADATA = {
    "NEE": ("Predicted net ecosystem exchange", "umol CO2 m-2 s-1"),
    "GPP_DT": ("Predicted daytime gross primary productivity", "umol CO2 m-2 s-1"),
    "GPP_NT": ("Predicted nighttime gross primary productivity", "umol CO2 m-2 s-1"),
    "RECO_DT": ("Predicted daytime ecosystem respiration", "umol CO2 m-2 s-1"),
    "RECO_NT": ("Predicted nighttime ecosystem respiration", "umol CO2 m-2 s-1"),
    "FCH4": ("Predicted methane flux", "nmol CH4 m-2 s-1"),
    "LE": ("Predicted latent heat flux", "W m-2"),
}


def parse_prediction_targets(values: list[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None

    prediction_targets = []
    for value in values:
        for target in value.replace(",", " ").split():
            if not target.startswith("pred_"):
                target = f"pred_{target}"
            prediction_targets.append(target)

    prediction_targets = list(dict.fromkeys(prediction_targets))
    if not prediction_targets:
        raise ValueError("--prediction-targets requires at least one target when provided.")
    return tuple(prediction_targets)


def resolve_output_columns(
    input_columns: list[str],
    prediction_targets: tuple[str, ...] | None,
) -> list[str]:
    if prediction_targets is None:
        return [column for column in input_columns if column != INTERNAL_ORDER_COLUMN]

    output_columns = list(BASE_OUTPUT_COLUMNS) + list(prediction_targets)
    missing_columns = [column for column in output_columns if column not in input_columns]
    if missing_columns:
        available_predictions = ", ".join(
            column for column in input_columns if column.startswith("pred_")
        )
        raise ValueError(
            "Requested output column(s) missing from shard header: "
            f"{', '.join(missing_columns)}. "
            f"Available prediction columns: {available_predictions or '<none>'}"
        )
    return output_columns


def coordinate_key(value) -> float:
    return round(float(value), COORDINATE_DECIMALS)


def regularize_coordinate_axis(values, descending: bool, np):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values

    ascending_values = values[::-1] if descending else values
    diffs = np.diff(ascending_values)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return values

    rounded_diffs = np.round(positive_diffs, COORDINATE_DECIMALS)
    unique_diffs, counts = np.unique(rounded_diffs, return_counts=True)
    step = float(unique_diffs[np.argmax(counts)])
    if step <= 0 or not np.isfinite(step):
        return values

    span = float(ascending_values[-1] - ascending_values[0])
    expected_count = int(round(span / step)) + 1
    if expected_count <= values.size:
        return values

    coverage = values.size / expected_count
    if coverage < REGULAR_GRID_MIN_COVERAGE:
        return values

    regular_axis = np.round(
        ascending_values[0] + np.arange(expected_count, dtype=np.float64) * step,
        COORDINATE_DECIMALS,
    )
    if descending:
        regular_axis = regular_axis[::-1]
    return regular_axis


def coordinate_increment(values, np) -> float:
    if len(values) < 2:
        return float("nan")
    diffs = np.diff(np.asarray(values, dtype=np.float64))
    return float(abs(np.nanmedian(diffs)))


def era5_cube_chunks(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(
        max(1, min(size, target))
        for size, target in zip(shape, ERA5_CHUNK_TARGETS)
    )


def prediction_variable_attrs(column: str, latitudes, longitudes, np) -> dict[str, object]:
    target = column.removeprefix("pred_").removeprefix("gt_")
    long_name, units = PREDICTION_TARGET_METADATA.get(
        target,
        (column.replace("_", " "), "unknown"),
    )
    if column.startswith("gt_"):
        long_name = f"Ground truth {long_name.removeprefix('Predicted ').lower()}"

    return {
        "long_name": long_name,
        "units": units,
        "standard_name": "unknown",
        "coordinates": ERA5_COORDINATES_ATTR,
        "GRIB_dataType": "fc",
        "GRIB_numberOfPoints": int(len(latitudes) * len(longitudes)),
        "GRIB_stepType": "instant",
        "GRIB_stepUnits": 1,
        "GRIB_gridType": "regular_ll",
        "GRIB_typeOfLevel": "surface",
        "GRIB_uvRelativeToGrid": 0,
        "GRIB_NV": 0,
        "GRIB_cfName": "unknown",
        "GRIB_cfVarName": column,
        "GRIB_shortName": column,
        "GRIB_gridDefinitionDescription": "Latitude/Longitude Grid",
        "GRIB_iDirectionIncrementInDegrees": coordinate_increment(longitudes, np),
        "GRIB_iScansNegatively": 0,
        "GRIB_jDirectionIncrementInDegrees": coordinate_increment(latitudes, np),
        "GRIB_jPointsAreConsecutive": 0,
        "GRIB_jScansPositively": 0,
        "GRIB_latitudeOfFirstGridPointInDegrees": float(latitudes[0]) if len(latitudes) else np.nan,
        "GRIB_latitudeOfLastGridPointInDegrees": float(latitudes[-1]) if len(latitudes) else np.nan,
        "GRIB_longitudeOfFirstGridPointInDegrees": float(longitudes[0]) if len(longitudes) else np.nan,
        "GRIB_longitudeOfLastGridPointInDegrees": float(longitudes[-1]) if len(longitudes) else np.nan,
        "GRIB_Nx": int(len(longitudes)),
        "GRIB_Ny": int(len(latitudes)),
        "GRIB_missingValue": float(np.finfo(np.float32).max),
        "GRIB_name": long_name,
        "GRIB_totalNumber": 0,
        "GRIB_units": units,
        "GRIB_surface": 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_year_nc.sh",
        description=(
            "Build one calendar-year EcoPerceiver ERA5 NetCDF directly from "
            "quarterly multi-GPU CSV shard directories."
        ),
    )
    parser.add_argument("year", nargs="?", help="Four-digit year to assemble.")
    parser.add_argument("--year", dest="year_option", help="Four-digit year to assemble.")
    parser.add_argument(
        "--run-path",
        default=os.environ.get("RUN_PATH", DEFAULT_RUN_PATH),
        help="EcoPerceiver run directory. Default: RUN_PATH env or the repo default run.",
    )
    parser.add_argument(
        "--input-dir",
        default=os.environ.get("INPUT_DIR"),
        help="Directory containing quarter shard directories. Default: <run-path>/eval.",
    )
    parser.add_argument(
        "--output-path",
        default=os.environ.get("OUTPUT_PATH"),
        help="Yearly NetCDF output path. Default: <input-dir>/era5_predictions_<YEAR>.nc.",
    )
    parser.add_argument(
        "--shard-dirs",
        nargs=4,
        metavar="PATH",
        help=(
            "Explicit Q1 Q2 Q3 Q4 shard directories. If omitted, directories are "
            "inferred from .era5_predictions_<START>_to_<END>_multi_gpu_shards."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=int(os.environ.get("NUM_SHARDS", "4")),
        help="Expected rank_*.csv files per shard directory. Default: NUM_SHARDS env or 4.",
    )
    parser.add_argument(
        "--prediction-targets",
        nargs="+",
        default=os.environ.get("PREDICTION_TARGETS", DEFAULT_PREDICTION_TARGETS).split(),
        help=(
            "Prediction columns to include, with or without pred_ prefix. "
            f"Default: {DEFAULT_PREDICTION_TARGETS}."
        ),
    )
    parser.add_argument(
        "--netcdf-duplicate-policy",
        choices=DIRECT_DUPLICATE_POLICIES,
        default=os.environ.get("NETCDF_DUPLICATE_POLICY", "last"),
        help=(
            "How duplicate (timestamp, lat, lon) rows are handled while filling "
            "the yearly cube. Default: last."
        ),
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=int(os.environ.get("BUILD_YEAR_CHUNK_ROWS", str(DEFAULT_CHUNK_ROWS))),
        help=f"Rows per pandas CSV chunk. Default: {DEFAULT_CHUNK_ROWS}.",
    )
    parser.add_argument(
        "--write-time-chunk",
        type=int,
        default=int(os.environ.get("BUILD_YEAR_WRITE_TIME_CHUNK", str(DEFAULT_WRITE_TIME_CHUNK))),
        help=f"Time steps per NetCDF write slice. Default: {DEFAULT_WRITE_TIME_CHUNK}.",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=float(os.environ.get("BUILD_YEAR_MAX_MEMORY_GB", str(DEFAULT_MAX_MEMORY_GB))),
        help=f"Refuse estimated array allocations above this many GiB. Default: {DEFAULT_MAX_MEMORY_GB}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print inferred paths, selected columns, and memory estimate without writing.",
    )
    parser.add_argument(
        "--validate-shard-order",
        action="store_true",
        help=(
            "Validate that __sample_order is nondecreasing within each quarter's "
            "rank files. This is optional because direct NetCDF filling tracks "
            "__sample_order per output cell."
        ),
    )
    parser.add_argument(
        "--no-validate-shard-order",
        action="store_false",
        dest="validate_shard_order",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    year = args.year_option or args.year
    if year is None:
        parser.error("YEAR is required as a positional argument or --year.")
    if not year.isdigit() or len(year) != 4:
        parser.error(f"YEAR must be a four-digit year, got: {year}")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive.")
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive.")
    if args.write_time_chunk <= 0:
        parser.error("--write-time-chunk must be positive.")
    if args.max_memory_gb <= 0:
        parser.error("--max-memory-gb must be positive.")

    args.year = year
    args.prediction_targets = parse_prediction_targets(args.prediction_targets)
    return args


def quarter_date_tags(year: str) -> list[str]:
    ranges = (
        (f"{year}0101", f"{year}0331"),
        (f"{year}0401", f"{year}0630"),
        (f"{year}0701", f"{year}0930"),
        (f"{year}1001", f"{year}1231"),
    )
    return [f"{start}_to_{end}" for start, end in ranges]


def resolve_shard_dirs(args: argparse.Namespace) -> list[Path]:
    run_path = Path(args.run_path).expanduser()
    input_dir = Path(args.input_dir).expanduser() if args.input_dir else run_path / "eval"
    if args.shard_dirs:
        shard_dirs = [Path(path).expanduser() for path in args.shard_dirs]
    else:
        shard_dirs = [
            input_dir / f".era5_predictions_{date_tag}_multi_gpu_shards"
            for date_tag in quarter_date_tags(args.year)
        ]

    missing = [path for path in shard_dirs if not path.is_dir()]
    if missing:
        missing_list = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing shard directory input(s):\n  {missing_list}")
    return shard_dirs


def resolve_output_path(args: argparse.Namespace) -> Path:
    run_path = Path(args.run_path).expanduser()
    input_dir = Path(args.input_dir).expanduser() if args.input_dir else run_path / "eval"
    return (
        Path(args.output_path).expanduser()
        if args.output_path
        else input_dir / f"era5_predictions_{args.year}.nc"
    )


def resolve_rank_paths(shard_dir: Path, num_shards: int) -> list[Path]:
    shard_paths = [shard_dir / f"rank_{rank:05d}.csv" for rank in range(num_shards)]
    missing = [path for path in shard_paths if not path.is_file()]
    if missing:
        missing_list = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing shard file(s) in {shard_dir}:\n  {missing_list}")
    return shard_paths


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8", buffering=1024 * 1024) as handle:
        header = next(csv.reader(handle), None)
    if header is None:
        raise RuntimeError(f"Shard file is empty: {path}")
    return header


def resolve_headers(
    shard_groups: list[tuple[Path, list[Path]]],
    prediction_targets: tuple[str, ...] | None,
) -> tuple[list[str], list[str], list[str]]:
    expected_header: list[str] | None = None
    for _, shard_paths in shard_groups:
        for path in shard_paths:
            header = read_csv_header(path)
            if expected_header is None:
                expected_header = header
            elif header != expected_header:
                raise RuntimeError(f"CSV header mismatch in shard {path}")

    if expected_header is None:
        raise RuntimeError("No shard headers were available.")

    output_columns = resolve_output_columns(expected_header, prediction_targets)
    missing_required = [column for column in REQUIRED_COORDINATE_COLUMNS if column not in output_columns]
    if missing_required:
        raise RuntimeError(
            "Direct yearly NetCDF output requires coordinate column(s): "
            f"{', '.join(missing_required)}"
        )

    prediction_columns = [
        column for column in output_columns if column not in BASE_OUTPUT_COLUMNS
    ]
    if not prediction_columns:
        raise RuntimeError("No prediction columns selected for NetCDF output.")
    return expected_header, output_columns, prediction_columns


def iter_csv_chunks(path: Path, use_columns: list[str], chunk_rows: int, pandas, dtypes=None):
    yield from pandas.read_csv(
        path,
        usecols=use_columns,
        chunksize=chunk_rows,
        dtype=dtypes,
    )


def scan_coordinate_axes(
    shard_groups: list[tuple[Path, list[Path]]],
    chunk_rows: int,
) -> tuple[object, object]:
    import numpy as np
    import pandas as pd

    lat_values: set[float] = set()
    lon_values: set[float] = set()
    dtypes = {"lat": "float64", "lon": "float64"}
    for shard_dir, shard_paths in shard_groups:
        print(f"Scanning coordinate axes in {shard_dir}")
        for path in shard_paths:
            for chunk in iter_csv_chunks(path, ["lat", "lon"], chunk_rows, pd, dtypes):
                lat_values.update(float(value) for value in chunk["lat"].dropna().unique())
                lon_values.update(float(value) for value in chunk["lon"].dropna().unique())

    if not lat_values or not lon_values:
        raise RuntimeError("No latitude/longitude coordinates found in shard inputs.")

    latitudes = regularize_coordinate_axis(
        np.sort(np.asarray(list(lat_values), dtype=np.float64))[::-1],
        descending=True,
        np=np,
    )
    longitudes = regularize_coordinate_axis(
        np.sort(np.asarray(list(lon_values), dtype=np.float64)),
        descending=False,
        np=np,
    )
    return latitudes, longitudes


def yearly_timestamp_axis(year: str):
    import pandas as pd

    start = f"{year}-01-01 00:00:00"
    end = f"{year}-12-31 23:00:00"
    valid_times = pd.date_range(start, end, freq="h")
    timestamp_keys = valid_times.strftime("%Y%m%d%H%M%S").astype("int64").to_numpy()
    return valid_times.to_numpy(dtype="datetime64[ns]"), timestamp_keys


def timestamp_seconds(valid_times, numpy) -> object:
    seconds = valid_times.astype("datetime64[s]").astype("int64")
    return numpy.asarray(seconds, dtype=numpy.int64)


def gibibytes(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0**3)


def estimate_array_bytes(
    num_times: int,
    num_latitudes: int,
    num_longitudes: int,
    num_predictions: int,
    duplicate_policy: str,
    tracks_sample_order: bool,
) -> int:
    cube_cells = num_times * num_latitudes * num_longitudes
    total = cube_cells * num_predictions * 4
    if duplicate_policy == "error":
        total += cube_cells
    if tracks_sample_order:
        total += cube_cells * 8
    return total


def lat_lon_lookup(values) -> dict[float, int]:
    return {coordinate_key(value): index for index, value in enumerate(values)}


def map_series(series, lookup: dict, name: str):
    mapped = series.map(lookup)
    if mapped.isna().any():
        examples = series[mapped.isna()].head(5).tolist()
        raise RuntimeError(f"Cannot map {name} value(s) onto the NetCDF grid. Examples: {examples}")
    return mapped.to_numpy(dtype="int64")


def update_igbp_array(igbp_array, lat_indices, lon_indices, values, numpy, pandas) -> int:
    frame = pandas.DataFrame(
        {
            "lat_idx": lat_indices,
            "lon_idx": lon_indices,
            "igbp": values.fillna("").astype(str).str.slice(0, 3),
        }
    ).drop_duplicates(["lat_idx", "lon_idx"], keep="last")
    if frame.empty:
        return 0

    row_lat = frame["lat_idx"].to_numpy(dtype=numpy.int64)
    row_lon = frame["lon_idx"].to_numpy(dtype=numpy.int64)
    new_values = frame["igbp"].to_numpy(dtype="<U3")
    existing = igbp_array[row_lat, row_lon]
    conflicts = int(numpy.count_nonzero((existing != "") & (existing != new_values)))
    igbp_array[row_lat, row_lon] = new_values
    return conflicts


def drop_duplicate_indices_for_policy(time_indices, lat_indices, lon_indices, policy, pandas):
    frame = pandas.DataFrame(
        {"time_idx": time_indices, "lat_idx": lat_indices, "lon_idx": lon_indices}
    )
    duplicated = frame.duplicated(["time_idx", "lat_idx", "lon_idx"], keep=False)
    if policy == "error":
        if duplicated.any():
            examples = frame.loc[duplicated].head(5).to_dict("records")
            raise RuntimeError(
                "Duplicate row(s) found for (timestamp, lat, lon) within one CSV chunk. "
                f"Examples: {examples}"
            )
        return slice(None)
    keep = ~frame.duplicated(["time_idx", "lat_idx", "lon_idx"], keep=policy)
    return keep.to_numpy(dtype=bool)


def best_order_positions(time_indices, lat_indices, lon_indices, sample_orders, policy, pandas):
    frame = pandas.DataFrame(
        {
            "time_idx": time_indices,
            "lat_idx": lat_indices,
            "lon_idx": lon_indices,
            "sample_order": sample_orders,
        }
    )
    if not frame.duplicated(["time_idx", "lat_idx", "lon_idx"]).any():
        return slice(None)
    grouped = frame.groupby(["time_idx", "lat_idx", "lon_idx"], sort=False)["sample_order"]
    if policy == "last":
        return grouped.idxmax().to_numpy(dtype="int64")
    if policy == "first":
        return grouped.idxmin().to_numpy(dtype="int64")
    raise ValueError(f"Unsupported ordered duplicate policy: {policy}")


def validate_order(chunk, previous_order: int | None, shard_dir: Path, path: Path) -> int | None:
    if INTERNAL_ORDER_COLUMN not in chunk.columns:
        return previous_order
    orders = chunk[INTERNAL_ORDER_COLUMN].to_numpy(dtype="int64")
    if orders.size == 0:
        return previous_order
    if previous_order is not None and int(orders[0]) < previous_order:
        raise RuntimeError(
            f"{INTERNAL_ORDER_COLUMN} decreases between shard chunks in {shard_dir}: {path}"
        )
    if orders.size > 1 and (orders[1:] < orders[:-1]).any():
        raise RuntimeError(f"{INTERNAL_ORDER_COLUMN} decreases within shard file: {path}")
    return int(orders[-1])


def fill_prediction_arrays(
    shard_groups: list[tuple[Path, list[Path]]],
    header: list[str],
    prediction_columns: list[str],
    latitudes,
    longitudes,
    timestamp_keys,
    duplicate_policy: str,
    chunk_rows: int,
    validate_shard_order: bool,
):
    import numpy as np
    import pandas as pd

    cube_shape = (len(timestamp_keys), len(latitudes), len(longitudes))
    arrays = {
        column: np.full(cube_shape, np.nan, dtype=np.float32)
        for column in prediction_columns
    }
    igbp_array = np.full(cube_shape[1:], "", dtype="<U3")
    tracks_sample_order = (
        duplicate_policy in {"first", "last"} and INTERNAL_ORDER_COLUMN in header
    )
    sample_order_array = None
    if tracks_sample_order:
        initial_order = (
            np.iinfo(np.int64).max
            if duplicate_policy == "first"
            else np.iinfo(np.int64).min
        )
        sample_order_array = np.full(cube_shape, initial_order, dtype=np.int64)
    filled_mask = (
        np.zeros(cube_shape, dtype=bool)
        if duplicate_policy == "error"
        else None
    )

    time_lookup = {int(value): index for index, value in enumerate(timestamp_keys)}
    lat_lookup = lat_lon_lookup(latitudes)
    lon_lookup = lat_lon_lookup(longitudes)
    use_columns = ["lat", "lon", "igbp", "timestamp", *prediction_columns]
    if INTERNAL_ORDER_COLUMN in header:
        use_columns.insert(0, INTERNAL_ORDER_COLUMN)
    use_columns = list(dict.fromkeys(use_columns))
    dtypes = {
        INTERNAL_ORDER_COLUMN: "int64",
        "lat": "float64",
        "lon": "float64",
        "igbp": "string",
        "timestamp": "int64",
        **{column: "float32" for column in prediction_columns},
    }

    rows_read = 0
    rows_written = 0
    duplicate_rows_skipped = 0
    igbp_conflicts = 0
    for shard_dir, shard_paths in shard_groups:
        print(f"Filling yearly cube from {shard_dir}")
        previous_order: int | None = None
        for path in shard_paths:
            for chunk in iter_csv_chunks(path, use_columns, chunk_rows, pd, dtypes):
                if validate_shard_order:
                    previous_order = validate_order(chunk, previous_order, shard_dir, path)
                rows_read += int(len(chunk))

                timestamps = chunk["timestamp"].astype("int64")
                lat_keys = chunk["lat"].astype("float64").round(COORDINATE_DECIMALS)
                lon_keys = chunk["lon"].astype("float64").round(COORDINATE_DECIMALS)

                time_indices = map_series(timestamps, time_lookup, "timestamp")
                lat_indices = map_series(lat_keys, lat_lookup, "latitude")
                lon_indices = map_series(lon_keys, lon_lookup, "longitude")
                igbp_conflicts += update_igbp_array(
                    igbp_array,
                    lat_indices,
                    lon_indices,
                    chunk["igbp"],
                    np,
                    pd,
                )

                keep = slice(None)
                if tracks_sample_order:
                    sample_orders = chunk[INTERNAL_ORDER_COLUMN].to_numpy(dtype=np.int64)
                    keep = best_order_positions(
                        time_indices,
                        lat_indices,
                        lon_indices,
                        sample_orders,
                        duplicate_policy,
                        pd,
                    )
                    if not isinstance(keep, slice):
                        duplicate_rows_skipped += int(len(time_indices) - len(keep))
                        time_indices = time_indices[keep]
                        lat_indices = lat_indices[keep]
                        lon_indices = lon_indices[keep]
                        sample_orders = sample_orders[keep]

                    indexer = (time_indices, lat_indices, lon_indices)
                    previous_orders = sample_order_array[indexer]
                    if duplicate_policy == "last":
                        better = sample_orders > previous_orders
                    else:
                        better = sample_orders < previous_orders
                    duplicate_rows_skipped += int(len(better) - better.sum())
                    time_indices = time_indices[better]
                    lat_indices = lat_indices[better]
                    lon_indices = lon_indices[better]
                    sample_orders = sample_orders[better]
                    sample_order_array[(time_indices, lat_indices, lon_indices)] = sample_orders
                    keep_for_values = keep
                    better_for_values = better
                elif duplicate_policy == "error":
                    keep = drop_duplicate_indices_for_policy(
                        time_indices,
                        lat_indices,
                        lon_indices,
                        duplicate_policy,
                        pd,
                    )
                    if not isinstance(keep, slice):
                        duplicate_rows_skipped += int(len(keep) - keep.sum())
                        time_indices = time_indices[keep]
                        lat_indices = lat_indices[keep]
                        lon_indices = lon_indices[keep]
                    indexer = (time_indices, lat_indices, lon_indices)
                    if filled_mask[indexer].any():
                        raise RuntimeError(
                            "Duplicate row(s) found for (timestamp, lat, lon) across CSV chunks."
                        )
                    filled_mask[(time_indices, lat_indices, lon_indices)] = True
                    keep_for_values = keep
                    better_for_values = slice(None)
                elif duplicate_policy != "last":
                    raise ValueError(f"Unsupported duplicate policy: {duplicate_policy}")
                else:
                    keep_for_values = slice(None)
                    better_for_values = slice(None)

                indexer = (time_indices, lat_indices, lon_indices)
                for column in prediction_columns:
                    values = chunk[column]
                    if not isinstance(keep_for_values, slice):
                        values = values.iloc[keep_for_values]
                    if not isinstance(better_for_values, slice):
                        values = values.iloc[better_for_values]
                    arrays[column][indexer] = values.to_numpy(dtype=np.float32)
                rows_written += int(len(time_indices))
                if rows_read and rows_read % (chunk_rows * 20) < chunk_rows:
                    print(f"  rows read: {rows_read:,}; rows written: {rows_written:,}")

    if igbp_conflicts:
        print(f"Warning: observed {igbp_conflicts:,} conflicting IGBP cell assignments; kept last value.")
    if duplicate_rows_skipped:
        print(f"Skipped {duplicate_rows_skipped:,} duplicate row(s) with policy {duplicate_policy!r}.")
    return arrays, igbp_array, rows_read, rows_written


def write_attrs(attrs, values: dict[str, object]) -> None:
    for key, value in values.items():
        attrs[key] = value


def create_numeric_variable(file_handle, name, dimensions, dtype, fillvalue=None, chunks=None):
    kwargs = {}
    if chunks is not None:
        kwargs["chunks"] = chunks
    if fillvalue is not None:
        kwargs["fillvalue"] = fillvalue
    if dimensions:
        kwargs.update({"compression": "gzip", "compression_opts": 1, "shuffle": True})
    return file_handle.create_variable(name, dimensions, dtype=dtype, **kwargs)


def write_yearly_netcdf(
    output_path: Path,
    arrays: dict[str, object],
    igbp_array,
    valid_times,
    timestamp_keys,
    latitudes,
    longitudes,
    rows_read: int,
    rows_written: int,
    shard_dirs: list[Path],
    duplicate_policy: str,
    write_time_chunk: int,
) -> None:
    import h5netcdf
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)

    cube_shape = (len(valid_times), len(latitudes), len(longitudes))
    cube_chunks = era5_cube_chunks(cube_shape)
    write_step = max(1, min(int(write_time_chunk), cube_shape[0]))

    try:
        with h5netcdf.File(tmp_path, "w") as dataset:
            dataset.dimensions = {
                ERA5_TIME_DIM: cube_shape[0],
                "latitude": cube_shape[1],
                "longitude": cube_shape[2],
            }
            write_attrs(dataset.attrs, ERA5_GLOBAL_ATTRS)
            dataset.attrs["title"] = "EcoPerceiver predictions on an ERA5 latitude-longitude grid"
            dataset.attrs["source"] = "EcoPerceiver ERA5 torchrun inference shards"
            dataset.attrs["history"] = (
                f"{datetime.now(timezone.utc).isoformat()} EcoPerceiver predictions "
                "converted directly from quarterly shard CSVs to one calendar-year "
                "ERA5-like NetCDF via eval/era5/build_year_nc_from_shards.py"
            )
            dataset.attrs["num_shard_dirs"] = len(shard_dirs)
            dataset.attrs["assembled_input_shard_dirs"] = " ".join(str(path) for path in shard_dirs)
            dataset.attrs["num_input_rows"] = int(rows_read)
            dataset.attrs["num_output_rows"] = int(rows_written)
            dataset.attrs["netcdf_duplicate_policy"] = duplicate_policy
            dataset.attrs["timestamp_source_column"] = "timestamp"
            dataset.attrs["timestamp_source_format"] = "YYYYMMDDHHMMSS"
            dataset.attrs["timestamp_coordinate"] = ERA5_TIME_DIM
            dataset.attrs["timestamp_note"] = (
                "valid_time coordinate values are parsed from the shard timestamp "
                "column. They may be local wall-clock times when the source ERA5 "
                "database was built with local timestamp_policy."
            )

            number_var = dataset.create_variable("number", (), dtype="int64")
            number_var[...] = np.asarray(0, dtype=np.int64)
            number_var.attrs["long_name"] = "ensemble member numerical id"
            number_var.attrs["units"] = "1"
            number_var.attrs["standard_name"] = "realization"

            time_var = dataset.create_variable(ERA5_TIME_DIM, (ERA5_TIME_DIM,), dtype="int64")
            time_var[:] = timestamp_seconds(valid_times, np)
            time_var.attrs["units"] = "seconds since 1970-01-01"
            time_var.attrs["calendar"] = "proleptic_gregorian"
            time_var.attrs["long_name"] = "time"
            time_var.attrs["standard_name"] = "time"

            lat_var = dataset.create_variable(
                "latitude",
                ("latitude",),
                dtype="float64",
                fillvalue=np.nan,
            )
            lat_var[:] = latitudes
            lat_var.attrs["long_name"] = "latitude"
            lat_var.attrs["units"] = "degrees_north"
            lat_var.attrs["standard_name"] = "latitude"
            lat_var.attrs["stored_direction"] = "decreasing"

            lon_var = dataset.create_variable(
                "longitude",
                ("longitude",),
                dtype="float64",
                fillvalue=np.nan,
            )
            lon_var[:] = longitudes
            lon_var.attrs["long_name"] = "longitude"
            lon_var.attrs["units"] = "degrees_east"
            lon_var.attrs["standard_name"] = "longitude"

            expver_var = dataset.create_variable("expver", (ERA5_TIME_DIM,), dtype="S4")
            expver_var[:] = np.full(cube_shape[0], b"0001", dtype="S4")

            igbp_var = dataset.create_variable(
                "igbp",
                ERA5_SPATIAL_DIMS,
                dtype="S3",
                chunks=tuple(max(1, min(size, target)) for size, target in zip(cube_shape[1:], cube_chunks[1:])),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            igbp_var[:] = igbp_array.astype("S3")
            igbp_var.attrs["long_name"] = "IGBP land cover class"
            igbp_var.attrs["coordinates"] = "latitude longitude"

            for name in list(arrays):
                array = arrays.pop(name)
                variable = create_numeric_variable(
                    dataset,
                    name,
                    ERA5_CUBE_DIMS,
                    dtype="float32",
                    fillvalue=np.float32(np.nan),
                    chunks=cube_chunks,
                )
                variable.attrs.update(
                    prediction_variable_attrs(name, latitudes, longitudes, np)
                )
                variable.attrs["coordinates"] = ERA5_COORDINATES_ATTR
                for start in range(0, cube_shape[0], write_step):
                    end = min(cube_shape[0], start + write_step)
                    variable[start:end, :, :] = array[start:end, :, :]
                del array
                print(f"Wrote variable {name}")

        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    shard_dirs = resolve_shard_dirs(args)
    output_path = resolve_output_path(args)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite to replace it: {output_path}")

    shard_groups = [(path, resolve_rank_paths(path, args.num_shards)) for path in shard_dirs]
    header, output_columns, prediction_columns = resolve_headers(
        shard_groups,
        args.prediction_targets,
    )
    valid_times, timestamp_keys = yearly_timestamp_axis(args.year)

    print("ERA5 yearly NetCDF direct shard build")
    print(f"Year: {args.year}")
    print("Shard directories:")
    for path, shard_paths in shard_groups:
        print(f"  {path} ({len(shard_paths)} shard files)")
    print(f"Output: {output_path}")
    print("Writer: h5netcdf")
    print(f"Prediction columns: {', '.join(prediction_columns)}")
    print(f"Duplicate policy: {args.netcdf_duplicate_policy}")
    print(f"CSV chunk rows: {args.chunk_rows:,}")

    latitudes, longitudes = scan_coordinate_axes(shard_groups, args.chunk_rows)
    tracks_sample_order = (
        args.netcdf_duplicate_policy in {"first", "last"}
        and INTERNAL_ORDER_COLUMN in header
    )
    estimated_bytes = estimate_array_bytes(
        len(valid_times),
        len(latitudes),
        len(longitudes),
        len(prediction_columns),
        args.netcdf_duplicate_policy,
        tracks_sample_order,
    )
    print(
        "Output cube shape: "
        f"time={len(valid_times):,}, latitude={len(latitudes):,}, longitude={len(longitudes):,}"
    )
    print(f"Estimated in-memory arrays: {gibibytes(estimated_bytes):.1f} GiB")
    if gibibytes(estimated_bytes) > args.max_memory_gb:
        raise MemoryError(
            f"Estimated array allocation is {gibibytes(estimated_bytes):.1f} GiB, "
            f"above --max-memory-gb={args.max_memory_gb:.1f} GiB."
        )

    if args.dry_run:
        print("Dry run only; no file written.")
        return 0

    arrays, igbp_array, rows_read, rows_written = fill_prediction_arrays(
        shard_groups=shard_groups,
        header=header,
        prediction_columns=prediction_columns,
        latitudes=latitudes,
        longitudes=longitudes,
        timestamp_keys=timestamp_keys,
        duplicate_policy=args.netcdf_duplicate_policy,
        chunk_rows=args.chunk_rows,
        validate_shard_order=args.validate_shard_order,
    )
    print(f"Finished filling arrays from {rows_read:,} input row(s).")
    write_yearly_netcdf(
        output_path=output_path,
        arrays=arrays,
        igbp_array=igbp_array,
        valid_times=valid_times,
        timestamp_keys=timestamp_keys,
        latitudes=latitudes,
        longitudes=longitudes,
        rows_read=rows_read,
        rows_written=rows_written,
        shard_dirs=shard_dirs,
        duplicate_policy=args.netcdf_duplicate_policy,
        write_time_chunk=args.write_time_chunk,
    )
    print(f"Saved yearly NetCDF to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
