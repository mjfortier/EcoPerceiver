import argparse
import csv
import io
import shutil
import subprocess
import os
from datetime import datetime, timezone
from pathlib import Path

INTERNAL_ORDER_COLUMN = "__sample_order"
BASE_OUTPUT_COLUMNS = ("lat", "lon", "igbp", "timestamp")
DEFAULT_SORT_COLUMNS = ("lat", "lon", "timestamp")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Merge ERA5 torchrun CSV shards.")
    parser.add_argument(
        "--shard-dir",
        type=Path,
        required=True,
        help="Directory containing rank_*.csv shard files.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "netcdf"),
        default="csv",
        help="Post-processing output format.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Final output path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Final merged CSV path. Kept as an alias for --output-path.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Expected number of rank shards. If omitted, all rank_*.csv files are merged.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove the shard directory after a successful merge.",
    )
    parser.add_argument(
        "--prediction-targets",
        nargs="+",
        default=None,
        metavar="TARGET",
        help=(
            "Prediction columns to keep in the final output. Accepts comma-separated "
            "or space-separated values, with or without the pred_ prefix. "
            "Default: keep all prediction columns present in the shards."
        ),
    )
    parser.add_argument(
        "--sort-output",
        action="store_true",
        help=(
            "Sort final rows during post-processing. Uses the temporary "
            "__sample_order shard column when present, otherwise falls back to "
            "lat, lon, and timestamp."
        ),
    )
    parser.add_argument(
        "--sort-tmp-dir",
        type=Path,
        default=None,
        help="Temporary directory for external sort files (default: sort chooses its temp location).",
    )
    parser.add_argument(
        "--netcdf-duplicate-policy",
        choices=("error", "first", "last", "mean"),
        default="error",
        help=(
            "How NetCDF cube output handles duplicate (timestamp, lat, lon) rows. "
            "Default: error."
        ),
    )
    return parser.parse_args()


def resolve_shard_paths(shard_dir: Path, num_shards: int | None) -> list[Path]:
    if num_shards is not None:
        if num_shards <= 0:
            raise ValueError("--num-shards must be positive when provided.")
        shard_paths = [shard_dir / f"rank_{rank:05d}.csv" for rank in range(num_shards)]
    else:
        shard_paths = sorted(shard_dir.glob("rank_*.csv"))

    if not shard_paths:
        raise FileNotFoundError(f"No rank_*.csv shards found in {shard_dir}")

    missing = [path for path in shard_paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing expected shard file(s): {missing_list}")

    return shard_paths


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


def merge_csv_shards(
    shard_paths: list[Path],
    output_csv_path: Path,
    prediction_targets: tuple[str, ...] | None,
    sort_output: bool,
    sort_tmp_dir: Path | None,
):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_csv_path.with_suffix(output_csv_path.suffix + ".tmp")
    sorted_rows_tmp = output_csv_path.with_suffix(output_csv_path.suffix + ".rows.sorted.tmp")
    header_written = False
    expected_header = None
    output_columns = None
    output_indices = None

    for shard_path in shard_paths:
        with shard_path.open("r", newline="", encoding="utf-8", buffering=1024 * 1024) as shard_file:
            reader = csv.reader(shard_file)
            header = next(reader, None)
            if header is None:
                continue
            if expected_header is None:
                expected_header = header
                output_columns = resolve_output_columns(header, prediction_targets)
                output_indices = [header.index(column) for column in output_columns]
            elif header != expected_header:
                raise RuntimeError(f"CSV header mismatch in shard {shard_path}")
    if output_columns is None or output_indices is None or expected_header is None:
        raise RuntimeError("No shard headers were available to merge.")

    if sort_output:
        sort_key_args = []
        if INTERNAL_ORDER_COLUMN in expected_header:
            sort_input_columns = [INTERNAL_ORDER_COLUMN] + output_columns
            sort_input_indices = [expected_header.index(column) for column in sort_input_columns]
            sort_key_args.append("-k1,1n")
            drop_sort_key_column = True
        else:
            sort_input_columns = output_columns
            sort_input_indices = output_indices
            for column in DEFAULT_SORT_COLUMNS:
                try:
                    column_index = sort_input_columns.index(column) + 1
                except ValueError as exc:
                    raise RuntimeError(f"Cannot sort output; missing column: {column}") from exc
                sort_key_args.append(f"-k{column_index},{column_index}n")
            drop_sort_key_column = False

        sort_command = ["sort", "-t,", *sort_key_args]
        if sort_tmp_dir is not None:
            sort_tmp_dir.mkdir(parents=True, exist_ok=True)
            sort_command.extend(["-T", str(sort_tmp_dir)])
        sort_env = os.environ.copy()
        sort_env["LC_ALL"] = "C"
        try:
            with sorted_rows_tmp.open("wb") as sorted_rows_file:
                sort_proc = subprocess.Popen(
                    sort_command,
                    stdin=subprocess.PIPE,
                    stdout=sorted_rows_file,
                    env=sort_env,
                )
                assert sort_proc.stdin is not None
                sort_stdin = io.TextIOWrapper(
                    sort_proc.stdin,
                    encoding="utf-8",
                    newline="",
                    write_through=True,
                )
                writer = csv.writer(sort_stdin, lineterminator="\n")
                for shard_path in shard_paths:
                    with shard_path.open("r", newline="", encoding="utf-8", buffering=1024 * 1024) as shard_file:
                        reader = csv.reader(shard_file)
                        next(reader, None)
                        for row in reader:
                            writer.writerow([row[index] for index in sort_input_indices])
                sort_stdin.close()
                returncode = sort_proc.wait()
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, sort_command)

            with output_tmp.open("w", newline="", encoding="utf-8", buffering=1024 * 1024) as out_file:
                writer = csv.writer(out_file)
                writer.writerow(output_columns)
                if drop_sort_key_column:
                    with sorted_rows_tmp.open("r", newline="", encoding="utf-8", buffering=1024 * 1024) as rows_file:
                        reader = csv.reader(rows_file)
                        for row in reader:
                            writer.writerow(row[1:])
                else:
                    with sorted_rows_tmp.open("r", encoding="utf-8", buffering=1024 * 1024) as rows_file:
                        shutil.copyfileobj(rows_file, out_file, length=1024 * 1024)
            output_tmp.replace(output_csv_path)
        finally:
            sorted_rows_tmp.unlink(missing_ok=True)
            output_tmp.unlink(missing_ok=True)
        return

    if prediction_targets is None and INTERNAL_ORDER_COLUMN not in expected_header:
        with output_tmp.open("wb") as out_file:
            for shard_path in shard_paths:
                with shard_path.open("rb") as shard_file:
                    header = shard_file.readline()
                    if not header:
                        continue
                    header_columns = next(
                        csv.reader([header.decode("utf-8").rstrip("\r\n")])
                    )
                    if header_columns != expected_header:
                        raise RuntimeError(f"CSV header mismatch in shard {shard_path}")

                    if not header_written:
                        out_file.write(header)
                        header_written = True
                    shutil.copyfileobj(shard_file, out_file, length=1024 * 1024)

        output_tmp.replace(output_csv_path)
        return

    with output_tmp.open("w", newline="", encoding="utf-8", buffering=1024 * 1024) as out_file:
        writer = csv.writer(out_file)
        writer.writerow(output_columns)
        for shard_path in shard_paths:
            with shard_path.open("r", newline="", encoding="utf-8", buffering=1024 * 1024) as shard_file:
                reader = csv.reader(shard_file)
                header = next(reader, None)
                if header is None:
                    continue
                if header != expected_header:
                    raise RuntimeError(f"CSV header mismatch in shard {shard_path}")

                for row in reader:
                    writer.writerow([row[index] for index in output_indices])

    output_tmp.replace(output_csv_path)


def write_netcdf_from_csv_shards(
    shard_paths: list[Path],
    output_netcdf_path: Path,
    prediction_targets: tuple[str, ...] | None,
    sort_output: bool,
    duplicate_policy: str,
):
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "NetCDF output requires numpy, pandas, and xarray in the active environment."
        ) from exc

    output_netcdf_path.parent.mkdir(parents=True, exist_ok=True)
    input_columns = pd.read_csv(shard_paths[0], nrows=0).columns.tolist()
    output_columns = resolve_output_columns(input_columns, prediction_targets)
    use_columns = None
    if prediction_targets is not None or INTERNAL_ORDER_COLUMN in input_columns:
        use_columns = list(output_columns)
        if sort_output and INTERNAL_ORDER_COLUMN in input_columns:
            use_columns = [INTERNAL_ORDER_COLUMN] + use_columns

    frames = [pd.read_csv(shard_path, usecols=use_columns) for shard_path in shard_paths]
    if not frames:
        raise RuntimeError("No shard rows were available to write NetCDF output.")

    df = pd.concat(frames, ignore_index=True)
    if sort_output:
        if INTERNAL_ORDER_COLUMN in df.columns:
            df[INTERNAL_ORDER_COLUMN] = pd.to_numeric(df[INTERNAL_ORDER_COLUMN], errors="coerce")
            df = df.sort_values(INTERNAL_ORDER_COLUMN, kind="mergesort", ignore_index=True)
        else:
            missing_sort_columns = [column for column in DEFAULT_SORT_COLUMNS if column not in df.columns]
            if missing_sort_columns:
                raise RuntimeError(
                    "Cannot sort NetCDF output; missing column(s): "
                    f"{', '.join(missing_sort_columns)}"
                )
            for column in DEFAULT_SORT_COLUMNS:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            df = df.sort_values(list(DEFAULT_SORT_COLUMNS), kind="mergesort", ignore_index=True)
    if INTERNAL_ORDER_COLUMN in df.columns:
        df = df.drop(columns=[INTERNAL_ORDER_COLUMN])
    df = df.loc[:, output_columns]
    for column in df.columns:
        if column in {"lat", "lon", "timestamp"} or column.startswith(("pred_", "gt_")):
            numeric = pd.to_numeric(df[column], errors="coerce")
            if column == "timestamp" and not numeric.isna().any():
                df[column] = numeric.astype("int64")
            else:
                df[column] = numeric

    dataset = build_era5_cube_dataset(
        df,
        output_columns,
        len(shard_paths),
        duplicate_policy,
        np,
        pd,
        xr,
    )

    output_tmp = output_netcdf_path.with_suffix(output_netcdf_path.suffix + ".tmp")
    dataset.to_netcdf(
        output_tmp,
        engine="h5netcdf",
        encoding=era5_netcdf_encoding(dataset, np),
    )
    output_tmp.replace(output_netcdf_path)


def build_era5_cube_dataset(df, output_columns, num_shards, duplicate_policy, np, pd, xr):
    coordinate_columns = ("timestamp", "lat", "lon")
    missing_coordinate_columns = [
        column for column in coordinate_columns if column not in df.columns
    ]
    if missing_coordinate_columns:
        raise RuntimeError(
            "Cannot write ERA5 NetCDF cube; missing coordinate column(s): "
            f"{', '.join(missing_coordinate_columns)}"
        )

    null_coordinate_columns = [
        column for column in coordinate_columns if df[column].isna().any()
    ]
    if null_coordinate_columns:
        raise RuntimeError(
            "Cannot write ERA5 NetCDF cube; null coordinate value(s) found in: "
            f"{', '.join(null_coordinate_columns)}"
        )

    input_row_count = int(len(df))
    duplicate_row_count = 0
    duplicate_group_count = 0
    duplicate_mask = df.duplicated(list(coordinate_columns), keep=False)
    if duplicate_mask.any():
        duplicate_row_count = int(duplicate_mask.sum())
        duplicate_keys = df.loc[duplicate_mask, list(coordinate_columns)].drop_duplicates()
        duplicate_group_count = int(len(duplicate_keys))
        examples = duplicate_keys.head(5).to_dict("records")
        if duplicate_policy == "error":
            raise RuntimeError(
                "Cannot write ERA5 NetCDF cube; duplicate row(s) found for "
                f"(timestamp, lat, lon): {duplicate_row_count}. Examples: {examples}"
            )

        print(
            "Applying NetCDF duplicate policy "
            f"{duplicate_policy!r}: {duplicate_row_count} row(s) across "
            f"{duplicate_group_count} duplicate coordinate group(s)."
        )
        if duplicate_policy in {"first", "last"}:
            df = df.drop_duplicates(
                list(coordinate_columns),
                keep=duplicate_policy,
                ignore_index=True,
            )
        elif duplicate_policy == "mean":
            aggregations = {}
            for column in output_columns:
                if column in coordinate_columns:
                    continue
                if pd.api.types.is_numeric_dtype(df[column]):
                    aggregations[column] = "mean"
                else:
                    aggregations[column] = "first"
            df = df.groupby(list(coordinate_columns), as_index=False, sort=False).agg(
                aggregations
            )
            df = df.loc[:, output_columns]
        else:
            raise ValueError(f"Unsupported NetCDF duplicate policy: {duplicate_policy}")

    timestamps = np.sort(df["timestamp"].astype("int64").unique())
    latitudes = regularize_coordinate_axis(
        np.sort(df["lat"].astype("float64").unique())[::-1],
        descending=True,
        np=np,
    )
    longitudes = regularize_coordinate_axis(
        np.sort(df["lon"].astype("float64").unique()),
        descending=False,
        np=np,
    )

    time_lookup = {value: index for index, value in enumerate(timestamps)}
    lat_lookup = coordinate_lookup(latitudes)
    lon_lookup = coordinate_lookup(longitudes)

    time_indices = df["timestamp"].map(time_lookup).to_numpy(dtype=np.int64)
    lat_indices = map_coordinate_indices(df["lat"], lat_lookup, np)
    lon_indices = map_coordinate_indices(df["lon"], lon_lookup, np)

    coords = {
        "number": np.asarray(0, dtype=np.int64),
        ERA5_TIME_DIM: (
            ERA5_TIME_DIM,
            timestamp_to_valid_time_coordinate(timestamps, np, pd),
        ),
        "latitude": ("latitude", latitudes),
        "longitude": ("longitude", longitudes),
        "expver": (ERA5_TIME_DIM, np.full(len(timestamps), "0001", dtype="<U4")),
    }
    data_vars = {}
    cube_shape = (len(timestamps), len(latitudes), len(longitudes))

    for column in output_columns:
        if column in BASE_OUTPUT_COLUMNS:
            continue

        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=np.float32)
            array = np.full(cube_shape, np.nan, dtype=np.float32)
        else:
            values = series.fillna("").astype(str).to_numpy(dtype=object)
            array = np.full(cube_shape, "", dtype=object)

        array[time_indices, lat_indices, lon_indices] = values
        data_vars[column] = (
            ERA5_CUBE_DIMS,
            array,
            prediction_variable_attrs(column, latitudes, longitudes, np),
        )

    if "igbp" in output_columns:
        add_igbp_variable(
            data_vars,
            df,
            cube_shape,
            lat_indices,
            lon_indices,
            time_indices,
            lat_lookup,
            lon_lookup,
            np,
        )

    dataset = xr.Dataset(data_vars=data_vars, coords=coords)
    dataset = dataset.transpose(*ERA5_CUBE_DIMS, missing_dims="ignore")
    dataset.attrs.update(ERA5_GLOBAL_ATTRS)
    dataset.attrs["title"] = "EcoPerceiver predictions on an ERA5 latitude-longitude grid"
    dataset.attrs["source"] = "EcoPerceiver ERA5 torchrun inference shards"
    dataset.attrs["history"] = (
        f"{datetime.now(timezone.utc).isoformat()} EcoPerceiver predictions "
        "converted to ERA5-like NetCDF via eval/merge_era5_shards.py"
    )
    dataset.attrs["num_shards"] = num_shards
    dataset.attrs["num_input_rows"] = input_row_count
    dataset.attrs["num_output_rows"] = int(len(df))
    dataset.attrs["netcdf_duplicate_policy"] = duplicate_policy
    dataset.attrs["num_duplicate_coordinate_rows"] = duplicate_row_count
    dataset.attrs["num_duplicate_coordinate_groups"] = duplicate_group_count
    dataset.attrs["timestamp_source_column"] = "timestamp"
    dataset.attrs["timestamp_source_format"] = "YYYYMMDDHHMMSS"
    dataset.attrs["timestamp_coordinate"] = ERA5_TIME_DIM
    dataset.attrs["timestamp_note"] = (
        "valid_time coordinate values are parsed from the shard timestamp "
        "column. They may be local wall-clock times when the source ERA5 "
        "database was built with local timestamp_policy."
    )

    dataset["number"].attrs["long_name"] = "ensemble member numerical id"
    dataset["number"].attrs["units"] = "1"
    dataset["number"].attrs["standard_name"] = "realization"
    dataset[ERA5_TIME_DIM].attrs["long_name"] = "time"
    dataset[ERA5_TIME_DIM].attrs["standard_name"] = "time"
    dataset["latitude"].attrs["long_name"] = "latitude"
    dataset["latitude"].attrs["units"] = "degrees_north"
    dataset["latitude"].attrs["standard_name"] = "latitude"
    dataset["latitude"].attrs["stored_direction"] = "decreasing"
    dataset["longitude"].attrs["long_name"] = "longitude"
    dataset["longitude"].attrs["units"] = "degrees_east"
    dataset["longitude"].attrs["standard_name"] = "longitude"
    return dataset


def coordinate_key(value) -> float:
    return round(float(value), COORDINATE_DECIMALS)


def coordinate_lookup(values) -> dict[float, int]:
    return {coordinate_key(value): index for index, value in enumerate(values)}


def map_coordinate_indices(series, lookup: dict[float, int], np):
    indices = []
    missing = []
    for value in series:
        index = lookup.get(coordinate_key(value))
        if index is None:
            missing.append(float(value))
            continue
        indices.append(index)

    if missing:
        examples = ", ".join(str(value) for value in missing[:5])
        raise RuntimeError(
            "Cannot map coordinate value(s) onto the NetCDF grid. "
            f"Examples: {examples}"
        )
    return np.asarray(indices, dtype=np.int64)


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


def prediction_variable_attrs(column: str, latitudes, longitudes, np) -> dict[str, object]:
    target = column.removeprefix("pred_").removeprefix("gt_")
    long_name, units = PREDICTION_TARGET_METADATA.get(
        target,
        (column.replace("_", " "), "unknown"),
    )
    if column.startswith("gt_"):
        long_name = f"Ground truth {long_name.removeprefix('Predicted ').lower()}"

    lat_increment = coordinate_increment(latitudes, np)
    lon_increment = coordinate_increment(longitudes, np)
    attrs: dict[str, object] = {
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
        "GRIB_iDirectionIncrementInDegrees": lon_increment,
        "GRIB_iScansNegatively": 0,
        "GRIB_jDirectionIncrementInDegrees": lat_increment,
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
    return attrs


def coordinate_increment(values, np) -> float:
    if len(values) < 2:
        return float("nan")
    diffs = np.diff(np.asarray(values, dtype=np.float64))
    return float(abs(np.nanmedian(diffs)))


def era5_netcdf_encoding(dataset, np) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {
        "number": {"dtype": "int64"},
        "latitude": {"dtype": "float64", "_FillValue": np.nan},
        "longitude": {"dtype": "float64", "_FillValue": np.nan},
    }
    if np.issubdtype(dataset[ERA5_TIME_DIM].dtype, np.datetime64):
        encoding[ERA5_TIME_DIM] = {
            "dtype": "int64",
            "units": "seconds since 1970-01-01",
            "calendar": "proleptic_gregorian",
        }
    else:
        encoding[ERA5_TIME_DIM] = {"dtype": "int64"}

    if all(dim in dataset.sizes for dim in ERA5_CUBE_DIMS):
        cube_chunks = era5_cube_chunks(
            tuple(int(dataset.sizes[dim]) for dim in ERA5_CUBE_DIMS)
        )
    else:
        cube_chunks = None

    for name, variable in dataset.data_vars.items():
        if variable.dtype.kind in {"f", "i", "u"}:
            variable_encoding: dict[str, object] = {
                "zlib": True,
                "complevel": 1,
                "shuffle": True,
            }
            if variable.dtype.kind == "f":
                variable_encoding["dtype"] = str(variable.dtype)
                variable_encoding["_FillValue"] = (
                    np.float32(np.nan) if variable.dtype == np.float32 else np.nan
                )
            if variable.dims == ERA5_CUBE_DIMS and cube_chunks is not None:
                variable_encoding["chunksizes"] = cube_chunks
            elif variable.dims == ERA5_SPATIAL_DIMS:
                variable_encoding["chunksizes"] = cube_chunks[1:] if cube_chunks is not None else None
            encoding[name] = {
                key: value
                for key, value in variable_encoding.items()
                if value is not None
            }
    return encoding


def era5_cube_chunks(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(
        max(1, min(size, target))
        for size, target in zip(shape, ERA5_CHUNK_TARGETS)
    )


def timestamp_to_valid_time_coordinate(timestamps, np, pd):
    timestamp_text = pd.Series(timestamps).astype("int64").astype(str)
    lengths = set(timestamp_text.str.len())
    if lengths == {14}:
        try:
            return pd.to_datetime(
                timestamp_text,
                format="%Y%m%d%H%M%S",
                errors="raise",
            ).to_numpy(dtype="datetime64[ns]")
        except ValueError:
            pass
    if lengths == {8}:
        try:
            return pd.to_datetime(
                timestamp_text,
                format="%Y%m%d",
                errors="raise",
            ).to_numpy(dtype="datetime64[ns]")
        except ValueError:
            pass

    return np.asarray(timestamps, dtype=np.int64)


def add_igbp_variable(
    data_vars,
    df,
    cube_shape,
    lat_indices,
    lon_indices,
    time_indices,
    lat_lookup,
    lon_lookup,
    np,
):
    igbp_values = df["igbp"].fillna("").astype(str)
    unique_by_cell = (
        df.assign(igbp=igbp_values)
        .groupby(["lat", "lon"], sort=False)["igbp"]
        .nunique(dropna=False)
    )
    igbp_is_static = bool((unique_by_cell <= 1).all())

    if igbp_is_static:
        array = np.full(cube_shape[1:], "", dtype="<U3")
        cell_values = (
            df.assign(igbp=igbp_values)[["lat", "lon", "igbp"]]
            .drop_duplicates(["lat", "lon"])
        )
        cell_lat_indices = map_coordinate_indices(cell_values["lat"], lat_lookup, np)
        cell_lon_indices = map_coordinate_indices(cell_values["lon"], lon_lookup, np)
        array[cell_lat_indices, cell_lon_indices] = cell_values["igbp"].to_numpy(dtype="<U3")
        data_vars["igbp"] = (
            ERA5_SPATIAL_DIMS,
            array,
            {
                "long_name": "IGBP land cover class",
                "coordinates": "latitude longitude",
            },
        )
        return

    array = np.full(cube_shape, "", dtype="<U3")
    array[time_indices, lat_indices, lon_indices] = igbp_values.to_numpy(dtype="<U3")
    data_vars["igbp"] = (
        ERA5_CUBE_DIMS,
        array,
        {
            "long_name": "IGBP land cover class",
            "coordinates": ERA5_COORDINATES_ATTR,
        },
    )


def resolve_output_path(args: argparse.Namespace) -> Path:
    output_path = args.output_path or args.output_csv
    if output_path is None:
        raise ValueError("Provide --output-path, or --output-csv for CSV-compatible usage.")
    return output_path.expanduser().resolve()


def main():
    args = parse_args()
    shard_dir = args.shard_dir.expanduser().resolve()
    output_path = resolve_output_path(args)
    shard_paths = resolve_shard_paths(shard_dir, args.num_shards)
    prediction_targets = parse_prediction_targets(args.prediction_targets)

    print(f"Merging {len(shard_paths)} shard(s) from {shard_dir}")
    if prediction_targets is not None:
        print(f"Prediction targets: {', '.join(prediction_targets)}")
    if args.sort_output:
        print(
            f"Sorting output by {INTERNAL_ORDER_COLUMN} when present; "
            f"fallback: {', '.join(DEFAULT_SORT_COLUMNS)}"
        )
    if args.format == "csv":
        merge_csv_shards(
            shard_paths,
            output_path,
            prediction_targets,
            sort_output=args.sort_output,
            sort_tmp_dir=args.sort_tmp_dir,
        )
        print(f"Saved merged CSV to {output_path}")
    else:
        write_netcdf_from_csv_shards(
            shard_paths,
            output_path,
            prediction_targets,
            sort_output=args.sort_output,
            duplicate_policy=args.netcdf_duplicate_policy,
        )
        print(f"Saved NetCDF to {output_path}")

    if args.cleanup:
        shutil.rmtree(shard_dir)
        print(f"Removed shard directory {shard_dir}")


if __name__ == "__main__":
    main()
