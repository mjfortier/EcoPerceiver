#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${ECOPERCEIVER_ENV:-}" && -f "$ECOPERCEIVER_ENV/bin/activate" ]]; then
  # Optional override for standalone use outside the usual SCRATCH environment.
  source "$ECOPERCEIVER_ENV/bin/activate"
elif [[ -n "${SCRATCH:-}" && -f "$SCRATCH/env/ecoperceiver/bin/activate" ]]; then
  source "$SCRATCH/env/ecoperceiver/bin/activate"
fi

export PYTHONUNBUFFERED=1

python3 - "$@" <<'PY'
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
import sys


DEFAULT_RUN_PATH = (
    "experiments/runs/"
    "final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0"
)
TIME_DIM = "valid_time"
SPATIAL_DIMS = ("latitude", "longitude")
CUBE_DIMS = (TIME_DIM, *SPATIAL_DIMS)
CHUNK_TARGETS = (186, 31, 360)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_year_nc.sh",
        description=(
            "Unify four quarterly EcoPerceiver ERA5 NetCDF outputs into one "
            "calendar-year NetCDF file."
        )
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
        help="Directory containing quarter .nc files. Default: <run-path>/eval.",
    )
    parser.add_argument(
        "--output-path",
        default=os.environ.get("OUTPUT_PATH"),
        help="Yearly NetCDF output path. Default: <input-dir>/era5_predictions_<YEAR>.nc.",
    )
    parser.add_argument(
        "--engine",
        default=os.environ.get("XARRAY_ENGINE", "h5netcdf"),
        help="xarray NetCDF engine. Default: h5netcdf.",
    )
    parser.add_argument(
        "--inputs",
        nargs=4,
        metavar="PATH",
        help=(
            "Explicit Q1 Q2 Q3 Q4 NetCDF paths. If omitted, paths are inferred "
            "from the quarterly post-processing output naming convention."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Use the existing quarter files and skip missing inferred inputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print input and output paths without writing the assembled file.",
    )
    args = parser.parse_args()

    year = args.year_option or args.year
    if year is None:
        parser.error("YEAR is required as a positional argument or --year.")
    if not year.isdigit() or len(year) != 4:
        parser.error(f"YEAR must be a four-digit year, got: {year}")
    args.year = year
    return args


def quarter_paths(year: str, input_dir: Path) -> list[Path]:
    ranges = (
        (f"{year}-01-01", f"{year}-03-31"),
        (f"{year}-04-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-12-31"),
    )
    paths = []
    for start, end in ranges:
        date_tag = f"{start.replace('-', '')}_to_{end.replace('-', '')}"
        paths.append(input_dir / f"era5_predictions_{date_tag}.nc")
    return paths


def resolve_paths(args: argparse.Namespace) -> tuple[list[Path], Path]:
    run_path = Path(args.run_path).expanduser()
    input_dir = Path(args.input_dir).expanduser() if args.input_dir else run_path / "eval"
    if args.inputs:
        inputs = [Path(path).expanduser() for path in args.inputs]
    else:
        inputs = quarter_paths(args.year, input_dir)

    output_path = (
        Path(args.output_path).expanduser()
        if args.output_path
        else input_dir / f"era5_predictions_{args.year}.nc"
    )
    return inputs, output_path


def existing_inputs(inputs: list[Path], allow_missing: bool) -> list[Path]:
    missing = [path for path in inputs if not path.is_file()]
    if missing and not allow_missing:
        missing_list = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing quarter NetCDF input(s):\n  {missing_list}")

    present = [path for path in inputs if path.is_file()]
    if not present:
        raise FileNotFoundError("No quarter NetCDF inputs found.")
    if allow_missing and len(present) < len(inputs):
        print(f"Skipping {len(inputs) - len(present)} missing quarter input(s).")
    return present


def has_duplicate_times(dataset, pandas) -> bool:
    if TIME_DIM not in dataset.coords:
        raise RuntimeError(f"Assembled dataset is missing required coordinate: {TIME_DIM}")
    index = dataset.indexes.get(TIME_DIM)
    if index is None:
        index = pandas.Index(dataset[TIME_DIM].values)
    return bool(index.has_duplicates)


def build_encoding(dataset, numpy) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}

    if TIME_DIM in dataset.variables:
        if numpy.issubdtype(dataset[TIME_DIM].dtype, numpy.datetime64):
            encoding[TIME_DIM] = {
                "dtype": "int64",
                "units": "seconds since 1970-01-01",
                "calendar": "proleptic_gregorian",
            }
        else:
            encoding[TIME_DIM] = {"dtype": "int64"}

    for coord in SPATIAL_DIMS:
        if coord in dataset.variables:
            encoding[coord] = {"dtype": "float64", "_FillValue": numpy.nan}

    cube_chunks = None
    if all(dim in dataset.sizes for dim in CUBE_DIMS):
        cube_chunks = tuple(
            max(1, min(int(dataset.sizes[dim]), target))
            for dim, target in zip(CUBE_DIMS, CHUNK_TARGETS)
        )

    for name, variable in dataset.data_vars.items():
        if variable.dtype.kind not in {"f", "i", "u"}:
            continue

        variable_encoding: dict[str, object] = {
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
        }
        if variable.dtype.kind == "f":
            variable_encoding["dtype"] = str(variable.dtype)
            variable_encoding["_FillValue"] = (
                numpy.float32(numpy.nan) if variable.dtype == numpy.float32 else numpy.nan
            )
        if variable.dims == CUBE_DIMS and cube_chunks is not None:
            variable_encoding["chunksizes"] = cube_chunks
        elif variable.dims == SPATIAL_DIMS and cube_chunks is not None:
            variable_encoding["chunksizes"] = cube_chunks[1:]
        encoding[name] = variable_encoding

    return encoding


def assemble_year(inputs: list[Path], output_path: Path, engine: str, overwrite: bool) -> None:
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "Merging NetCDF files requires numpy, pandas, and xarray in the active environment."
        ) from exc

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite to replace it: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = []
    try:
        for path in inputs:
            ds = xr.open_dataset(path, engine=engine)
            if TIME_DIM not in ds.dims and TIME_DIM not in ds.coords:
                raise RuntimeError(f"{path} is missing required time dimension: {TIME_DIM}")
            datasets.append(ds)

        assembled = xr.concat(
            datasets,
            dim=TIME_DIM,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="exact",
            combine_attrs="override",
        )
        assembled = assembled.sortby(TIME_DIM)
        if has_duplicate_times(assembled, pd):
            raise RuntimeError(
                f"Duplicate {TIME_DIM} coordinate values found after assembly. "
                "Check for overlapping quarter files."
            )

        history = assembled.attrs.get("history", "")
        assembly_note = (
            f"{datetime.now(timezone.utc).isoformat()} assembled quarterly "
            "EcoPerceiver ERA5 NetCDF outputs into one calendar-year file"
        )
        assembled.attrs["history"] = f"{history}\n{assembly_note}".strip()
        assembled.attrs["assembled_input_files"] = " ".join(str(path) for path in inputs)
        assembled.attrs["num_assembled_input_files"] = len(inputs)

        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.unlink(missing_ok=True)
        try:
            assembled.to_netcdf(tmp_path, engine=engine, encoding=build_encoding(assembled, np))
            tmp_path.replace(output_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        assembled.close()
    finally:
        for dataset in datasets:
            dataset.close()


def main() -> int:
    args = parse_args()
    inputs, output_path = resolve_paths(args)
    if not args.dry_run:
        inputs = existing_inputs(inputs, args.allow_missing)

    print("ERA5 yearly NetCDF assembly")
    print(f"Year: {args.year}")
    print("Inputs:")
    for path in inputs:
        print(f"  {path}")
    print(f"Output: {output_path}")
    print(f"Engine: {args.engine}")

    if args.dry_run:
        print("Dry run only; no file written.")
        return 0

    assemble_year(inputs, output_path, args.engine, args.overwrite)
    print(f"Saved yearly NetCDF to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
PY
