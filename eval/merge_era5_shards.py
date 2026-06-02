import argparse
import shutil
from pathlib import Path


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


def merge_csv_shards(shard_paths: list[Path], output_csv_path: Path):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_csv_path.with_suffix(output_csv_path.suffix + ".tmp")
    header_written = False
    expected_header = None

    with output_tmp.open("wb") as out_file:
        for shard_path in shard_paths:
            with shard_path.open("rb") as shard_file:
                header = shard_file.readline()
                if not header:
                    continue
                if expected_header is None:
                    expected_header = header
                elif header != expected_header:
                    raise RuntimeError(f"CSV header mismatch in shard {shard_path}")

                if not header_written:
                    out_file.write(header)
                    header_written = True
                shutil.copyfileobj(shard_file, out_file, length=1024 * 1024)

    output_tmp.replace(output_csv_path)


def write_netcdf_from_csv_shards(shard_paths: list[Path], output_netcdf_path: Path):
    try:
        import pandas as pd
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "NetCDF output requires pandas and xarray in the active environment."
        ) from exc

    output_netcdf_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(shard_path) for shard_path in shard_paths]
    if not frames:
        raise RuntimeError("No shard rows were available to write NetCDF output.")

    df = pd.concat(frames, ignore_index=True)
    for column in df.columns:
        if column in {"lat", "lon", "timestamp"} or column.startswith(("pred_", "gt_")):
            numeric = pd.to_numeric(df[column], errors="coerce")
            if column == "timestamp" and not numeric.isna().any():
                df[column] = numeric.astype("int64")
            else:
                df[column] = numeric

    data_vars = {}
    for column in df.columns:
        series = df[column]
        if series.dtype == object:
            data_vars[column] = ("sample", series.fillna("").astype(str).to_numpy())
        else:
            data_vars[column] = ("sample", series.to_numpy())

    dataset = xr.Dataset(data_vars=data_vars)
    dataset.attrs["source"] = "EcoPerceiver ERA5 torchrun inference shards"
    dataset.attrs["num_shards"] = len(shard_paths)

    output_tmp = output_netcdf_path.with_suffix(output_netcdf_path.suffix + ".tmp")
    dataset.to_netcdf(output_tmp, engine="h5netcdf")
    output_tmp.replace(output_netcdf_path)


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

    print(f"Merging {len(shard_paths)} shard(s) from {shard_dir}")
    if args.format == "csv":
        merge_csv_shards(shard_paths, output_path)
        print(f"Saved merged CSV to {output_path}")
    else:
        write_netcdf_from_csv_shards(shard_paths, output_path)
        print(f"Saved NetCDF to {output_path}")

    if args.cleanup:
        shutil.rmtree(shard_dir)
        print(f"Removed shard directory {shard_dir}")


if __name__ == "__main__":
    main()
