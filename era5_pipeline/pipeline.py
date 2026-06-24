#!/usr/bin/env python3
"""Run the unified EcoPerceiver ERA5/MODIS data pipeline."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from config_utils import (
    PipelineDateRange,
    PipelinePaths,
    configured_date_range,
    inferred_pipeline_paths,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "pipeline_config.yml"
DEFAULT_STEPS = (
    "download_era5",
    "process_era5",
)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    description: str


STEPS = {
    "download_era5": PipelineStep(
        name="download_era5",
        description="Download ERA5 NetCDF chunks from CDS.",
    ),
    "process_era5": PipelineStep(
        name="process_era5",
        description="Convert ERA5 NetCDF chunks to the EcoPerceiver SQLite DB.",
    ),
    "index_era5": PipelineStep(
        name="index_era5",
        description="Create persistent ec_data indexes for ERA5 eval and MODIS lookup.",
    ),
    "download_modis": PipelineStep(
        name="download_modis",
        description="Download raw MODIS GeoTIFFs from Earth Engine.",
    ),
    "assign_igbp_from_modis": PipelineStep(
        name="assign_igbp_from_modis",
        description="Assign coord_data.igbp from a MODIS land-cover raster.",
    ),
    "process_modis": PipelineStep(
        name="process_modis",
        description="Process MODIS GeoTIFFs into modis_data SQLite rows.",
    ),
}


def resolve_cli_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def resolve_repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    if yaml is None:
        raise SystemExit(
            "Missing Python dependency: PyYAML. Install it or run inside the "
            "project environment."
        )

    resolved_path = resolve_cli_path(config_path)
    if not resolved_path.exists():
        raise SystemExit(f"Pipeline config does not exist: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise SystemExit(f"Pipeline config must be a YAML mapping: {resolved_path}")

    return config, resolved_path


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {}) or {}
    if not isinstance(value, dict):
        raise SystemExit(f"Config section {name!r} must be a YAML mapping.")
    return value


def config_list(value: Any, *, default: list[str], field_name: str) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise SystemExit(f"Config field {field_name!r} must be a string or list.")


def config_optional_list(value: Any, *, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise SystemExit(f"Config field {field_name!r} must be a string or list.")


def bool_config(value: Any, *, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise SystemExit(f"Config field {field_name!r} must be true or false.")


def validate_steps(steps: list[str]) -> None:
    unknown_steps = [step for step in steps if step not in STEPS]
    if unknown_steps:
        raise SystemExit(
            "Unknown pipeline step(s): "
            f"{', '.join(unknown_steps)}. Valid steps: {', '.join(STEPS)}"
        )


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Pipeline YAML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    config_args, _ = config_parser.parse_known_args()
    config, resolved_config_path = load_config(config_args.config_path)

    pipeline_config = config_section(config, "pipeline")
    date_range = configured_date_range(config)
    inferred_paths = inferred_pipeline_paths(config, REPO_ROOT)
    index_era5_config = config_section(config, "index_era5")
    download_modis_config = config_section(config, "download_modis")
    assign_igbp_config = config_section(config, "assign_igbp_from_modis")
    process_modis_config = config_section(config, "process_modis")

    default_steps = config_list(
        pipeline_config.get("steps"),
        default=list(DEFAULT_STEPS),
        field_name="pipeline.steps",
    )
    validate_steps(default_steps)

    default_modis_dates = config_list(
        process_modis_config.get("dates"),
        default=[],
        field_name="process_modis.dates",
    )
    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Run selected pieces of the unified ERA5/MODIS pipeline.",
    )
    parser.set_defaults(config_path=resolved_config_path)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=tuple(STEPS),
        default=default_steps,
        help=f"Pipeline steps to run, in order. Config default: {' '.join(default_steps)}.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print available pipeline steps and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            pipeline_config.get("dry_run"),
            default=False,
            field_name="pipeline.dry_run",
        ),
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--python",
        default=pipeline_config.get("python") or sys.executable,
        help="Python executable used for steps.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=resolve_repo_path(pipeline_config.get("db_path")) or inferred_paths.db_path,
        help="SQLite DB used by ERA5/MODIS pipeline steps.",
    )
    parser.add_argument(
        "--limit-era5-groups",
        type=int,
        default=None,
        help="Limit ERA5 request/group count for smoke tests.",
    )
    parser.add_argument(
        "--era5-max-workers",
        type=int,
        default=None,
        help="Override download_era5.max_workers for concurrent ERA5 CDS downloads.",
    )
    parser.add_argument(
        "--overwrite-era5-downloads",
        action="store_true",
        help="Overwrite existing downloaded ERA5 chunks.",
    )
    parser.add_argument(
        "--overwrite-era5-db",
        action="store_true",
        help="Recreate the ERA5 SQLite DB during process_era5.",
    )

    add_index_era5_args(parser, index_era5_config, inferred_paths)
    add_download_modis_args(parser, download_modis_config, date_range, inferred_paths)
    add_assign_igbp_args(parser, assign_igbp_config, inferred_paths)
    add_process_modis_args(parser, process_modis_config, inferred_paths)
    args = parser.parse_args()

    if args.modis_dates is None:
        args.modis_dates = default_modis_dates
    if args.era5_max_workers is not None and args.era5_max_workers < 1:
        parser.error("--era5-max-workers must be at least 1.")
    validate_steps(list(args.steps))
    return args


def add_index_era5_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    inferred_paths: PipelinePaths,
) -> None:
    group = parser.add_argument_group("index_era5 options")
    group.add_argument(
        "--era5-index-table",
        default=config.get("table", "ec_data"),
        help="ERA5 table to index.",
    )
    group.add_argument(
        "--era5-index-name",
        default=config.get("index_name", "idx_ec_data_coord_id_timestamp_id"),
        help="Composite index name used by ERA5 eval.",
    )
    group.add_argument(
        "--era5-index-columns",
        nargs="+",
        default=config_list(
            config.get("columns"),
            default=["coord_id", "timestamp", "id"],
            field_name="index_era5.columns",
        ),
        help="Columns for the composite ERA5 eval index.",
    )
    group.add_argument(
        "--era5-index-analyze",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("analyze"),
            default=True,
            field_name="index_era5.analyze",
        ),
        help="Run ANALYZE after creating the index.",
    )
    group.add_argument(
        "--era5-index-dry-run",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("dry_run"),
            default=False,
            field_name="index_era5.dry_run",
        ),
        help="Run the index_era5 step in dry-run mode.",
    )
    group.add_argument(
        "--era5-index-temp-dir",
        type=Path,
        default=resolve_repo_path(config.get("temp_dir")) or inferred_paths.sqlite_temp_dir,
        help="SQLite temporary sorter directory for the ERA5 index build.",
    )
    group.add_argument(
        "--era5-index-journal-mode",
        choices=("DELETE", "TRUNCATE", "PERSIST", "WAL"),
        default=config.get("journal_mode", "DELETE"),
        help="SQLite journal mode to use while building the ERA5 index.",
    )
    group.add_argument(
        "--era5-index-skip-preflight",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("skip_preflight"),
            default=False,
            field_name="index_era5.skip_preflight",
        ),
        help="Skip disk/page-count estimates before creating the ERA5 index.",
    )
    group.add_argument(
        "--era5-index-progress",
        choices=("auto", "tqdm", "heartbeat", "none"),
        default=config.get("progress", "auto"),
        help="Progress display mode for the ERA5 index step.",
    )
    group.add_argument(
        "--era5-index-threads",
        type=int,
        default=config.get("threads", 8),
        help="SQLite worker threads for the ERA5 index step.",
    )


def add_download_modis_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    date_range: PipelineDateRange,
    inferred_paths: PipelinePaths,
) -> None:
    group = parser.add_argument_group("download_modis options")
    group.add_argument(
        "--download-start-date",
        default=config.get("start_date") or date_range.start.isoformat(),
        help="Optional MODIS download start date in YYYY-MM-DD format.",
    )
    group.add_argument(
        "--download-end-date",
        default=config.get("end_date") or date_range.end.isoformat(),
        help="Optional MODIS download end date in YYYY-MM-DD format.",
    )
    group.add_argument(
        "--download-output-dir",
        type=Path,
        default=resolve_repo_path(config.get("output_dir")) or inferred_paths.raw_modis_dir,
        help="MODIS download output directory.",
    )
    group.add_argument(
        "--download-products",
        nargs="+",
        default=config_optional_list(
            config.get("products"),
            field_name="download_modis.products",
        ),
        help="Optional MODIS products to download, e.g. A4 A2 C1.",
    )
    group.add_argument(
        "--download-max-workers",
        type=int,
        default=config.get("max_workers"),
        help="Concurrent MODIS tile downloads per image.",
    )
    group.add_argument(
        "--download-overwrite",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("overwrite"),
            default=False,
            field_name="download_modis.overwrite",
        ),
        help="Overwrite existing downloaded MODIS files.",
    )
    group.add_argument(
        "--download-authenticate",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("authenticate"),
            default=False,
            field_name="download_modis.authenticate",
        ),
        help="Run Earth Engine authentication before downloading.",
    )


def add_assign_igbp_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    inferred_paths: PipelinePaths,
) -> None:
    group = parser.add_argument_group("assign_igbp_from_modis options")
    group.add_argument(
        "--igbp-modis-path",
        dest="igbp_modis_path",
        type=Path,
        default=inferred_paths.modis_landcover_path,
        help="MODIS MCD12C1 GeoTIFF used to assign coord_data.igbp.",
    )
    group.add_argument(
        "--igbp-table",
        default=config.get("table", "coord_data"),
        help="ERA5 coordinate table containing lat, lon, and igbp columns.",
    )
    group.add_argument(
        "--igbp-only-null",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("only_null"),
            default=False,
            field_name="assign_igbp_from_modis.only_null",
        ),
        help="Only assign coord_data rows where igbp is NULL.",
    )
    group.add_argument(
        "--igbp-write",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("write"),
            default=True,
            field_name="assign_igbp_from_modis.write",
        ),
        help="Apply IGBP assignments. Use --no-igbp-write to run read-only.",
    )
    group.add_argument(
        "--igbp-batch-size",
        type=int,
        default=config.get("batch_size", 10_000),
        help="SQLite assignment batch size for assign_igbp_from_modis.",
    )


def add_process_modis_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    inferred_paths: PipelinePaths,
) -> None:
    group = parser.add_argument_group("process_modis options")
    group.add_argument(
        "--modis-input-dir",
        type=Path,
        default=resolve_repo_path(config.get("input_dir")) or inferred_paths.raw_modis_dir,
        help="Raw MODIS GeoTIFF directory.",
    )
    group.add_argument(
        "--modis-date",
        dest="modis_dates",
        action="append",
        default=None,
        help=(
            "Specific MODIS timestamp to process, e.g. 201712031200. "
            "Repeat for multiple dates."
        ),
    )
    group.add_argument(
        "--limit-modis-dates",
        type=int,
        default=config.get("limit_dates"),
        help="Limit the number of MODIS timestamp pairs processed.",
    )
    group.add_argument(
        "--modis-example-count",
        type=int,
        default=config.get("example_count", 10),
        help="Number of processed MODIS example cells to print.",
    )
    group.add_argument(
        "--overwrite-modis",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("overwrite"),
            default=False,
            field_name="process_modis.overwrite",
        ),
        help="Overwrite existing modis_data rows for matching coord/date pairs.",
    )
    group.add_argument(
        "--active-ec-coords-only",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("active_ec_coords_only"),
            default=True,
            field_name="process_modis.active_ec_coords_only",
        ),
        help="Only process MODIS for coord_id values that remain in ec_data.",
    )


def print_available_steps(default_steps: list[str]) -> None:
    print(f"Configured default steps: {' '.join(default_steps)}")
    print()
    print("Available steps:")
    for name, step in STEPS.items():
        default_marker = " (default)" if name in default_steps else ""
        print(f"  {name}{default_marker}: {step.description}")


def command_for_step(step: str, args: argparse.Namespace) -> list[str]:
    if step == "download_era5":
        return download_era5_command(args)
    if step == "process_era5":
        return process_era5_command(args)
    if step == "index_era5":
        return index_era5_command(args)
    if step == "download_modis":
        return download_modis_command(args)
    if step == "assign_igbp_from_modis":
        return assign_igbp_from_modis_command(args)
    if step == "process_modis":
        return process_modis_command(args)
    raise ValueError(f"Unsupported pipeline step: {step}")


def download_era5_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(SCRIPT_DIR / "download_era5.py"),
        "--config",
        str(args.config_path),
        "--download-only",
    ]
    if args.limit_era5_groups is not None:
        command.extend(["--limit-groups", str(args.limit_era5_groups)])
    if args.era5_max_workers is not None:
        command.extend(["--max-workers", str(args.era5_max_workers)])
    if args.overwrite_era5_downloads:
        command.append("--overwrite-downloads")
    return command


def process_era5_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(SCRIPT_DIR / "download_era5.py"),
        "--config",
        str(args.config_path),
        "--process-only",
    ]
    if args.limit_era5_groups is not None:
        command.extend(["--limit-groups", str(args.limit_era5_groups)])
    if args.overwrite_era5_db:
        command.append("--overwrite-db")
    return command


def index_era5_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(SCRIPT_DIR / "index_era5.py"),
        "--db-path",
        str(args.db_path),
        "--table",
        args.era5_index_table,
        "--index-name",
        args.era5_index_name,
        "--columns",
        *args.era5_index_columns,
    ]
    if not args.era5_index_analyze:
        command.append("--no-analyze")
    if args.era5_index_dry_run:
        command.append("--dry-run")
    if args.era5_index_temp_dir is not None:
        command.extend(["--temp-dir", str(args.era5_index_temp_dir)])
    command.extend(["--journal-mode", args.era5_index_journal_mode])
    command.extend(["--progress", args.era5_index_progress])
    command.extend(["--threads", str(args.era5_index_threads)])
    if args.era5_index_skip_preflight:
        command.append("--skip-preflight")
    return command


def download_modis_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(SCRIPT_DIR / "download_modis.py"),
    ]
    if args.download_start_date is not None:
        command.append(str(args.download_start_date))
    if args.download_end_date is not None:
        if args.download_start_date is None:
            raise SystemExit(
                "--download-end-date requires --download-start-date so the "
                "positional arguments stay unambiguous."
            )
        command.append(str(args.download_end_date))
    command.extend(["--output-dir", str(args.download_output_dir)])
    if args.download_products:
        command.extend(["--products", *args.download_products])
    if args.download_max_workers is not None:
        command.extend(["--max-workers", str(args.download_max_workers)])
    if args.download_overwrite:
        command.append("--overwrite")
    if args.download_authenticate:
        command.append("--authenticate")
    return command


def assign_igbp_from_modis_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(SCRIPT_DIR / "assign_igbp_from_modis.py"),
        "--db-path",
        str(args.db_path),
        "--modis-path",
        str(args.igbp_modis_path),
        "--table",
        args.igbp_table,
        "--batch-size",
        str(args.igbp_batch_size),
    ]
    if args.igbp_only_null:
        command.append("--only-null")
    if args.igbp_write:
        command.append("--write")
    return command


def process_modis_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(SCRIPT_DIR / "process_modis.py"),
        "--input-dir",
        str(args.modis_input_dir),
        "--db-path",
        str(args.db_path),
        "--example-count",
        str(args.modis_example_count),
    ]
    for modis_date in args.modis_dates:
        command.extend(["--date", modis_date])
    if args.limit_modis_dates is not None:
        command.extend(["--limit-dates", str(args.limit_modis_dates)])
    if args.overwrite_modis:
        command.append("--overwrite")
    if args.active_ec_coords_only:
        command.append("--active-ec-coords-only")
    return command


def run_command(command: list[str], dry_run: bool) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> int:
    args = parse_args()
    if args.list_steps:
        print_available_steps(list(args.steps))
        return 0

    steps = list(args.steps)
    print(f"Config: {args.config_path}")
    print(f"Selected pipeline steps: {' '.join(steps)}")
    for step in steps:
        print(f"\n==> {step}: {STEPS[step].description}", flush=True)
        run_command(command_for_step(step, args), dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run only; no pipeline commands were executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
