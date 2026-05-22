#!/usr/bin/env python3
"""Run selected EcoPerceiver MODIS/ERA5 pipeline steps."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "pipeline_config.yml"
DEFAULT_STEPS = ("drop_nighttime", "drop_areas", "rebuild_ids_era5", "transform_modis")


@dataclass(frozen=True)
class PipelineStep:
    name: str
    description: str


STEPS = {
    "download_modis": PipelineStep(
        name="download_modis",
        description="Download raw MODIS GeoTIFFs from Earth Engine.",
    ),
    "drop_nighttime": PipelineStep(
        name="drop_nighttime",
        description="Drop ec_data rows with solar radiation below the daytime threshold.",
    ),
    "drop_areas": PipelineStep(
        name="drop_areas",
        description="Drop ec_data rows inside configured geographic areas.",
    ),
    "rebuild_ids_era5": PipelineStep(
        name="rebuild_ids_era5",
        description="Rebuild ec_data ids once after all row filters.",
    ),
    "transform_modis": PipelineStep(
        name="transform_modis",
        description="Transform MODIS GeoTIFFs into modis_data SQLite rows.",
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
    download_config = config_section(config, "download_modis")
    transform_config = config_section(config, "transform_modis")
    drop_config = config_section(config, "drop_nighttime")
    drop_areas_config = config_section(config, "drop_areas")
    rebuild_ids_config = config_section(config, "rebuild_ids_era5")

    default_steps = config_list(
        pipeline_config.get("steps"),
        default=list(DEFAULT_STEPS),
        field_name="pipeline.steps",
    )
    validate_steps(default_steps)

    default_modis_dates = config_list(
        transform_config.get("dates"),
        default=[],
        field_name="transform_modis.dates",
    )
    default_drop_areas = config_list(
        drop_areas_config.get("areas"),
        default=[],
        field_name="drop_areas.areas",
    )

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description=(
            "Run selected pieces of the EcoPerceiver ERA5/MODIS pipeline. "
            "By default, download_modis is not run."
        ),
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
        "--include-download",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            pipeline_config.get("include_download"),
            default=False,
            field_name="pipeline.include_download",
        ),
        help="Prepend download_modis to the selected steps if it is not already present.",
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
        default=resolve_repo_path(
            pipeline_config.get("db_path", "experiments/data/poc_era5.db")
        ),
        help="SQLite DB used by transform/filter steps.",
    )

    add_download_args(parser, download_config)
    add_transform_args(parser, transform_config)
    add_drop_nighttime_args(parser, drop_config)
    add_drop_areas_args(parser, drop_areas_config)
    add_rebuild_ids_args(parser, rebuild_ids_config)
    args = parser.parse_args()

    if args.modis_dates is None:
        args.modis_dates = default_modis_dates
    if args.drop_areas is None:
        args.drop_areas = default_drop_areas
    validate_steps(list(args.steps))
    return args


def add_download_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
) -> None:
    group = parser.add_argument_group("download_modis options")
    group.add_argument(
        "--download-start-date",
        default=config.get("start_date"),
        help="Optional download start date in YYYY-MM-DD format.",
    )
    group.add_argument(
        "--download-end-date",
        default=config.get("end_date"),
        help="Optional download end date in YYYY-MM-DD format.",
    )
    group.add_argument(
        "--download-output-dir",
        type=Path,
        default=resolve_repo_path(config.get("output_dir", "experiments/data/raw_modis")),
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


def add_transform_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
) -> None:
    group = parser.add_argument_group("transform_modis options")
    group.add_argument(
        "--modis-input-dir",
        type=Path,
        default=resolve_repo_path(config.get("input_dir", "experiments/data/raw_modis")),
        help="Raw MODIS GeoTIFF directory.",
    )
    group.add_argument(
        "--modis-date",
        dest="modis_dates",
        action="append",
        default=None,
        help=(
            "Specific MODIS timestamp to transform, e.g. 201712031200. "
            "Repeat for multiple dates."
        ),
    )
    group.add_argument(
        "--limit-modis-dates",
        type=int,
        default=config.get("limit_dates"),
        help="Limit the number of MODIS timestamp pairs transformed.",
    )
    group.add_argument(
        "--modis-example-count",
        type=int,
        default=config.get("example_count", 10),
        help="Number of transformed MODIS example cells to print.",
    )
    group.add_argument(
        "--overwrite-modis",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("overwrite"),
            default=False,
            field_name="transform_modis.overwrite",
        ),
        help="Overwrite existing modis_data rows for matching coord/date pairs.",
    )
    group.add_argument(
        "--active-ec-coords-only",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("active_ec_coords_only"),
            default=True,
            field_name="transform_modis.active_ec_coords_only",
        ),
        help="Only transform MODIS for coord_id values that remain in ec_data.",
    )


def add_drop_nighttime_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
) -> None:
    group = parser.add_argument_group("drop_nighttime options")
    group.add_argument(
        "--radiation-column",
        default=config.get("radiation_column", "SW_IN"),
        help="Solar radiation column used to detect nighttime.",
    )
    group.add_argument(
        "--threshold-w-m2",
        type=float,
        default=config.get("threshold_w_m2", 2.0),
        help="Daytime threshold in W m-2.",
    )
    group.add_argument(
        "--drop-missing-radiation",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("drop_missing_radiation"),
            default=False,
            field_name="drop_nighttime.drop_missing_radiation",
        ),
        help="Also drop rows with NULL radiation.",
    )
    group.add_argument(
        "--vacuum",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("vacuum"),
            default=False,
            field_name="drop_nighttime.vacuum",
        ),
        help="Run VACUUM after dropping nighttime rows.",
    )
    group.add_argument(
        "--filter-dry-run",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("dry_run"),
            default=False,
            field_name="drop_nighttime.dry_run",
        ),
        help="Run the nighttime filter in dry-run mode even when the pipeline runs.",
    )
    group.add_argument(
        "--nighttime-delete-chunk-size",
        type=int,
        default=config.get("delete_chunk_size", 250_000),
        help="Number of id values scanned per drop_nighttime delete chunk.",
    )


def add_drop_areas_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
) -> None:
    group = parser.add_argument_group("drop_areas options")
    group.add_argument(
        "--areas-config",
        type=Path,
        default=resolve_repo_path(
            config.get("areas_config", "scripts/modis/drop_areas_config.yml")
        ),
        help="YAML file with geographic area definitions.",
    )
    group.add_argument(
        "--drop-area",
        dest="drop_areas",
        action="append",
        default=None,
        help=(
            "Area name to drop. Repeat for multiple areas. "
            "Default: enabled areas in --areas-config."
        ),
    )
    group.add_argument(
        "--areas-vacuum",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("vacuum"),
            default=False,
            field_name="drop_areas.vacuum",
        ),
        help="Run VACUUM after dropping configured areas.",
    )
    group.add_argument(
        "--areas-dry-run",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("dry_run"),
            default=False,
            field_name="drop_areas.dry_run",
        ),
        help="Run the area filter in dry-run mode even when the pipeline runs.",
    )
    group.add_argument(
        "--areas-delete-chunk-size",
        type=int,
        default=config.get(
            "delete_chunk_size",
            config.get("delete_coord_chunk_size", 2_000_000),
        ),
        help="Number of ec_data id values scanned per drop_areas delete chunk.",
    )


def add_rebuild_ids_args(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
) -> None:
    group = parser.add_argument_group("rebuild_ids_era5 options")
    group.add_argument(
        "--rebuild-ids-table",
        default=config.get("table", "ec_data"),
        help="ERA5 table whose id column should be rebuilt.",
    )
    group.add_argument(
        "--rebuild-ids-vacuum",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("vacuum"),
            default=False,
            field_name="rebuild_ids_era5.vacuum",
        ),
        help="Run VACUUM after the final ERA5 id rebuild step.",
    )
    group.add_argument(
        "--rebuild-ids-dry-run",
        action=argparse.BooleanOptionalAction,
        default=bool_config(
            config.get("dry_run"),
            default=False,
            field_name="rebuild_ids_era5.dry_run",
        ),
        help="Run the final ERA5 id rebuild step in dry-run mode.",
    )
    group.add_argument(
        "--rebuild-ids-copy-chunk-size",
        type=int,
        default=config.get("copy_chunk_size", 100_000),
        help="Number of rows copied per rebuild_ids_era5 chunk.",
    )


def selected_steps(args: argparse.Namespace) -> list[str]:
    steps = list(args.steps)
    if args.include_download and "download_modis" not in steps:
        steps.insert(0, "download_modis")
    return steps


def print_available_steps(default_steps: list[str]) -> None:
    print(f"Configured default steps: {' '.join(default_steps)}")
    print()
    print("Available steps:")
    for name, step in STEPS.items():
        default_marker = " (default)" if name in default_steps else ""
        print(f"  {name}{default_marker}: {step.description}")


def command_for_step(step: str, args: argparse.Namespace) -> list[str]:
    if step == "download_modis":
        return download_modis_command(args)
    if step == "transform_modis":
        return transform_modis_command(args)
    if step == "drop_nighttime":
        return drop_nighttime_command(args)
    if step == "drop_areas":
        return drop_areas_command(args)
    if step == "rebuild_ids_era5":
        return rebuild_ids_era5_command(args)
    raise ValueError(f"Unsupported pipeline step: {step}")


def download_modis_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "modis" / "download_modis.py"),
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
    if args.download_overwrite:
        command.append("--overwrite")
    if args.download_authenticate:
        command.append("--authenticate")
    return command


def transform_modis_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "modis" / "transform_modis.py"),
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


def drop_nighttime_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "modis" / "drop_nighttime_era5.py"),
        "--db-path",
        str(args.db_path),
        "--radiation-column",
        args.radiation_column,
        "--threshold-w-m2",
        str(args.threshold_w_m2),
        "--delete-chunk-size",
        str(args.nighttime_delete_chunk_size),
    ]
    if args.drop_missing_radiation:
        command.append("--drop-missing-radiation")
    if args.vacuum:
        command.append("--vacuum")
    if args.filter_dry_run:
        command.append("--dry-run")
    return command


def drop_areas_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "modis" / "drop_areas_era5.py"),
        "--db-path",
        str(args.db_path),
        "--areas-config",
        str(args.areas_config),
        "--delete-chunk-size",
        str(args.areas_delete_chunk_size),
    ]
    for area_name in args.drop_areas:
        command.extend(["--area", area_name])
    if args.areas_vacuum:
        command.append("--vacuum")
    if args.areas_dry_run:
        command.append("--dry-run")
    return command


def rebuild_ids_era5_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "modis" / "rebuild_ids_era5.py"),
        "--db-path",
        str(args.db_path),
        "--table",
        args.rebuild_ids_table,
        "--copy-chunk-size",
        str(args.rebuild_ids_copy_chunk_size),
    ]
    if args.rebuild_ids_vacuum:
        command.append("--vacuum")
    if args.rebuild_ids_dry_run:
        command.append("--dry-run")
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

    steps = selected_steps(args)
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
