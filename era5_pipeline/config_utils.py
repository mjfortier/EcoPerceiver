"""Shared date-range and path inference for the ERA5/MODIS pipeline."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineDateRange:
    start: date
    end: date
    label: str

    @property
    def start_datetime(self) -> datetime:
        return datetime(self.start.year, self.start.month, self.start.day, 0, 0, 0)

    @property
    def end_datetime(self) -> datetime:
        return datetime(self.end.year, self.end.month, self.end.day, 23, 0, 0)

    @property
    def local_window_start(self) -> str:
        return f"{self.start.isoformat()} 00:00:00"

    @property
    def local_window_end(self) -> str:
        return f"{self.end.isoformat()} 23:59:59"


@dataclass(frozen=True)
class PipelinePaths:
    data_root: Path
    output_dir: Path
    db_path: Path
    netcdf_dir: Path
    zip_dir: Path
    raw_modis_dir: Path
    sqlite_temp_dir: Path
    modis_landcover_path: Path


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {}) or {}
    if not isinstance(value, dict):
        raise SystemExit(f"Config section {name!r} must be a YAML mapping.")
    return value


def resolve_repo_path(
    value: str | Path | None,
    repo_root: Path,
    *,
    default: Path | None = None,
) -> Path | None:
    if value is None:
        return default

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def resolve_raw_modis_file(
    value: str | Path | None,
    repo_root: Path,
    raw_modis_dir: Path,
    *,
    default_name: str,
) -> Path:
    if value is None:
        return raw_modis_dir / default_name

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return raw_modis_dir / path
    return repo_root / path


def parse_config_date(value: Any, *, field_name: str) -> date:
    if value is None:
        raise SystemExit(f"Config field {field_name!r} is required.")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise SystemExit(
            f"Config field {field_name!r} must be an ISO date like YYYY-MM-DD."
        ) from exc


def date_range_label(start: date, end: date) -> str:
    full_year_start = start.month == 1 and start.day == 1
    full_year_end = (
        end.month == 12
        and end.day == calendar.monthrange(end.year, 12)[1]
    )
    if full_year_start and full_year_end:
        if start.year == end.year:
            return str(start.year)
        return f"{start.year}_{end.year}"
    return f"{start:%Y%m%d}_{end:%Y%m%d}"


def configured_date_range(config: dict[str, Any]) -> PipelineDateRange:
    start = parse_config_date(config.get("start_date"), field_name="start_date")
    end = parse_config_date(config.get("end_date"), field_name="end_date")
    if end < start:
        raise SystemExit(
            f"Config field 'end_date' must be on or after 'start_date': {start} > {end}."
        )
    return PipelineDateRange(start=start, end=end, label=date_range_label(start, end))


def inferred_pipeline_paths(config: dict[str, Any], repo_root: Path) -> PipelinePaths:
    date_range = configured_date_range(config)
    path_config = config_section(config, "paths")
    index_config = config_section(config, "index_era5")
    download_modis_config = config_section(config, "download_modis")
    assign_igbp_config = config_section(config, "assign_igbp_from_modis")
    process_modis_config = config_section(config, "process_modis")

    data_root = resolve_repo_path(
        path_config.get("data_root", "experiments/data"),
        repo_root,
    )
    assert data_root is not None

    output_dir = resolve_repo_path(
        path_config.get("output_dir"),
        repo_root,
        default=data_root / date_range.label,
    )
    assert output_dir is not None

    db_path = resolve_repo_path(
        path_config.get("db_path"),
        repo_root,
        default=output_dir / f"era5_{date_range.label}.db",
    )
    assert db_path is not None

    netcdf_dir = resolve_repo_path(
        path_config.get("netcdf_dir"),
        repo_root,
        default=output_dir / "era5_data",
    )
    assert netcdf_dir is not None

    zip_dir = resolve_repo_path(
        path_config.get("zip_dir"),
        repo_root,
        default=output_dir / "era5_zip",
    )
    assert zip_dir is not None

    raw_modis_dir = resolve_repo_path(
        (
            path_config.get("raw_modis_dir")
            or download_modis_config.get("output_dir")
            or process_modis_config.get("input_dir")
        ),
        repo_root,
        default=output_dir / "raw_modis",
    )
    assert raw_modis_dir is not None

    sqlite_temp_dir = resolve_repo_path(
        index_config.get("temp_dir") or path_config.get("sqlite_temp_dir"),
        repo_root,
        default=output_dir / "sqlite-tmp",
    )
    assert sqlite_temp_dir is not None

    modis_landcover_path = resolve_raw_modis_file(
        assign_igbp_config.get("modis_path"),
        repo_root,
        raw_modis_dir,
        default_name=f"{date_range.end.year}01011200C1.tiff",
    )

    return PipelinePaths(
        data_root=data_root,
        output_dir=output_dir,
        db_path=db_path,
        netcdf_dir=netcdf_dir,
        zip_dir=zip_dir,
        raw_modis_dir=raw_modis_dir,
        sqlite_temp_dir=sqlite_temp_dir,
        modis_landcover_path=modis_landcover_path,
    )
