#!/usr/bin/env python3
"""Download ERA5 and export EcoPerceiver-compatible SQLite.

The download layout follows CarbonCast's ERA5/CDS request model: request
ERA5 single-level variables, unzip NetCDF chunks, convert ERA5 short names
to predictor variables, then write ``coord_data`` and ``ec_data`` tables.

The post-processing deliberately computes local time with minute-precision
UTC offsets. This avoids the India/Australia bug caused by assuming every
time-zone offset is a whole number of hours.
"""

from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import zipfile

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "pipeline_config.yml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_utils import configured_date_range, inferred_pipeline_paths

MISSING_DEPENDENCIES: dict[str, str] = {}

try:
    import yaml
except ModuleNotFoundError:
    MISSING_DEPENDENCIES["yaml"] = "PyYAML"
    yaml = None

try:
    import pandas as pd
except ModuleNotFoundError:
    MISSING_DEPENDENCIES["pandas"] = "pandas"
    pd = None

try:
    import xarray as xr
except ModuleNotFoundError:
    MISSING_DEPENDENCIES["xarray"] = "xarray"
    xr = None

try:
    import cdsapi
except ModuleNotFoundError:
    MISSING_DEPENDENCIES["cdsapi"] = "cdsapi"
    cdsapi = None

try:
    from timezonefinder import TimezoneFinder
except ModuleNotFoundError:
    MISSING_DEPENDENCIES["timezonefinder"] = "timezonefinder"
    TimezoneFinder = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

from ecoperceiver.constants import DEFAULT_NORM, EC_PREDICTORS


ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux_clear_sky",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_latent_heat_flux",
    "mean_surface_sensible_heat_flux",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "forecast_albedo",
    "friction_velocity",
    "geopotential",
]

SHORTNAME_TO_FULLNAME = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "sp": "surface_pressure",
    "tp": "total_precipitation",
    "avg_sdlwrf": "mean_surface_downward_long_wave_radiation_flux",
    "avg_sdswrf": "mean_surface_downward_short_wave_radiation_flux",
    "avg_sdswrfcs": "mean_surface_downward_short_wave_radiation_flux_clear_sky",
    "avg_snlwrf": "mean_surface_net_long_wave_radiation_flux",
    "avg_slhtf": "mean_surface_latent_heat_flux",
    "avg_ishf": "mean_surface_sensible_heat_flux",
    "stl1": "soil_temperature_level_1",
    "stl2": "soil_temperature_level_2",
    "stl3": "soil_temperature_level_3",
    "swvl1": "volumetric_soil_water_layer_1",
    "swvl2": "volumetric_soil_water_layer_2",
    "swvl3": "volumetric_soil_water_layer_3",
    "fal": "forecast_albedo",
    "zust": "friction_velocity",
    "z": "geopotential",
    # CDS/GRIB aliases observed in ERA5 NetCDF exports.
    "msdwlwrf": "mean_surface_downward_long_wave_radiation_flux",
    "msdwswrf": "mean_surface_downward_short_wave_radiation_flux",
    "msdwswrfcs": "mean_surface_downward_short_wave_radiation_flux_clear_sky",
    "msnlwrf": "mean_surface_net_long_wave_radiation_flux",
    "mslhf": "mean_surface_latent_heat_flux",
    "msshf": "mean_surface_sensible_heat_flux",
}

VARIABLES_FOR_PREDICTOR = {
    "TA": ["2m_temperature"],
    "P": ["total_precipitation"],
    "RH": ["2m_temperature", "2m_dewpoint_temperature"],
    "VPD": ["2m_temperature", "2m_dewpoint_temperature"],
    "PA": ["surface_pressure"],
    "CO2": ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure", "xco2"],
    "SW_IN": ["mean_surface_downward_short_wave_radiation_flux"],
    "SW_IN_POT": ["mean_surface_downward_short_wave_radiation_flux_clear_sky"],
    "SW_OUT": ["mean_surface_downward_short_wave_radiation_flux", "forecast_albedo"],
    "LW_IN": ["mean_surface_downward_long_wave_radiation_flux"],
    "LW_OUT": [
        "mean_surface_downward_long_wave_radiation_flux",
        "mean_surface_net_long_wave_radiation_flux",
    ],
    "NETRAD": [
        "mean_surface_downward_short_wave_radiation_flux",
        "mean_surface_downward_long_wave_radiation_flux",
        "mean_surface_net_long_wave_radiation_flux",
        "forecast_albedo",
    ],
    "WS": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    "WD": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    "USTAR": ["friction_velocity"],
    "SWC_1": ["volumetric_soil_water_layer_1"],
    "SWC_2": ["volumetric_soil_water_layer_1"],
    "SWC_3": ["volumetric_soil_water_layer_2"],
    "SWC_4": ["volumetric_soil_water_layer_2"],
    "SWC_5": ["volumetric_soil_water_layer_3"],
    "TS_1": ["soil_temperature_level_1"],
    "TS_2": ["soil_temperature_level_1"],
    "TS_3": ["soil_temperature_level_2"],
    "TS_4": ["soil_temperature_level_2"],
    "TS_5": ["soil_temperature_level_3"],
    "G": [
        "mean_surface_sensible_heat_flux",
        "mean_surface_latent_heat_flux",
        "mean_surface_downward_short_wave_radiation_flux",
        "mean_surface_downward_long_wave_radiation_flux",
        "mean_surface_net_long_wave_radiation_flux",
        "forecast_albedo",
    ],
    "H": ["mean_surface_sensible_heat_flux"],
    "LE": ["mean_surface_latent_heat_flux"],
    "PPFD_IN": ["mean_surface_downward_short_wave_radiation_flux"],
    "PPFD_OUT": ["mean_surface_downward_short_wave_radiation_flux", "forecast_albedo"],
    "WTD": ["wtd"],
    "ELEVATION": ["geopotential"],
}

GRAVITATIONAL_ACC = 9.8
ZERO_C_IN_K = 273.15
DRY_AIR_MOLE_FRACTION_N2 = 0.7808
DRY_AIR_MOLE_FRACTION_O2 = 0.2095
DRY_AIR_MOLE_FRACTION_AR = 0.0093


def kelvin_to_celsius(t_k):
    return t_k - ZERO_C_IN_K


def pa_to_kpa(p_pa):
    return p_pa / 1000.0


def kpa_to_pa(p_kpa):
    return p_kpa * 1000.0


def kpa_to_hpa(p_kpa):
    return p_kpa * 10.0


def wind_speed_magnitude(u10, v10):
    return np.hypot(u10, v10)


def wind_speed_direction(u10, v10):
    return (180.0 + (180.0 / np.pi) * np.arctan2(u10, v10)) % 360.0


def saturated_vapor_pressure(t2m_c):
    a = np.where(t2m_c >= 0, 17.27, 21.875)
    b = np.where(t2m_c >= 0, 237.3, 265.5)
    return 0.61078 * np.exp(a * t2m_c / (t2m_c + b))


def relative_humidity(t2m, d2m):
    t_air_c = kelvin_to_celsius(t2m)
    t_dew_c = kelvin_to_celsius(d2m)
    a, b = 17.625, 243.04
    gamma_air = (a * t_air_c) / (b + t_air_c)
    gamma_dew = (a * t_dew_c) / (b + t_dew_c)
    return 100.0 * np.exp(gamma_dew - gamma_air)


def vapor_pressure_deficit(t2m, d2m):
    rh = relative_humidity(t2m, d2m)
    es_kpa = saturated_vapor_pressure(kelvin_to_celsius(t2m))
    vpd_kpa = es_kpa * (1.0 - (rh / 100.0))
    return kpa_to_hpa(vpd_kpa)


def shortwave_out(avg_sdswrf, fal):
    return avg_sdswrf * fal


def longwave_out(avg_sdlwrf, avg_snlwrf):
    return avg_sdlwrf - avg_snlwrf


def net_radiation(avg_sdswrf, avg_sdlwrf, avg_snlwrf, fal):
    return (
        avg_sdswrf
        + avg_sdlwrf
        - shortwave_out(avg_sdswrf, fal)
        - longwave_out(avg_sdlwrf, avg_snlwrf)
    )


def dry_to_wet_co2_fraction(t2m, d2m, sp, xco2_dry):
    rh = relative_humidity(t2m, d2m)
    t_air_c = kelvin_to_celsius(t2m)
    es_pa = kpa_to_pa(saturated_vapor_pressure(t_air_c))

    xh2o_wet = (rh / 100.0) * es_pa / sp
    xdry_wet = 1.0 - xh2o_wet
    xh2o_dry = xh2o_wet / xdry_wet

    n_tot = (
        DRY_AIR_MOLE_FRACTION_N2
        + DRY_AIR_MOLE_FRACTION_O2
        + DRY_AIR_MOLE_FRACTION_AR
        + xco2_dry / 1e6
        + xh2o_dry
    )
    return xco2_dry / n_tot


def soil_heat_flux(avg_ishf, avg_slhtf, avg_sdswrf, avg_sdlwrf, avg_snlwrf, fal):
    return net_radiation(avg_sdswrf, avg_sdlwrf, avg_snlwrf, fal) + avg_ishf + avg_slhtf


def photosynthesis_photo_flux_density(avg_sdswrf, fal=None):
    if fal is None:
        return 1.741 * avg_sdswrf + 1.45
    return 1.741 * avg_sdswrf * fal + 1.45


PROCESSORS = {
    "RH": relative_humidity,
    "VPD": vapor_pressure_deficit,
    "TA": kelvin_to_celsius,
    "PA": pa_to_kpa,
    "SW_OUT": shortwave_out,
    "LW_OUT": longwave_out,
    "NETRAD": net_radiation,
    "WS": wind_speed_magnitude,
    "WD": wind_speed_direction,
    "G": soil_heat_flux,
    "H": lambda x: -1.0 * x,
    "LE": lambda x: -1.0 * x,
    "TS_1": kelvin_to_celsius,
    "TS_2": kelvin_to_celsius,
    "TS_3": kelvin_to_celsius,
    "TS_4": kelvin_to_celsius,
    "TS_5": kelvin_to_celsius,
    "SWC_1": lambda x: x * 100.0,
    "SWC_2": lambda x: x * 100.0,
    "SWC_3": lambda x: x * 100.0,
    "SWC_4": lambda x: x * 100.0,
    "SWC_5": lambda x: x * 100.0,
    "PPFD_IN": photosynthesis_photo_flux_density,
    "PPFD_OUT": photosynthesis_photo_flux_density,
    "CO2": dry_to_wet_co2_fraction,
    "WTD": lambda x: x,
    "ELEVATION": lambda x: x / GRAVITATIONAL_ACC,
}


@dataclass(frozen=True)
class RequestGroup:
    year: str
    month: str
    days: list[str]
    hours: list[str]
    area: list[float]

    @property
    def stem(self) -> str:
        north, west, south, east = self.area
        day_token = (
            self.days[0]
            if len(self.days) == 1
            else f"{self.days[0]}to{self.days[-1]}"
        )
        hour_token = (
            "all-hours"
            if len(self.hours) == 24
            else f"{self.hours[0].replace(':', '')}to{self.hours[-1].replace(':', '')}"
        )
        area_token = (
            f"N{north:g}_W{west:g}_S{south:g}_E{east:g}"
            .replace("-", "m")
            .replace(".", "p")
        )
        return f"ERA5_{self.year}-{self.month}-{day_token}_{hour_token}_{area_token}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERA5 and build an EcoPerceiver SQLite database."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"ERA5 YAML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Run only the CDS download stage.",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Run only NetCDF to SQLite post-processing.",
    )
    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Overwrite downloaded ZIPs and extracted NetCDF chunk directories.",
    )
    parser.add_argument(
        "--overwrite-db",
        action="store_true",
        help="Recreate the SQLite database even if the config disables recreation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned requests without downloading or writing the database.",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=None,
        help="Limit request/group count for smoke tests.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override download_era5.max_workers for concurrent CDS downloads.",
    )
    args = parser.parse_args()

    if args.download_only and args.process_only:
        parser.error("--download-only and --process-only are mutually exclusive.")
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be at least 1.")
    return args


def ensure_dependencies(*module_names: str) -> None:
    missing = [
        MISSING_DEPENDENCIES[name]
        for name in module_names
        if name in MISSING_DEPENDENCIES
    ]
    if missing:
        raise SystemExit(
            "Missing Python dependencies: "
            f"{', '.join(sorted(set(missing)))}. Install them in the project "
            "environment before running this stage."
        )


def load_config(path: Path) -> dict[str, Any]:
    ensure_dependencies("yaml")
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"Config file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Config must be a YAML mapping: {resolved}")
    return config


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {}) or {}
    if not isinstance(value, dict):
        raise SystemExit(f"Config section {name!r} must be a mapping.")
    return value


def resolve_path(value: str | os.PathLike | None, *, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def configured_time_range(config: dict[str, Any]) -> tuple[datetime, datetime]:
    date_range = configured_date_range(config)
    return date_range.start_datetime, date_range.end_datetime


def full_hours() -> list[str]:
    return [f"{hour:02d}:00" for hour in range(24)]


def month_days(year: int, month: int) -> list[str]:
    days = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, days + 1)]


def iter_month_starts(start: datetime, end: datetime) -> Iterable[datetime]:
    current = datetime(start.year, start.month, 1)
    final = datetime(end.year, end.month, 1)
    while current <= final:
        yield current
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)


def iter_day_starts(start: datetime, end: datetime) -> Iterable[datetime]:
    current = datetime(start.year, start.month, start.day)
    final = datetime(end.year, end.month, end.day)
    while current <= final:
        yield current
        current += timedelta(days=1)


def split_bbox_latitude_bands(
    bbox: list[float],
    latitude_band_degrees: float | int | None,
) -> list[list[float]]:
    north, west, south, east = [float(value) for value in bbox]
    if latitude_band_degrees is None or float(latitude_band_degrees) <= 0:
        return [[north, west, south, east]]

    bands = []
    band_north = north
    step = float(latitude_band_degrees)
    while band_north > south:
        band_south = max(south, band_north - step)
        bands.append([band_north, west, band_south, east])
        band_north = band_south
    return bands


def build_request_groups(config: dict[str, Any]) -> list[RequestGroup]:
    start, end = configured_time_range(config)
    if end < start:
        raise SystemExit(f"Download end date is before start date: {start} > {end}")

    download_config = section(config, "download_era5")
    bbox = download_config.get("bbox", [90, -180, -90, 180])
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise SystemExit("download_era5.bbox must be [north, west, south, east].")
    areas = split_bbox_latitude_bands(
        bbox,
        download_config.get("latitude_band_degrees"),
    )
    chunk = str(download_config.get("temporal_chunk", "daily")).lower()
    hours = full_hours()

    groups: list[RequestGroup] = []
    for month_start in iter_month_starts(start, end):
        month_end = datetime(
            month_start.year,
            month_start.month,
            calendar.monthrange(month_start.year, month_start.month)[1],
            23,
            0,
            0,
        )
        active_start = max(start, month_start)
        active_end = min(end, month_end)
        if active_start > active_end:
            continue

        if chunk == "monthly" and active_start == month_start and active_end >= month_end:
            days = month_days(month_start.year, month_start.month)
            for area in areas:
                groups.append(
                    RequestGroup(
                        str(month_start.year),
                        f"{month_start.month:02d}",
                        days,
                        hours,
                        area,
                    )
                )
            continue

        if chunk not in {"daily", "monthly"}:
            raise SystemExit("download_era5.temporal_chunk must be 'daily' or 'monthly'.")

        for day in iter_day_starts(active_start, active_end):
            day_start = datetime(day.year, day.month, day.day, 0, 0, 0)
            day_end = datetime(day.year, day.month, day.day, 23, 0, 0)
            hour_start = max(active_start, day_start).hour
            hour_end = min(active_end, day_end).hour
            day_hours = [f"{hour:02d}:00" for hour in range(hour_start, hour_end + 1)]
            for area in areas:
                groups.append(
                    RequestGroup(
                        str(day.year),
                        f"{day.month:02d}",
                        [f"{day.day:02d}"],
                        day_hours,
                        area,
                    )
                )
    return groups


def cds_request_payload(config: dict[str, Any], group: RequestGroup) -> dict[str, Any]:
    download_config = section(config, "download_era5")
    product_type = download_config.get("product_type", "reanalysis")
    variables = download_config.get("variables") or ERA5_VARIABLES
    return {
        "product_type": [product_type],
        "variable": variables,
        "year": [group.year],
        "month": [group.month],
        "day": group.days,
        "time": group.hours,
        "area": group.area,
        "data_format": download_config.get("data_format", "netcdf"),
        "download_format": download_config.get("download_format", "zip"),
    }


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def configured_download_workers(config: dict[str, Any], args: argparse.Namespace) -> int:
    download_config = section(config, "download_era5")
    value = args.max_workers
    if value is None:
        value = download_config.get("max_workers", 1)
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit("download_era5.max_workers must be an integer >= 1.") from exc
    if workers < 1:
        raise SystemExit("download_era5.max_workers must be an integer >= 1.")
    return workers


def download_groups(config: dict[str, Any], args: argparse.Namespace) -> None:
    inferred_paths = inferred_pipeline_paths(config, REPO_ROOT)
    download_config = section(config, "download_era5")
    zip_dir = inferred_paths.zip_dir
    netcdf_dir = inferred_paths.netcdf_dir

    groups = build_request_groups(config)
    if args.limit_groups is not None:
        groups = groups[: args.limit_groups]
    overwrite = bool(download_config.get("overwrite", False) or args.overwrite_downloads)
    max_workers = configured_download_workers(config, args)

    print(f"Planned ERA5 CDS request groups: {len(groups)}")
    for index, group in enumerate(groups[:5], start=1):
        print(f"  {index:>3}: {group.stem} area={group.area}")
    if len(groups) > 5:
        print(f"  ... {len(groups) - 5} more")
    print(f"ERA5 CDS download workers: {max_workers}")
    if args.dry_run:
        return

    ensure_dependencies("cdsapi")
    zip_dir.mkdir(parents=True, exist_ok=True)
    netcdf_dir.mkdir(parents=True, exist_ok=True)
    dataset = download_config.get("dataset", "reanalysis-era5-single-levels")
    progress = (
        tqdm(
            total=len(groups),
            desc="ERA5 downloads",
            unit="group",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )
    completed = 0
    skipped = 0
    pending: list[RequestGroup] = []
    log_lock = threading.Lock()
    thread_local = threading.local()

    def log(message: str) -> None:
        with log_lock:
            if progress is not None:
                progress.write(message)
            else:
                print(message)

    def update_progress() -> None:
        if progress is not None:
            progress.set_postfix(done=completed, skipped=skipped, refresh=False)

    def get_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            client = cdsapi.Client(wait_until_complete=True, delete=False)
            thread_local.client = client
        return client

    def download_one(group: RequestGroup) -> None:
        client = get_client()
        group_dir = netcdf_dir / group.stem
        sentinel = group_dir / ".complete"
        zip_path = zip_dir / f"{group.stem}.zip"
        if overwrite and group_dir.exists():
            shutil.rmtree(group_dir)
        if overwrite and zip_path.exists():
            zip_path.unlink()

        payload = cds_request_payload(config, group)
        log(f"[request] {group.stem}")
        result = client.retrieve(dataset, payload)
        log(f"[download] {zip_path}")
        result.download(str(zip_path))
        log(f"[extract] {group_dir}")
        extract_zip(zip_path, group_dir)
        sentinel.write_text("complete\n", encoding="utf-8")
        zip_path.unlink(missing_ok=True)

    try:
        for group in groups:
            group_dir = netcdf_dir / group.stem
            sentinel = group_dir / ".complete"
            if sentinel.exists() and not overwrite:
                skipped += 1
                log(f"[skip] {group.stem}")
                if progress is not None:
                    progress.update(1)
                update_progress()
            else:
                pending.append(group)

        if not pending:
            return

        effective_workers = min(max_workers, len(pending))
        if effective_workers == 1:
            for group in pending:
                if progress is not None:
                    progress.set_postfix_str(group.stem, refresh=False)
                download_one(group)
                completed += 1
                if progress is not None:
                    progress.update(1)
                update_progress()
            return

        pending_iter = iter(pending)
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_group = {}
            for _ in range(effective_workers):
                group = next(pending_iter, None)
                if group is None:
                    break
                future_to_group[executor.submit(download_one, group)] = group

            while future_to_group:
                for future in as_completed(future_to_group):
                    group = future_to_group.pop(future)
                    if progress is not None:
                        progress.set_postfix_str(group.stem, refresh=False)
                    try:
                        future.result()
                    except Exception as exc:
                        raise RuntimeError(f"ERA5 download failed for {group.stem}") from exc
                    completed += 1
                    if progress is not None:
                        progress.update(1)
                    update_progress()

                    next_group = next(pending_iter, None)
                    if next_group is not None:
                        future_to_group[executor.submit(download_one, next_group)] = next_group
                    break
    finally:
        if progress is not None:
            progress.close()


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def init_sqlite(conn: sqlite3.Connection, config: dict[str, Any]) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS coord_data (
            coord_id INTEGER PRIMARY KEY,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            elev REAL,
            igbp TEXT,
            UNIQUE(lat, lon)
        );

        CREATE TABLE IF NOT EXISTS ec_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coord_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            {", ".join(f"{quote_identifier(column)} REAL" for column in EC_PREDICTORS)},
            FOREIGN KEY(coord_id) REFERENCES coord_data(coord_id)
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
    date_range = configured_date_range(config)
    metadata = {
        "generator": "era5_pipeline/download_era5.py",
        "config_metadata": section(config, "metadata"),
        "start_date": date_range.start.isoformat(),
        "end_date": date_range.end.isoformat(),
        "date_range_label": date_range.label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    conn.executemany(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        [(key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()],
    )

def normalize_longitudes(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    normalized = ((arr + 180.0) % 360.0) - 180.0
    normalized[np.isclose(normalized, -180.0)] = 180.0
    return normalized


def rounded_coord_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(float(lat), 6), round(float(lon), 6))


class TimezoneResolver:
    def __init__(self, config: dict[str, Any]):
        timezone_config = section(section(config, "process_era5"), "timezone")
        self.enabled = bool(timezone_config.get("enabled", True))
        self.timestamp_policy = str(timezone_config.get("timestamp_policy", "local")).lower()
        self.method = str(timezone_config.get("method", "timezonefinder")).lower()
        self.fallback = str(timezone_config.get("fallback", "longitude_quarter_hour")).lower()
        self.require_timezonefinder = bool(timezone_config.get("require_timezonefinder", True))
        self.zone_cache: dict[tuple[float, float], str | None] = {}
        self.zone_offset_cache: dict[tuple[str, datetime], int | None] = {}
        self.timezone_finder = None
        if self.enabled and self.method == "timezonefinder":
            if TimezoneFinder is None:
                if self.require_timezonefinder:
                    ensure_dependencies("timezonefinder")
                else:
                    print(
                        "timezonefinder is not installed; using longitude-quarter-hour fallback.",
                        file=sys.stderr,
                    )
            else:
                self.timezone_finder = TimezoneFinder()

    def zone_for_coordinate(self, lat: float, lon: float) -> str | None:
        key = rounded_coord_key(lat, lon)
        if key in self.zone_cache:
            return self.zone_cache[key]
        zone_name = None
        if self.timezone_finder is not None:
            zone_name = self.timezone_finder.timezone_at(lng=lon, lat=lat)
            if zone_name is None:
                zone_name = self.timezone_finder.closest_timezone_at(lng=lon, lat=lat)
        self.zone_cache[key] = zone_name
        return zone_name

    def fallback_offsets_for_longitudes(self, lons: np.ndarray) -> np.ndarray:
        if self.fallback not in {"longitude_quarter_hour", "longitude"}:
            return np.zeros(len(lons), dtype=np.int32)
        minutes = np.asarray(lons, dtype=float) * 4.0
        if self.fallback == "longitude_quarter_hour":
            minutes = np.round(minutes / 15.0) * 15.0
        minutes = np.clip(minutes, -12 * 60, 14 * 60)
        return minutes.astype(np.int32)

    def offset_for_named_zone(self, zone_name: str, utc_dt: datetime) -> int | None:
        utc_key = utc_dt.replace(tzinfo=timezone.utc)
        cache_key = (zone_name, utc_key)
        if cache_key in self.zone_offset_cache:
            return self.zone_offset_cache[cache_key]
        try:
            local_dt = utc_dt.astimezone(ZoneInfo(zone_name))
            offset = local_dt.utcoffset()
            minutes = int(offset.total_seconds() // 60) if offset is not None else None
        except ZoneInfoNotFoundError:
            minutes = None
        self.zone_offset_cache[cache_key] = minutes
        return minutes


class LandSeaMask:
    def __init__(self, config: dict[str, Any]):
        process_config = section(config, "process_era5")
        mask_config = section(process_config, "land_sea_mask")
        path_config = section(config, "paths")
        self.enabled = bool(mask_config.get("enabled", False))
        self.threshold = float(mask_config.get("threshold", 0.5))
        self.include_land_neighbors = bool(mask_config.get("include_land_neighbors", True))
        self.cache: dict[tuple[float, float], bool] = {}
        self.da = None
        path = resolve_path(mask_config.get("path") or path_config.get("lsm_path"))
        if not self.enabled:
            return
        if path is None or not path.exists():
            if mask_config.get("allow_missing", False):
                print(f"Land-sea mask not found; skipping mask: {path}", file=sys.stderr)
                self.enabled = False
                return
            raise SystemExit(f"Land-sea mask file does not exist: {path}")
        ds = xr.open_dataset(
            path,
            engine=mask_config.get("xarray_engine") or process_config.get("xarray_engine"),
        )
        variable = mask_config.get("variable")
        if variable is None:
            variable = "lsm" if "lsm" in ds.data_vars else next(iter(ds.data_vars))
        self.da = ds[variable]
        if "time" in self.da.dims:
            self.da = self.da.isel(time=0)

    def filter(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if not self.enabled or self.da is None or df.empty:
            return df
        coords = df[["lat", "lon"]].drop_duplicates()
        missing = [
            rounded_coord_key(row.lat, row.lon)
            for row in coords.itertuples(index=False)
            if rounded_coord_key(row.lat, row.lon) not in self.cache
        ]
        if missing:
            self.populate_cache(missing)
        keys = pd.MultiIndex.from_frame(df[["lat", "lon"]].round(6))
        keep = keys.map(self.cache).fillna(False).to_numpy(dtype=bool)
        return df.loc[keep].copy()

    def populate_cache(self, keys: list[tuple[float, float]]) -> None:
        lats = np.array(sorted({lat for lat, _ in keys}), dtype=float)
        lons = np.array(sorted({lon for _, lon in keys}), dtype=float)
        mask_lons = self.to_mask_longitudes(lons)
        interp = self.da.interp(latitude=lats, longitude=mask_lons, method="nearest")
        interp = interp.transpose("latitude", "longitude")
        values = np.asarray(interp.values)
        if values.ndim > 2:
            values = np.squeeze(values)
        land = np.nan_to_num(values, nan=0.0) > self.threshold
        keep = land
        if self.include_land_neighbors:
            keep = land | neighbor_any(land)
        for lat_index, lat in enumerate(lats):
            for lon_index, lon in enumerate(lons):
                self.cache[rounded_coord_key(lat, lon)] = bool(keep[lat_index, lon_index])

    def to_mask_longitudes(self, lons: np.ndarray) -> np.ndarray:
        mask_lons = np.asarray(self.da["longitude"].values, dtype=float)
        if np.nanmin(mask_lons) >= 0 and np.nanmax(mask_lons) > 180:
            return (lons + 360.0) % 360.0
        return lons


def neighbor_any(land: np.ndarray) -> np.ndarray:
    padded = np.pad(land.astype(bool), 1, mode="constant", constant_values=False)
    neighbors = np.zeros_like(land, dtype=bool)
    for y_shift in range(3):
        for x_shift in range(3):
            if y_shift == 1 and x_shift == 1:
                continue
            neighbors |= padded[y_shift : y_shift + land.shape[0], x_shift : x_shift + land.shape[1]]
    return neighbors


class Era5DatabaseWriter:
    def __init__(self, conn: sqlite3.Connection, batch_size: int):
        self.conn = conn
        self.batch_size = int(batch_size)
        self.coord_lookup: dict[tuple[float, float], int] = {}
        self.next_coord_id = 1
        self.load_existing_coords()

    def load_existing_coords(self) -> None:
        rows = self.conn.execute("SELECT coord_id, lat, lon FROM coord_data").fetchall()
        for coord_id, lat, lon in rows:
            self.coord_lookup[rounded_coord_key(lat, lon)] = int(coord_id)
            self.next_coord_id = max(self.next_coord_id, int(coord_id) + 1)

    def coord_ids_for_arrays(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        elev: np.ndarray,
        igbp: str | None = None,
    ) -> np.ndarray:
        coord_ids = np.empty(len(lat), dtype=np.int64)
        inserts = []
        for index, (row_lat, row_lon, row_elev) in enumerate(zip(lat, lon, elev, strict=True)):
            key = rounded_coord_key(row_lat, row_lon)
            coord_id = self.coord_lookup.get(key)
            if coord_id is None:
                coord_id = self.next_coord_id
                self.next_coord_id += 1
                self.coord_lookup[key] = coord_id
                inserts.append(
                    (
                        coord_id,
                        float(row_lat),
                        float(row_lon),
                        nullable_float(row_elev),
                        igbp,
                    )
                )
            coord_ids[index] = coord_id
        if inserts:
            self.conn.executemany(
                """
                INSERT INTO coord_data(coord_id, lat, lon, elev, igbp)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(lat, lon) DO NOTHING
                """,
                inserts,
            )
        return coord_ids

    def insert_ec_data(self, df: "pd.DataFrame") -> int:
        for predictor in EC_PREDICTORS:
            if predictor not in df.columns:
                df[predictor] = np.nan
        columns = ["coord_id", "timestamp", *EC_PREDICTORS]
        df[columns].to_sql(
            "ec_data",
            self.conn,
            if_exists="append",
            index=False,
            chunksize=self.batch_size,
        )
        return len(df)


def nullable_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


@dataclass(frozen=True)
class Era5GroupGrid:
    lat: np.ndarray
    lon: np.ndarray
    land_indices: np.ndarray
    spatial_shape: tuple[int, int]
    fallback_offsets: np.ndarray
    timezone_groups: tuple[tuple[str, np.ndarray], ...]
    coord_id: np.ndarray | None = None


@dataclass(frozen=True)
class Era5TimeBlock:
    variables: dict[str, np.ndarray]
    timestamp: np.ndarray
    local_timestamp: np.ndarray
    doy: np.ndarray
    tod: np.ndarray
    length: int


class Era5PostProcessor:
    def __init__(self, config: dict[str, Any], conn: sqlite3.Connection):
        process_config = section(config, "process_era5")
        self.config = config
        self.conn = conn
        self.batch_size = int(process_config.get("batch_size", 50_000))
        time_block_size = process_config.get("time_block_size")
        self.time_block_size = int(time_block_size) if time_block_size else None
        self.insert_hours_per_batch = max(
            1,
            int(process_config.get("insert_hours_per_batch", 4)),
        )
        self.xarray_engine = process_config.get("xarray_engine")
        self.timezone_resolver = TimezoneResolver(config)
        self.land_mask = LandSeaMask(config)
        self.writer = Era5DatabaseWriter(conn, self.batch_size)
        self.local_window = self.parse_local_window()
        self.local_window_int = tuple(
            self.timestamp_bound_to_int(bound) for bound in self.local_window
        )

    def parse_local_window(self) -> tuple["pd.Timestamp | None", "pd.Timestamp | None"]:
        timezone_config = section(section(self.config, "process_era5"), "timezone")
        window_config = section(timezone_config, "local_window")
        if not window_config.get("enabled", False):
            return None, None
        date_range = configured_date_range(self.config)
        start = pd.Timestamp(window_config.get("start") or date_range.local_window_start)
        end = pd.Timestamp(window_config.get("end") or date_range.local_window_end)
        return start, end

    @staticmethod
    def timestamp_bound_to_int(timestamp: "pd.Timestamp | None") -> int | None:
        if timestamp is None:
            return None
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return int(ts.strftime("%Y%m%d%H%M%S"))

    def process_groups(self, netcdf_dir: Path, limit_groups: int | None = None) -> int:
        group_dirs = list(iter_netcdf_group_dirs(netcdf_dir))
        if limit_groups is not None:
            group_dirs = group_dirs[:limit_groups]
        if not group_dirs:
            raise SystemExit(f"No NetCDF groups found under {netcdf_dir}")

        total_rows = 0
        print(f"NetCDF groups to process: {len(group_dirs)}")
        progress = (
            tqdm(
                group_dirs,
                desc="ERA5 processing",
                unit="group",
                dynamic_ncols=True,
                file=sys.stdout,
            )
            if tqdm is not None
            else group_dirs
        )
        for group_dir in progress:
            rows = self.process_group(group_dir)
            total_rows += rows
            if tqdm is not None:
                progress.set_postfix(
                    last_rows=f"{rows:,}",
                    total_rows=f"{total_rows:,}",
                    refresh=False,
                )
                progress.write(f"[inserted] {rows:,} rows from {group_dir}")
            else:
                print(f"[inserted] {rows:,} rows from {group_dir}")
        return total_rows

    def process_group(self, group_dir: Path) -> int:
        paths = sorted(group_dir.glob("*.nc"))
        datasets = [open_dataset(path, self.xarray_engine) for path in paths]
        if not datasets:
            return 0
        try:
            ds = xr.combine_by_coords(datasets, combine_attrs="override") if len(datasets) > 1 else datasets[0]
            ds = standardize_dataset(ds)
            if "valid_time" not in ds.coords:
                raise RuntimeError(f"Dataset has no valid_time coordinate: {group_dir}")
            inserted = 0
            group_grid = self.prepare_group_grid(ds)
            valid_times = pd.to_datetime(ds["valid_time"].values)
            time_block_size = self.infer_time_block_size(ds, len(valid_times))
            pending_frames: list[pd.DataFrame] = []

            def flush_pending_frames() -> None:
                nonlocal inserted
                if not pending_frames:
                    return
                if len(pending_frames) == 1:
                    batch = pending_frames[0]
                else:
                    batch = pd.concat(pending_frames, ignore_index=True, copy=False)
                inserted += self.writer.insert_ec_data(batch)
                pending_frames.clear()

            for block_start in range(0, len(valid_times), time_block_size):
                block_end = min(block_start + time_block_size, len(valid_times))
                block = self.land_block_from_dataset(
                    ds,
                    group_grid,
                    valid_times[block_start:block_end],
                    block_start,
                    block_end,
                )
                if group_grid.coord_id is None:
                    group_grid = self.prepare_group_coord_ids(group_grid, block, valid_times)
                for block_offset in range(block.length):
                    frame = self.land_frame_from_block(group_grid, block, block_offset)
                    frame = self.prepare_land_frame(frame)
                    if frame.empty:
                        continue
                    pending_frames.append(frame)
                    if len(pending_frames) >= self.insert_hours_per_batch:
                        flush_pending_frames()
            flush_pending_frames()
            self.conn.commit()
            return inserted
        finally:
            for ds in datasets:
                ds.close()

    def prepare_group_grid(self, ds) -> Era5GroupGrid:
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            raise RuntimeError("ERA5 dataset is missing latitude/longitude coordinates.")

        lats = np.asarray(ds["latitude"].values, dtype=float)
        lons = normalize_longitudes(np.asarray(ds["longitude"].values, dtype=float))
        lat_flat = np.repeat(lats, len(lons))
        lon_flat = np.tile(lons, len(lats))
        coord_frame = pd.DataFrame({"lat": lat_flat, "lon": lon_flat})
        land_frame = self.land_mask.filter(coord_frame)
        land_indices = land_frame.index.to_numpy(dtype=np.int64)
        land_lat = lat_flat[land_indices]
        land_lon = lon_flat[land_indices]
        return Era5GroupGrid(
            lat=land_lat,
            lon=land_lon,
            land_indices=land_indices,
            spatial_shape=(len(lats), len(lons)),
            fallback_offsets=self.timezone_resolver.fallback_offsets_for_longitudes(land_lon),
            timezone_groups=self.timezone_groups_for_coordinates(land_lat, land_lon),
        )

    def timezone_groups_for_coordinates(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> tuple[tuple[str, np.ndarray], ...]:
        if not self.timezone_resolver.enabled:
            return ()
        grouped: dict[str, list[int]] = {}
        for index, (row_lat, row_lon) in enumerate(zip(lat, lon, strict=True)):
            zone_name = self.timezone_resolver.zone_for_coordinate(float(row_lat), float(row_lon))
            if zone_name is None:
                continue
            grouped.setdefault(zone_name, []).append(index)
        return tuple(
            (zone_name, np.asarray(indices, dtype=np.int64))
            for zone_name, indices in grouped.items()
        )

    def prepare_group_coord_ids(
        self,
        group_grid: Era5GroupGrid,
        block: Era5TimeBlock,
        valid_times: Iterable["pd.Timestamp"],
    ) -> Era5GroupGrid:
        if "geopotential" in block.variables:
            elev = block.variables["geopotential"][0] / GRAVITATIONAL_ACC
        else:
            elev = np.full(len(group_grid.lat), np.nan, dtype=float)
        coord_id = np.full(len(group_grid.lat), -1, dtype=np.int64)
        insert_order = self.coord_insert_order_for_times(group_grid, valid_times)
        ordered_coord_ids = self.writer.coord_ids_for_arrays(
            group_grid.lat[insert_order],
            group_grid.lon[insert_order],
            elev[insert_order],
        )
        coord_id[insert_order] = ordered_coord_ids
        return replace(group_grid, coord_id=coord_id)

    def coord_insert_order_for_times(
        self,
        group_grid: Era5GroupGrid,
        valid_times: Iterable["pd.Timestamp"],
    ) -> np.ndarray:
        point_count = len(group_grid.lat)
        start, end = self.local_window_int
        if start is None and end is None:
            return np.arange(point_count, dtype=np.int64)

        first_seen = np.full(point_count, -1, dtype=np.int32)
        valid_time_list = [pd.Timestamp(valid_time) for valid_time in valid_times]
        for time_index, utc_ts in enumerate(valid_time_list):
            if utc_ts.tzinfo is None:
                utc_ts = utc_ts.tz_localize("UTC")
            else:
                utc_ts = utc_ts.tz_convert("UTC")
            utc_dt = utc_ts.to_pydatetime()
            utc_naive = utc_ts.tz_localize(None)
            offset_minutes = (
                self.offset_minutes_for_group_grid(group_grid, utc_dt)
                if self.timezone_resolver.enabled
                else np.zeros(point_count, dtype=np.int32)
            )
            keep = np.zeros(point_count, dtype=bool)
            for offset in np.unique(offset_minutes):
                local_ts = utc_naive + pd.Timedelta(minutes=int(offset))
                local_timestamp_value = int(local_ts.strftime("%Y%m%d%H%M%S"))
                if start is not None and local_timestamp_value < start:
                    continue
                if end is not None and local_timestamp_value > end:
                    continue
                keep |= offset_minutes == offset
            newly_seen = keep & (first_seen < 0)
            first_seen[newly_seen] = time_index
            if np.all(first_seen >= 0):
                break

        ordered_indices = [
            np.flatnonzero(first_seen == time_index)
            for time_index in range(len(valid_time_list))
        ]
        if not ordered_indices:
            return np.array([], dtype=np.int64)
        return np.concatenate(ordered_indices).astype(np.int64, copy=False)

    def infer_time_block_size(self, ds, valid_time_count: int) -> int:
        if self.time_block_size is not None:
            return max(1, min(int(self.time_block_size), valid_time_count))
        for data_array in ds.data_vars.values():
            if "valid_time" not in data_array.dims:
                continue
            preferred_chunks = data_array.encoding.get("preferred_chunks") or {}
            chunk_size = preferred_chunks.get("valid_time")
            if chunk_size is None:
                chunksizes = data_array.encoding.get("chunksizes")
                if chunksizes:
                    chunk_size = chunksizes[data_array.dims.index("valid_time")]
            if chunk_size:
                return max(1, min(int(chunk_size), valid_time_count))
        return min(24, valid_time_count)

    def land_block_from_dataset(
        self,
        ds,
        group_grid: Era5GroupGrid,
        valid_times: Iterable["pd.Timestamp"],
        block_start: int,
        block_end: int,
    ) -> Era5TimeBlock:
        block_length = block_end - block_start
        variables: dict[str, np.ndarray] = {}
        spatial_dims = {"latitude", "longitude"}
        allowed_dims = {"valid_time", *spatial_dims}

        for name, data_array in ds.data_vars.items():
            if not spatial_dims.issubset(data_array.dims):
                continue
            values = data_array
            if "valid_time" in values.dims:
                values = values.isel(valid_time=slice(block_start, block_end))
            for dim in tuple(values.dims):
                if dim in allowed_dims:
                    continue
                if values.sizes[dim] != 1:
                    raise RuntimeError(
                        f"Cannot convert ERA5 variable {name!r}; unexpected dimension "
                        f"{dim!r} has size {values.sizes[dim]}."
                    )
                values = values.isel({dim: 0}, drop=True)

            if "valid_time" in values.dims:
                arr = np.asarray(values.transpose("valid_time", "latitude", "longitude").values)
                expected_shape = (block_length, *group_grid.spatial_shape)
                if arr.shape != expected_shape:
                    raise RuntimeError(
                        f"ERA5 variable {name!r} has shape {arr.shape}, expected "
                        f"{expected_shape}."
                    )
                variables[name] = arr.reshape(block_length, -1)[:, group_grid.land_indices]
            else:
                arr = np.asarray(values.transpose("latitude", "longitude").values)
                if arr.shape != group_grid.spatial_shape:
                    raise RuntimeError(
                        f"ERA5 variable {name!r} has shape {arr.shape}, expected "
                        f"{group_grid.spatial_shape}."
                    )
                land_values = arr.reshape(-1)[group_grid.land_indices]
                variables[name] = np.broadcast_to(
                    land_values,
                    (block_length, len(group_grid.land_indices)),
                )
        timestamp, local_timestamp, doy, tod = self.time_fields_for_block(
            group_grid,
            valid_times,
        )
        return Era5TimeBlock(
            variables=variables,
            timestamp=timestamp,
            local_timestamp=local_timestamp,
            doy=doy,
            tod=tod,
            length=block_length,
        )

    def time_fields_for_block(
        self,
        group_grid: Era5GroupGrid,
        valid_times: Iterable["pd.Timestamp"],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        valid_time_list = [pd.Timestamp(valid_time) for valid_time in valid_times]
        block_length = len(valid_time_list)
        point_count = len(group_grid.land_indices)
        timestamp = np.empty((block_length, point_count), dtype=np.int64)
        local_timestamp = np.empty((block_length, point_count), dtype=np.int64)
        doy = np.empty((block_length, point_count), dtype=float)
        tod = np.empty((block_length, point_count), dtype=float)

        for time_index, utc_ts in enumerate(valid_time_list):
            if utc_ts.tzinfo is None:
                utc_ts = utc_ts.tz_localize("UTC")
            else:
                utc_ts = utc_ts.tz_convert("UTC")
            utc_dt = utc_ts.to_pydatetime()
            utc_naive = utc_ts.tz_localize(None)
            utc_timestamp = int(utc_naive.strftime("%Y%m%d%H%M%S"))

            if self.timezone_resolver.enabled:
                offset_minutes = self.offset_minutes_for_group_grid(group_grid, utc_dt)
            else:
                offset_minutes = np.zeros(point_count, dtype=np.int32)

            for offset in np.unique(offset_minutes):
                offset_mask = offset_minutes == offset
                local_ts = utc_naive + pd.Timedelta(minutes=int(offset))
                local_timestamp_value = int(local_ts.strftime("%Y%m%d%H%M%S"))
                tod_value = (
                    float(local_ts.hour)
                    + float(local_ts.minute) / 60.0
                    + float(local_ts.second) / 3600.0
                    + 1.0
                )
                if tod_value > 24.0:
                    tod_value -= 24.0
                local_timestamp[time_index, offset_mask] = local_timestamp_value
                doy[time_index, offset_mask] = float(local_ts.dayofyear)
                tod[time_index, offset_mask] = tod_value

            if self.timezone_resolver.timestamp_policy == "local":
                timestamp[time_index] = local_timestamp[time_index]
            else:
                timestamp[time_index].fill(utc_timestamp)

        return timestamp, local_timestamp, doy, tod

    def offset_minutes_for_group_grid(
        self,
        group_grid: Era5GroupGrid,
        utc_dt: datetime,
    ) -> np.ndarray:
        offsets = group_grid.fallback_offsets.copy()
        for zone_name, indices in group_grid.timezone_groups:
            zone_offset = self.timezone_resolver.offset_for_named_zone(zone_name, utc_dt)
            if zone_offset is not None:
                offsets[indices] = zone_offset
        return offsets

    def land_frame_from_block(
        self,
        group_grid: Era5GroupGrid,
        block: Era5TimeBlock,
        block_offset: int,
    ) -> "pd.DataFrame":
        if group_grid.coord_id is None:
            raise RuntimeError("ERA5 group coord_id values were not prepared before frame creation.")
        if block_offset < 0 or block_offset >= block.length:
            raise IndexError(f"Block offset {block_offset} is outside block length {block.length}.")
        data: dict[str, np.ndarray] = {
            "coord_id": group_grid.coord_id,
            "timestamp": block.timestamp[block_offset],
            "_local_timestamp": block.local_timestamp[block_offset],
            "DOY": block.doy[block_offset],
            "TOD": block.tod[block_offset],
        }
        for name, values in block.variables.items():
            data[name] = values[block_offset]
        return pd.DataFrame(data, copy=False)

    def prepare_land_frame(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if df.empty:
            return df
        df = self.apply_local_window(df)
        if df.empty:
            return df
        df = self.add_predictors(df)
        return minmax_normalization(df)

    def add_predictors(self, df: "pd.DataFrame") -> "pd.DataFrame":
        for predictor in [*EC_PREDICTORS, "ELEVATION"]:
            if predictor in {"DOY", "TOD"}:
                continue
            required = VARIABLES_FOR_PREDICTOR.get(predictor)
            if not required or not all(column in df.columns for column in required):
                continue
            values = df[required].to_numpy(dtype=float)
            processor = PROCESSORS.get(predictor)
            if processor is None:
                df[predictor] = values[:, 0]
            else:
                df[predictor] = processor(*[values[:, index] for index in range(values.shape[1])])
        if "ELEVATION" in df.columns:
            df["elev"] = df["ELEVATION"]
        elif "geopotential" in df.columns:
            df["elev"] = df["geopotential"] / GRAVITATIONAL_ACC
        else:
            df["elev"] = np.nan
        return df

    def apply_local_window(self, df: "pd.DataFrame") -> "pd.DataFrame":
        start, end = self.local_window_int
        if start is None and end is None:
            return df
        local_timestamp = df["_local_timestamp"].to_numpy(dtype=np.int64)
        keep = np.ones(len(df), dtype=bool)
        if start is not None:
            keep &= local_timestamp >= start
        if end is not None:
            keep &= local_timestamp <= end
        return df.loc[keep].copy()


def iter_netcdf_group_dirs(netcdf_dir: Path) -> Iterable[Path]:
    paths = sorted(netcdf_dir.rglob("*.nc"))
    seen: set[Path] = set()
    for path in paths:
        parent = path.parent
        if parent in seen:
            continue
        seen.add(parent)
        yield parent


def open_dataset(path: Path, engine: str | None):
    kwargs = {"drop_variables": [name for name in ("number", "expver")]}
    if engine:
        kwargs["engine"] = engine
    try:
        return xr.open_dataset(path, **kwargs)
    except ValueError:
        kwargs.pop("drop_variables", None)
        return xr.open_dataset(path, **kwargs)


def standardize_dataset(ds):
    rename_map = {
        old: new
        for old, new in SHORTNAME_TO_FULLNAME.items()
        if old in ds.data_vars and new not in ds.data_vars
    }
    coord_renames = {}
    if "time" in ds.coords and "valid_time" not in ds.coords:
        coord_renames["time"] = "valid_time"
    if "lat" in ds.coords and "latitude" not in ds.coords:
        coord_renames["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        coord_renames["lon"] = "longitude"
    return ds.rename({**rename_map, **coord_renames})


def minmax_normalization(df: "pd.DataFrame") -> "pd.DataFrame":
    for predictor in EC_PREDICTORS:
        if predictor not in df.columns:
            continue
        vmax = float(DEFAULT_NORM[predictor]["norm_max"])
        vmin = float(DEFAULT_NORM[predictor]["norm_min"])
        vmid = (vmax + vmin) / 2.0
        vrange = vmax - vmin
        cyclic = bool(DEFAULT_NORM[predictor]["cyclic"])
        values = pd.to_numeric(df[predictor], errors="coerce")

        if cyclic:
            period = abs(vrange)
            high = max(vmax, vmin)
            low = min(vmax, vmin)
            values = values.where(values <= high, values - period)
            values = values.where(values >= low, values + period)
            vrange /= 2.0

        low = min(vmin, vmax)
        high = max(vmin, vmax)
        values = values.where(values.between(low, high), np.nan)
        df[predictor] = (values - vmid) / vrange
    return df


def configure_sqlite(conn: sqlite3.Connection, config: dict[str, Any]) -> None:
    process_config = section(config, "process_era5")
    inferred_paths = inferred_pipeline_paths(config, REPO_ROOT)
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute(f"PRAGMA threads = {int(process_config.get('sqlite_threads', 8))}")
    temp_dir = resolve_path(
        process_config.get("sqlite_temp_dir"),
        default=inferred_paths.sqlite_temp_dir,
    )
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            conn.execute(f"PRAGMA temp_store_directory = {str(temp_dir)!r}")
        except sqlite3.OperationalError as exc:
            print(f"Could not set sqlite temp_store_directory: {exc}", file=sys.stderr)

def process_to_database(config: dict[str, Any], args: argparse.Namespace) -> None:
    inferred_paths = inferred_pipeline_paths(config, REPO_ROOT)
    output_dir = inferred_paths.output_dir
    db_path = inferred_paths.db_path
    netcdf_dir = inferred_paths.netcdf_dir

    process_config = section(config, "process_era5")
    recreate_db = bool(process_config.get("recreate_db", True) or args.overwrite_db)
    if args.dry_run:
        groups = list(iter_netcdf_group_dirs(netcdf_dir)) if netcdf_dir.exists() else []
        print(f"Would process {len(groups)} NetCDF group(s) into {db_path}")
        return
    ensure_dependencies("pandas", "xarray")
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and recreate_db:
        db_path.unlink()

    with sqlite3.connect(db_path, timeout=60) as conn:
        configure_sqlite(conn, config)
        init_sqlite(conn, config)
        processor = Era5PostProcessor(config, conn)
        total_rows = processor.process_groups(netcdf_dir, args.limit_groups)
        conn.commit()
    print(f"SQLite export complete: {db_path} ({total_rows:,} ec_data rows inserted)")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    download_enabled = bool(section(config, "download_era5").get("enabled", True))
    if not args.process_only and download_enabled:
        download_groups(config, args)
    elif args.download_only:
        print("Download disabled by config; nothing to do.")

    if not args.download_only:
        process_to_database(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
