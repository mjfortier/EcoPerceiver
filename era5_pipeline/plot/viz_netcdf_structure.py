#!/usr/bin/env python3
"""Print a simple structure view for a NetCDF file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ERA5_CUBE_DIMS = ("valid_time", "latitude", "longitude")
PREFERRED_DIM_ORDER = ("valid_time", "latitude", "longitude", "sample")
DEFAULT_NETCDF_PATH = (
    REPO_ROOT
    / "experiments/runs"
    / "final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC"
    / "seed_0/eval/era5_predictions_2017.nc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print dimensions, variables, groups, and attributes from a NetCDF file."
    )
    parser.add_argument(
        "netcdf",
        nargs="?",
        type=Path,
        default=DEFAULT_NETCDF_PATH,
        help=f"NetCDF file to inspect. Default: {DEFAULT_NETCDF_PATH}",
    )
    parser.add_argument(
        "--attrs",
        action="store_true",
        help="Print global and variable attributes.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "xarray", "netCDF4", "h5py"),
        default="auto",
        help="Reader backend to use. Default: auto, preferring xarray with h5netcdf.",
    )
    parser.add_argument(
        "--xarray-engine",
        default="h5netcdf",
        help="xarray engine to use when the xarray backend is selected. Default: h5netcdf.",
    )
    parser.add_argument(
        "--raw-times",
        action="store_true",
        help="Disable xarray CF time decoding and print raw encoded time values.",
    )
    parser.add_argument(
        "--max-attrs",
        type=int,
        default=12,
        help="Maximum attributes to print per object when --attrs is enabled.",
    )
    parser.add_argument(
        "--max-attr-chars",
        type=int,
        default=160,
        help="Maximum characters to print for each attribute value.",
    )
    return parser.parse_args()


def ordered_names(names: Any, preferred: tuple[str, ...] = PREFERRED_DIM_ORDER) -> list[str]:
    available = list(names)
    ordered = [name for name in preferred if name in available]
    ordered.extend(name for name in available if name not in ordered)
    return ordered


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def compact(value: Any, max_chars: int) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = " ".join(str(value).split())
    if max_chars >= 3 and len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def print_attrs(attrs: dict[str, Any], indent: str, max_attrs: int, max_chars: int) -> None:
    if not attrs:
        print(f"{indent}attrs: none")
        return

    print(f"{indent}attrs:")
    for index, (name, value) in enumerate(attrs.items()):
        if index >= max_attrs:
            print(f"{indent}  ... {len(attrs) - max_attrs} more")
            break
        print(f"{indent}  {name}: {compact(value, max_chars)}")


def open_with_xarray(path: Path, engine: str, decode_times: bool) -> Any:
    import xarray as xr

    return xr.open_dataset(path, engine=engine, decode_times=decode_times)


def dims_text(dims: tuple[str, ...]) -> str:
    return ", ".join(dims) or "scalar"


def is_era5_cube(ds: Any) -> bool:
    return era5_cube_dims(ds) is not None


def era5_cube_dims(ds: Any) -> tuple[str, str, str] | None:
    if all(name in ds.sizes for name in ERA5_CUBE_DIMS):
        return ERA5_CUBE_DIMS
    return None


def print_era5_summary(ds: Any) -> None:
    if "sample" in ds.sizes and not is_era5_cube(ds):
        print("era5 layout: flat sample table")
        return

    if not is_era5_cube(ds):
        print("era5 layout: not a valid_time/latitude/longitude cube")
        return

    cube_dims = era5_cube_dims(ds)
    assert cube_dims is not None
    time_dim, lat_dim, lon_dim = cube_dims
    total_cells = ds.sizes[time_dim] * ds.sizes[lat_dim] * ds.sizes[lon_dim]
    print("era5 layout: cube")
    print(
        "  grid: "
        f"{time_dim}={ds.sizes[time_dim]} {lat_dim}={ds.sizes[lat_dim]} "
        f"{lon_dim}={ds.sizes[lon_dim]} cells={total_cells}"
    )

    prediction_vars = [
        name
        for name, var in ds.data_vars.items()
        if name.startswith("pred_") and var.dims == cube_dims
    ]
    print(
        "  prediction variables: "
        f"{', '.join(prediction_vars) if prediction_vars else 'none'}"
    )

    if "igbp" not in ds.data_vars:
        print("  igbp: missing")
        return

    igbp_dims = ds["igbp"].dims
    if igbp_dims == ("latitude", "longitude"):
        print("  igbp: static per grid cell, dims=(latitude, longitude)")
    elif igbp_dims == cube_dims:
        print(f"  igbp: time-varying, dims=({dims_text(cube_dims)})")
    else:
        print(f"  igbp: unexpected dims=({dims_text(igbp_dims)})")


def print_with_xarray(
    path: Path,
    show_attrs: bool,
    max_attrs: int,
    max_attr_chars: int,
    engine: str = "h5netcdf",
    decode_times: bool = True,
) -> None:
    ds = open_with_xarray(path, engine, decode_times)
    try:
        print(f"backend: xarray")
        print(f"xarray engine: {engine}")
        print_era5_summary(ds)
        print("dimensions:")
        for name in ordered_names(ds.sizes):
            size = ds.sizes[name]
            print(f"  {name}: {size}")

        if ds.coords:
            print("coordinates:")
            for name in ordered_names(ds.coords):
                var = ds.coords[name]
                print(
                    f"  {name}: dims=({dims_text(var.dims)}) "
                    f"shape={tuple(var.shape)} dtype={var.dtype}"
                )
                if show_attrs:
                    print_attrs(dict(var.attrs), "    ", max_attrs, max_attr_chars)
        else:
            print("coordinates: none")

        if ds.data_vars:
            print("data variables:")
            for name in ordered_names(ds.data_vars, ("igbp",)):
                var = ds.data_vars[name]
                print(
                    f"  {name}: dims=({dims_text(var.dims)}) "
                    f"shape={tuple(var.shape)} dtype={var.dtype}"
                )
                if show_attrs:
                    print_attrs(dict(var.attrs), "    ", max_attrs, max_attr_chars)
        else:
            print("data variables: none")

        if show_attrs:
            print("global attributes:")
            print_attrs(dict(ds.attrs), "  ", max_attrs, max_attr_chars)
    finally:
        ds.close()


def netcdf_attrs(obj: Any) -> dict[str, Any]:
    return {name: obj.getncattr(name) for name in obj.ncattrs()}


def scoped_name(group_path: str, name: str) -> str:
    if group_path == "/":
        return name
    return f"{group_path}/{name}"


def print_netcdf_group(
    group: Any,
    group_path: str,
    show_attrs: bool,
    max_attrs: int,
    max_attr_chars: int,
) -> None:
    print(f"group: {group_path}")

    if group.dimensions:
        print("  dimensions:")
        for name in ordered_names(group.dimensions):
            dim = group.dimensions[name]
            size = "unlimited" if dim.isunlimited() else len(dim)
            print(f"    {scoped_name(group_path, name)}: {size}")
    else:
        print("  dimensions: none")

    if group.variables:
        print("  variables:")
        for name in ordered_names(group.variables):
            var = group.variables[name]
            dims = ", ".join(scoped_name(group_path, dim) for dim in var.dimensions) or "scalar"
            print(
                f"    {scoped_name(group_path, name)}: "
                f"dims=({dims}) shape={tuple(var.shape)} dtype={var.dtype}"
            )
            if show_attrs:
                print_attrs(netcdf_attrs(var), "      ", max_attrs, max_attr_chars)
    else:
        print("  variables: none")

    if show_attrs:
        print("  group attributes:")
        print_attrs(netcdf_attrs(group), "    ", max_attrs, max_attr_chars)

    for name, child in group.groups.items():
        print_netcdf_group(child, scoped_name(group_path, name), show_attrs, max_attrs, max_attr_chars)


def print_with_netcdf4(path: Path, show_attrs: bool, max_attrs: int, max_attr_chars: int) -> None:
    from netCDF4 import Dataset

    with Dataset(path, mode="r") as ds:
        print("backend: netCDF4")
        print_netcdf_group(ds, "/", show_attrs, max_attrs, max_attr_chars)


def h5py_attrs(obj: Any) -> dict[str, Any]:
    return {str(name): value for name, value in obj.attrs.items()}


def print_h5py_node(
    name: str,
    node: Any,
    show_attrs: bool,
    max_attrs: int,
    max_attr_chars: int,
) -> None:
    import h5py

    if isinstance(node, h5py.Dataset):
        print(f"dataset: {name} shape={node.shape} dtype={node.dtype}")
        if show_attrs:
            print_attrs(h5py_attrs(node), "  ", max_attrs, max_attr_chars)
    elif isinstance(node, h5py.Group):
        print(f"group: {name}")
        if show_attrs:
            print_attrs(h5py_attrs(node), "  ", max_attrs, max_attr_chars)


def print_with_h5py(path: Path, show_attrs: bool, max_attrs: int, max_attr_chars: int) -> None:
    import h5py

    with h5py.File(path, mode="r") as h5:
        print("backend: h5py")
        print("root:")
        if show_attrs:
            print_attrs(h5py_attrs(h5), "  ", max_attrs, max_attr_chars)
        h5.visititems(lambda name, node: print_h5py_node(name, node, show_attrs, max_attrs, max_attr_chars))


def print_structure(
    path: Path,
    show_attrs: bool,
    max_attrs: int,
    max_attr_chars: int,
    backend: str,
    xarray_engine: str,
    decode_times: bool,
) -> None:
    errors: list[str] = []
    readers = []
    if backend in {"auto", "xarray"}:
        readers.append(
            (
                "xarray",
                lambda: print_with_xarray(
                    path,
                    show_attrs,
                    max_attrs,
                    max_attr_chars,
                    engine=xarray_engine,
                    decode_times=decode_times,
                ),
            )
        )
    if backend in {"auto", "netCDF4"}:
        readers.append(
            (
                "netCDF4",
                lambda: print_with_netcdf4(path, show_attrs, max_attrs, max_attr_chars),
            )
        )
    if backend in {"auto", "h5py"}:
        readers.append(
            (
                "h5py",
                lambda: print_with_h5py(path, show_attrs, max_attrs, max_attr_chars),
            )
        )

    for name, reader in readers:
        try:
            reader()
            return
        except ImportError as exc:
            errors.append(f"{name}: import failed: {exc}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    details = "\n".join(f"  - {error}" for error in errors)
    raise RuntimeError(
        f"Could not inspect the file with backend={backend}.\n"
        "Install xarray+h5netcdf, netCDF4, or h5py in the active environment, "
        "then rerun this script.\n"
        f"Tried:\n{details}"
    )


def main() -> int:
    args = parse_args()
    path = args.netcdf.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    if args.max_attrs < 0:
        raise ValueError("--max-attrs must be non-negative")
    if args.max_attr_chars < 1:
        raise ValueError("--max-attr-chars must be positive")

    print(f"file: {path}", flush=True)
    print(f"size: {human_size(path.stat().st_size)}", flush=True)
    print_structure(
        path,
        args.attrs,
        args.max_attrs,
        args.max_attr_chars,
        args.backend,
        args.xarray_engine,
        decode_times=not args.raw_times,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
