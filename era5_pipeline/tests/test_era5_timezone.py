import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import xarray as xr
except ModuleNotFoundError:
    pd = None
    xr = None


REPO_ROOT = Path(__file__).resolve().parents[2]
ERA5_PIPELINE_DIR = REPO_ROOT / "era5_pipeline"
sys.path.insert(0, str(ERA5_PIPELINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import download_era5


HALF_HOUR_COORDS = {
    "india": (22.0, 78.0),
    "australia_darwin": (-12.46, 130.84),
    "newfoundland": (48.5, -53.5),
}


def make_config(root: Path, *, timestamp_grid: str = "hourly") -> dict:
    return {
        "start_date": "2017-01-03",
        "end_date": "2017-01-03",
        "paths": {
            "output_dir": str(root),
            "db_path": str(root / "era5_timezone_test.db"),
            "netcdf_dir": str(root / "era5_data"),
        },
        "process_era5": {
            "recreate_db": True,
            "batch_size": 1_000,
            "insert_hours_per_batch": 4,
            "xarray_engine": "h5netcdf",
            "timezone": {
                "enabled": True,
                "timestamp_policy": "local",
                "timestamp_grid": timestamp_grid,
                "method": "timezonefinder",
                "require_timezonefinder": True,
                "fallback": "longitude_quarter_hour",
                "local_window": {"enabled": True},
            },
            "land_sea_mask": {"enabled": False},
        },
    }


@unittest.skipIf(
    pd is None or xr is None or download_era5.TimezoneFinder is None,
    "pandas, xarray, and timezonefinder are required",
)
class Era5TimezoneGridTest(unittest.TestCase):
    def test_half_hour_zones_use_hourly_timestamp_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config(root)
            grid = self.make_grid(config)
            times = [
                pd.Timestamp("2017-01-03T03:00:00"),  # India 08:30 -> 09:00
                pd.Timestamp("2017-01-02T23:00:00"),  # Darwin 08:30 -> 09:00
                pd.Timestamp("2017-01-03T12:00:00"),  # Newfoundland 08:30 -> 09:00
            ]

            timestamp, _, _, tod = self.make_processor(config).time_fields_for_block(grid, times)

            self.assertEqual(timestamp[0, 0], 20170103090000)
            self.assertEqual(timestamp[1, 1], 20170103090000)
            self.assertEqual(timestamp[2, 2], 20170103090000)
            self.assertAlmostEqual(tod[0, 0], 9.5)

    def test_exact_timestamp_grid_preserves_minute_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config(root, timestamp_grid="exact")
            grid = self.make_grid(config)

            timestamp, _, _, _ = self.make_processor(config).time_fields_for_block(
                grid,
                [pd.Timestamp("2017-01-03T03:00:00")],
            )

            self.assertEqual(timestamp[0, 0], 20170103083000)

    def test_tiny_database_has_half_hour_zone_rows_at_0900(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config(root)
            self.write_tiny_netcdf(root / "era5_data" / "smoke")
            db_path = root / "era5_timezone_test.db"

            with sqlite3.connect(db_path) as conn:
                download_era5.init_sqlite(conn, config)
                processor = download_era5.Era5PostProcessor(config, conn)
                processor.process_groups(root / "era5_data")
                conn.commit()

                minute_timestamp_rows = conn.execute(
                    "SELECT COUNT(*) FROM ec_data WHERE timestamp % 10000 != 0"
                ).fetchone()[0]
                self.assertEqual(minute_timestamp_rows, 0)

                for lat, lon in HALF_HOUR_COORDS.values():
                    rows = conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM ec_data ec
                        JOIN coord_data coord ON coord.coord_id = ec.coord_id
                        WHERE ABS(coord.lat - ?) < 1e-6
                          AND ABS(coord.lon - ?) < 1e-6
                          AND ec.timestamp = 20170103090000;
                        """,
                        (lat, lon),
                    ).fetchone()[0]
                    self.assertEqual(rows, 1, (lat, lon))

    def make_processor(self, config: dict) -> download_era5.Era5PostProcessor:
        conn = sqlite3.connect(":memory:")
        download_era5.init_sqlite(conn, config)
        self.addCleanup(conn.close)
        return download_era5.Era5PostProcessor(config, conn)

    def make_grid(self, config: dict) -> download_era5.Era5GroupGrid:
        lat = np.asarray([coord[0] for coord in HALF_HOUR_COORDS.values()], dtype=float)
        lon = np.asarray([coord[1] for coord in HALF_HOUR_COORDS.values()], dtype=float)
        processor = self.make_processor(config)
        return download_era5.Era5GroupGrid(
            lat=lat,
            lon=lon,
            land_indices=np.arange(len(lat), dtype=np.int64),
            spatial_shape=(len(lat), 1),
            fallback_offsets=processor.timezone_resolver.fallback_offsets_for_longitudes(lon),
            timezone_groups=processor.timezone_groups_for_coordinates(lat, lon),
            coord_id=np.arange(1, len(lat) + 1, dtype=np.int64),
        )

    def write_tiny_netcdf(self, group_dir: Path) -> None:
        group_dir.mkdir(parents=True, exist_ok=True)
        lats = np.asarray([22.0, -12.46, 48.5, 0.0], dtype=float)
        lons = np.asarray([78.0, 130.84, -53.5, 0.0], dtype=float)
        valid_time = pd.date_range("2017-01-02T22:00:00", "2017-01-03T12:00:00", freq="h")
        shape = (len(valid_time), len(lats), len(lons))
        values = np.ones(shape, dtype=np.float32)
        ds = xr.Dataset(
            {
                "geopotential": (("valid_time", "latitude", "longitude"), values),
            },
            coords={
                "valid_time": valid_time,
                "latitude": lats,
                "longitude": lons,
            },
        )
        ds.to_netcdf(group_dir / "tiny.nc", engine="h5netcdf")


if __name__ == "__main__":
    unittest.main()
