# The code below is intended for use with EcoPerceiver. The purpose of creating this database is to facilitate the
# inference process by maintaining consistency with the input format used in EcoPerceiver’s training phase.

import os
import glob
import sqlite3
import rioxarray as rxr
import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.spatial import cKDTree
from ecoperceiver.constants import EC_PREDICTORS, IGBP_ACRONYMS, DEFAULT_NORM


# ======================== Paths and Table Names ========================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'experiments/data'
ERA5_DATA_PATH = DATA_PATH / 'era5_data'
DB_SCHEMA_PATH = DATA_PATH / 'carbonpipeline_db_struct.sql'
IGBP_PATH = DATA_PATH / 'igbp.tiff'
DB_PATH = DATA_PATH / 'era5.db'
DATA_TABLE_NAME = 'ec_data'
COORD_TABLE_NAME = 'coord_data'



# ======================== Data Transfer ========================

def launch_sqlite():
    with open(DB_SCHEMA_PATH, 'r') as f:
        schema = f.read()

    sql = schema.format(
        table_name=DATA_TABLE_NAME,
        vars=', '.join([f'{c} REAL' for c in EC_PREDICTORS])
    )

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(sql)
    conn.commit()
    conn.close()

    print('\n~ The SQLite database has been initialized ~')


def port_netcdf_to_database():
    for path in glob.glob(os.path.join(ERA5_DATA_PATH, '*.nc')):
        ds = xr.open_dataset(path, engine="netcdf4")
        df = (ds.to_dataframe()
              .reset_index()
              .drop(columns=['region_id'])
              .rename(columns={
                'latitude': 'lat',
                'longitude': 'lon',
                'valid_time': 'timestamp',
                'LOCATION_ELEV': 'elev',
              })
            )
        
        df = add_doy_and_tod(df)
        df = add_igbp(df)
        df = minmax_normalization(df)
        df['timestamp'] = df['timestamp'].apply(format_datetime)

        coords = df[['lat', 'lon', 'elev', 'igbp']].drop_duplicates().reset_index(drop=True)
        coords['coord_id'] = coords.index + 1

        df = df.merge(coords, on=['lat', 'lon', 'elev', 'igbp'], how='left')

        with sqlite3.connect(DB_PATH) as conn:
            coords.to_sql(COORD_TABLE_NAME, conn, if_exists='replace', index=False)
            df[[ 'coord_id', 'timestamp', *EC_PREDICTORS ]].to_sql(DATA_TABLE_NAME, conn, if_exists='append', index=False)
            conn.commit()

        print(f'Inserted {len(df)} rows from {os.path.basename(path)}')



# ======================== Helpers ========================

def add_doy_and_tod(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['DOY'] = df['timestamp'].apply(lambda dt: dt.dayofyear)
    df_copy['TOD'] = df['timestamp'].apply(lambda dt: dt.hour + 1)
    return df_copy


def add_igbp(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    da = rxr.open_rasterio(IGBP_PATH).isel(band=0)
    df_igbp = pd.DataFrame({
        'x': np.repeat(da.x.values, len(da.y)),
        'y': np.tile(da.y.values, len(da.x)),
        'val': da.values.flatten()
    })

    tree = cKDTree(df_igbp[['x', 'y']])

    _, idx = tree.query(df_copy[['lon', 'lat']], k=1)
    igbp_vals = df_igbp.iloc[idx]['val'].values

    df_copy['igbp'] = [IGBP_ACRONYMS.get(v) for v in igbp_vals]
    return df_copy


def minmax_normalization(df: pd.DataFrame) -> pd.DataFrame:
    for pred in EC_PREDICTORS:
        vmax = DEFAULT_NORM[pred]['norm_max']
        vmin = DEFAULT_NORM[pred]['norm_min']
        vmid = (vmax + vmin) / 2
        vrange = vmax - vmin
        cyclic = DEFAULT_NORM[pred]['cyclic']

        if cyclic:
            vrange /= 2

        df.loc[~df[pred].between(vmin, vmax), pred] = np.nan
        df[pred] = (df[pred] - vmid) / vrange
    return df


def format_datetime(dt: pd.Timestamp) -> int:
    return int(dt.strftime('%Y%m%d%H%M%S'))


# ======================== Entry Point ========================

if __name__ == '__main__':
    launch_sqlite()
    port_netcdf_to_database()
