# The code below is intended for use with EcoPerceiver. The purpose of creating this database is to facilitate the
# inference process by maintaining consistency with the input format used in EcoPerceiver’s training phase.

import os
import glob
import sqlite3
import requests
import rioxarray as rxr
import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import convolve

EC_PREDICTORS = ('DOY', 'TOD', 'TA', 'P', 'RH', 'VPD', 'PA', 'CO2', 'SW_IN', 'SW_IN_POT', 'SW_OUT', 'LW_IN', 'LW_OUT',
                 'NETRAD', 'PPFD_IN', 'PPFD_OUT', 'WS', 'WD', 'USTAR', 'SWC_1', 'SWC_2', 'SWC_3', 'SWC_4', 'SWC_5',
                 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'TS_5', 'WTD', 'G', 'H')

EC_TARGETS = ('NEE', 'GPP_DT', 'GPP_NT', 'RECO_DT', 'RECO_NT', 'FCH4', 'LE')

GEO_PREDICTORS = ('lat', 'lon', 'elev')

IGBP_CODES = ('ENF', 'MF', 'WET', 'CRO', 'GRA', 'WAT', 'SAV', 'DBF', 'CSH', 'OSH', 'EBF', 'WSA', 'BSV', 'URB',
              'DNF', 'CVM', 'SNO')

IGBP_ACRONYMS = {
    0: 'WAT', 1: 'ENF', 2: 'EBF', 3: 'DNF', 4: 'DBF', 5: 'MF', 6: 'CSH',
    7: 'OSH', 8: 'WSA', 9: 'SAV', 10: 'GRA', 11: 'WET', 12: 'CRO',
    13: None, 14: 'CVM', 15: 'SNO', 16: None,
}

DEFAULT_NORM = {
    'DOY': {'cyclic': True, 'norm_max': 366.0, 'norm_min': 0.0},
    'TOD': {'cyclic': True, 'norm_max': 24.0, 'norm_min': 0.0},
    'TA': {'cyclic': False, 'norm_max': 80.0, 'norm_min': -80.0},
    'P': {'cyclic': False, 'norm_max': 100.0, 'norm_min': 0.0},
    'RH': {'cyclic': False, 'norm_max': 100.0, 'norm_min': 0.0},
    'VPD': {'cyclic': False, 'norm_max': 110.0, 'norm_min': 0.0},
    'PA': {'cyclic': False, 'norm_max': 110.0, 'norm_min': 0.0},
    'CO2': {'cyclic': False, 'norm_max': 750.0, 'norm_min': 0.0},
    'SW_IN': {'cyclic': False, 'norm_max': 1500.0, 'norm_min': -1500.0},
    'SW_IN_POT': {'cyclic': False, 'norm_max': 1500.0, 'norm_min': -1500.0},
    'SW_OUT': {'cyclic': False, 'norm_max': 500.0, 'norm_min': -500.0},
    'LW_IN': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'LW_OUT': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'NETRAD': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'PPFD_IN': {'cyclic': False, 'norm_max': 2500.0, 'norm_min': -2500.0},
    'PPFD_OUT': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'WS': {'cyclic': False, 'norm_max': 100.0, 'norm_min': -100.0},
    'WD': {'cyclic': True, 'norm_max': 360.0, 'norm_min': 0.0},
    'USTAR': {'cyclic': False, 'norm_max': 4.0, 'norm_min': -4.0},
    'SWC_1': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_2': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_3': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_4': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_5': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'TS_1': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_2': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_3': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_4': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_5': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'WTD': {'cyclic': False, 'norm_max': -3.0, 'norm_min': 3.0},
    'G': {'cyclic': False, 'norm_max': 700.0, 'norm_min': -700.0},
    'H': {'cyclic': False, 'norm_max': 700.0, 'norm_min': -700.0},
    'LE': {'cyclic': False, 'norm_max': 700.0, 'norm_min': -700.0},
}

# ======================== Paths and Table Names ========================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'experiments/data'
ERA5_DATA_PATH = DATA_PATH / 'era5_data'
DB_SCHEMA_PATH = DATA_PATH / 'carbonpipeline_db_struct.sql'
IGBP_PATH = DATA_PATH / 'igbp.tiff'
DB_PATH = DATA_PATH / 'era5.db'
DATA_TABLE_NAME = 'ec_data'
COORD_TABLE_NAME = 'coord_data'

LSM_PATH = DATA_PATH / 'lsm.nc'
LSM_URL = 'https://confluence.ecmwf.int/download/attachments/140385202/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc?version=1&modificationDate=1591983422208&api=v2'


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
    query_land_sea_mask()

    for path in glob.glob(os.path.join(ERA5_DATA_PATH, '*.nc')):
        ds = xr.open_dataset(path, engine="netcdf4")
        df = (ds.to_dataframe()
              .reset_index()
              .drop(columns=['region_id'])
              .rename(columns={
                'latitude': 'lat',
                'longitude': 'lon',
                'valid_time': 'timestamp',
                'ELEVATION': 'elev',
              })
            )
        
        df = land_sea_mask(df)
        df = add_doy_and_tod(df)
        df = add_igbp(df)
        df = minmax_normalization(df)
        df['timestamp'] = df['timestamp'].apply(format_datetime)

        wanted = ['lat', 'lon', 'elev', 'igbp']
        existing = [c for c in wanted if c in df.columns]

        coords = df[existing].drop_duplicates().reset_index(drop=True)
        coords['coord_id'] = coords.index + 1

        df = df.merge(coords, on=existing, how='left')

        existing_predictors = [p for p in EC_PREDICTORS if p in df.columns]
        with sqlite3.connect(DB_PATH) as conn:
            coords.to_sql(COORD_TABLE_NAME, conn, if_exists='replace', index=False)
            df[[ 'coord_id', 'timestamp', *existing_predictors ]].to_sql(DATA_TABLE_NAME, conn, if_exists='append', index=False)
            conn.commit()

        print(f'Inserted {len(df)} rows from {os.path.basename(path)}')



# ======================== Helpers ========================

def query_land_sea_mask():
    try:
        r = requests.get(LSM_URL, stream=True)
        r.raise_for_status()

        with open(LSM_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1): 
                if chunk:
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {LSM_URL}: {e}")


def land_sea_mask(df: pd.DataFrame) -> pd.DataFrame:
    ds_lsm = xr.open_dataset(LSM_PATH, engine="netcdf4")

    target_lats = np.arange(-90, 90.25, 0.25)
    target_lons = np.arange(0, 360, 0.25)

    mask_interp = ds_lsm.interp(
        latitude=target_lats,
        longitude=target_lons,
        method="linear"
    )

    mask_025 = mask_interp.to_dataframe().reset_index()
    mask_025['longitude'] = ((mask_025['longitude'] + 180) % 360) - 180

    df_merged = df.merge(
        mask_025[['latitude','longitude','lsm']],
        left_on=['lat','lon'],
        right_on=['latitude','longitude'],
        how='left'
    )

    df_spatial = df_merged[['lat','lon','lsm']].drop_duplicates()

    # 2D grid.head()
    grid = df_spatial.pivot(index="lat", columns="lon", values="lsm")

    # binary mask
    land = (grid > 0.5).astype(int)

    conv_kernel = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    land_neighbours = convolve(
        land.values,
        conv_kernel,
        mode='constant',
        cval=0.0
    )

    land_neighbours_df = pd.DataFrame(
        land_neighbours,
        index=land.index,
        columns=land.columns,
    )

    keep = (land == 1) | (land_neighbours_df > 0)
    keep_flat = keep.stack().reset_index()

    assert keep_flat.shape[0] == (keep.shape[0] * keep.shape[1])

    keep_flat.columns = ['lat', 'lon', 'keep']
    df_filtered = df_spatial.merge(keep_flat, on=['lat', 'lon'])
    df_filtered = df_filtered[df_filtered['keep']]

    return df_merged.merge(df_filtered, on=['lat', 'lon'])


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
        if pred not in df.columns:
            continue

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
