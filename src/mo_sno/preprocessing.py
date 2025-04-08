import os
import time
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
import plotly.io as pio
import geopandas as gpd
import xarray as xr
import requests
from tqdm import tqdm
import yaml

warnings.filterwarnings("ignore")
pio.renderers.default = 'colab'

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
CONFIG_PATH = os.path.join(ROOT_DIR, 'w210_snow_intro/config', 'preprocessing.yml')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

VARIABLES = ['prcp', 'tmin', 'tmax', 'dayl', 'vp']
LAT_COL = 'lat'
LON_COL = 'lon'

BASE_DIR = CONFIG.get("daymet_path")

def fetch_elevations(locations, batch_size=100):
    elevations = []
    for i in tqdm(range(0, len(locations), batch_size), desc="Fetching elevation data in batches"):
        batch = locations[i: i + batch_size]
        try:
            response = requests.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": batch})
            if response.status_code == 200:
                results = response.json().get("results", [])
                elevations.extend([result["elevation"] for result in results])
            else:
                print("Error fetching elevation data:", response.status_code)
                elevations.extend([None] * len(batch))
        except Exception as e:
            print("Exception during elevation lookup:", e)
            elevations.extend([None] * len(batch))
    return elevations

def add_elevation(insitu_gdf):
    insitu_gdf['z'] = None
    bgp_mask = insitu_gdf['BGP'].notna()
    insitu_gdf.loc[bgp_mask, 'z'] = 9650 * 0.3048  # ft to meters

    other_mask = ~bgp_mask
    other = insitu_gdf[other_mask]
    if not other.empty:
        locations = [{"latitude": row.y, "longitude": row.x} for _, row in other.iterrows()]
        elevations = fetch_elevations(locations, batch_size=100)
        insitu_gdf.loc[other_mask, 'z'] = elevations
    return insitu_gdf

def load_insitu_data():
    df = pd.read_parquet(CONFIG["insitu_path"])
    df = df.reset_index()[['time', 'y', 'x'] + CONFIG["sensors"]]
    df['time'] = pd.to_datetime(df['time'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=CONFIG["coord_crs"])
    return add_elevation(gdf)

def get_intermediate_insitu_data(save_path, overwrite=False):
    if os.path.exists(save_path) and not overwrite:
        print(f"Loading intermediate insitu data from {save_path}")
        df = pd.read_parquet(save_path)
        return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df.geometry), crs=CONFIG["coord_crs"])
    else:
        print("Processing raw insitu data to add elevation...")
        gdf = load_insitu_data()
        gdf.to_parquet(save_path, index=False)
        return gdf

def process_basin(insitu_gdf, basin_name):
    basin = gpd.read_file(CONFIG["basins"][basin_name]["shapefile"]).to_crs(CONFIG["working_crs"])
    basin_geom = basin.geometry.unary_union.buffer(CONFIG["buffer_meters"])

    points = insitu_gdf.to_crs(CONFIG["working_crs"])
    spatial_index = points.sindex
    bbox = basin_geom.bounds
    candidates = list(spatial_index.intersection(bbox))
    prelim_filter = points.iloc[candidates]
    mask = prelim_filter.within(basin_geom)
    filtered = prelim_filter[mask].to_crs(CONFIG["coord_crs"])
    filtered = filtered.set_index('time').sort_index()

    sensor_cols = CONFIG["sensors"]
    daily = filtered[sensor_cols].resample('D').mean()
    daily['z'] = filtered[['z']].resample('D').first()['z']
    daily['pillow_swe'] = daily[sensor_cols].mean(axis=1, skipna=True)
    daily['pillows_used'] = filtered[sensor_cols].resample('D').apply(lambda x: x.columns[x.notna().any()].tolist())

    aso_ds = xr.open_dataset(CONFIG["basins"][basin_name]["aso"], engine='netcdf4')
    aso_swe = aso_ds['aso_swe'].mean(dim=['x', 'y']) * 1000
    aso_df = aso_swe.to_series().to_frame('aso_swe').resample('D').mean()

    sampled_files = [os.path.join(BASE_DIR, var, f"sampled_{var}.parquet") for var in VARIABLES]
    print(f"Using sampled Daymet files: {sampled_files}")
    basin_daymet_crs = basin.to_crs("EPSG:4326")
    basin_geom_daymet = basin_daymet_crs.geometry.unary_union.buffer(0.1)

    def process_sampled_file(f):
        try:
            var = os.path.basename(f).split('_')[1].split('.')[0]
            df = pd.read_parquet(f)
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[LON_COL], df[LAT_COL]),
                crs="EPSG:4326"
            )
            mask = gdf.geometry.within(basin_geom_daymet)
            filtered = df[mask.values].dropna(subset=['value'])
            filtered['variable_type'] = var
            return filtered if not filtered.empty else None
        except Exception as e:
            print(f"Error processing {os.path.basename(f)}: {str(e)}")
            return None

    daymet_dfs = [process_sampled_file(f) for f in tqdm(sampled_files, desc="Processing Daymet files")]
    daymet_dfs = [df for df in daymet_dfs if df is not None]
    if not daymet_dfs:
        raise ValueError(f"No valid Daymet data found for {basin_name}")
    daymet_df = (pd.concat(daymet_dfs)
                 .pivot_table(index='time', columns='variable_type', values='value')
                 .resample('D').mean()
                 .add_prefix('daymet_'))

    merged = daily[['pillow_swe', 'pillows_used', 'z']].merge(
        aso_df, left_index=True, right_index=True, how='outer'
    ).merge(daymet_df, left_index=True, right_index=True, how='left')

    cols_to_interpolate = merged.columns.difference(['aso_swe'])
    merged[cols_to_interpolate] = merged[cols_to_interpolate].interpolate()

    merged['pillow_swe_corrected'] = merged['pillow_swe']
    mask = merged['aso_swe'].notna()
    merged.loc[mask, 'pillow_swe_corrected'] = merged.loc[mask, 'aso_swe']
    return merged

def get_preprocessed_dataset(basin_name, insitu_gdf, save_path, overwrite=False):
    if os.path.exists(save_path) and not overwrite:
        print(f"Loading preprocessed dataset for {basin_name} from {save_path}")
        return pd.read_parquet(save_path)
    else:
        print(f"Processing basin {basin_name} and saving to {save_path}")
        df = process_basin(insitu_gdf, basin_name)
        df.to_parquet(save_path, index=True)
        return df
