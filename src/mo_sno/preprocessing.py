"""
Preprocessing Module
~~~~~~~~~~~~~~~~~~~~

This module provides functions for loading and processing insitu data,
adding elevation, and preparing basin‐specific datasets.
"""

import os
import pandas as pd
import cudf
import geopandas as gpd
import xarray as xr
import requests
from tqdm.notebook import tqdm
import yaml

# Load configuration from YAML file.
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'preprocessing.yml')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)


def fetch_elevations(locations, batch_size=100):
    """
    Fetch elevation values for a list of locations in batches.

    :param locations: List of dictionaries with keys 'latitude' and 'longitude'.
    :type locations: list
    :param batch_size: Number of locations per API request.
    :type batch_size: int
    :return: List of elevation values.
    :rtype: list
    """
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
    """
    Adds an elevation column ('z') to the insitu GeoDataFrame. Rows with a non-null 'BGP'
    sensor are set to 9650 ft (converted to meters); the remaining rows are looked up via API.

    :param insitu_gdf: GeoDataFrame containing insitu sensor data.
    :type insitu_gdf: geopandas.GeoDataFrame
    :return: GeoDataFrame with an added elevation column.
    :rtype: geopandas.GeoDataFrame
    """
    insitu_gdf['z'] = None
    bgp_mask = insitu_gdf['BGP'].notna()
    insitu_gdf.loc[bgp_mask, 'z'] = 9650 * 0.3048  # Convert ft to meters
    other_mask = ~bgp_mask
    other = insitu_gdf[other_mask]
    if not other.empty:
        locations = [{"latitude": row.y, "longitude": row.x} for _, row in other.iterrows()]
        elevations = fetch_elevations(locations, batch_size=100)
        insitu_gdf.loc[other_mask, 'z'] = elevations
    return insitu_gdf


def load_insitu_data():
    """
    Loads raw insitu data, converts it to a GeoDataFrame, and adds elevation data.

    :return: Processed insitu GeoDataFrame.
    :rtype: geopandas.GeoDataFrame
    """
    df = pd.read_parquet(CONFIG["insitu_path"])
    df = df.reset_index()[['time', 'y', 'x'] + CONFIG["sensors"]]
    df['time'] = pd.to_datetime(df['time'])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.x, df.y),
        crs=CONFIG["coord_crs"]
    )
    gdf = add_elevation(gdf)
    return gdf


def get_intermediate_insitu_data(save_path, overwrite=False):
    """
    Loads an intermediate insitu dataset with elevation if available, otherwise processes the raw data.

    :param save_path: File path for the saved intermediate data.
    :type save_path: str
    :param overwrite: If True, reprocesses the data even if saved data exists.
    :type overwrite: bool
    :return: Insitu GeoDataFrame with elevation.
    :rtype: geopandas.GeoDataFrame
    """
    if os.path.exists(save_path) and not overwrite:
        print(f"Loading intermediate insitu data from {save_path}")
        df = pd.read_parquet(save_path)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkb(df.geometry), crs=CONFIG["coord_crs"])
        return gdf
    else:
        print("Processing raw insitu data to add elevation...")
        gdf = load_insitu_data()
        gdf.to_parquet(save_path, index=False)
        return gdf


def process_basin(insitu_gdf, basin_name):
    """
    Processes insitu data for a given basin by spatially filtering, aggregating sensor data,
    processing ASO and Daymet data, and merging them.

    :param insitu_gdf: Insitu GeoDataFrame.
    :type insitu_gdf: geopandas.GeoDataFrame
    :param basin_name: Name of the basin.
    :type basin_name: str
    :return: A cudf DataFrame with merged data.
    :rtype: cudf.DataFrame
    """
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
    filtered_cudf_sensors = cudf.from_pandas(filtered[sensor_cols])
    daily_sensors = filtered_cudf_sensors.resample('D').mean()

    filtered_cudf_z = cudf.from_pandas(filtered[['z']])
    daily_z = filtered_cudf_z.resample('D').first()
    daily = daily_sensors.to_pandas()
    daily['z'] = daily_z.to_pandas()['z']
    daily['pillow_swe'] = daily[sensor_cols].mean(axis=1, skipna=True)
    daily['pillows_used'] = filtered[sensor_cols].resample('D').apply(
        lambda x: x.columns[x.notna().any()].tolist()
    )

    # Process ASO data
    aso_ds = xr.open_dataset(CONFIG["basins"][basin_name]["aso"])
    aso_swe = aso_ds['aso_swe'].mean(dim=['x', 'y']) * 1000
    aso_df = cudf.from_pandas(
        aso_swe.to_series().to_frame('aso_swe').resample('D').mean()
    )

    # Process Daymet data
    from os.path import join
    sampled_files = [join(CONFIG["daymet_path"], var, f"sampled_{var}.parquet") for var in ['prcp', 'tmin', 'tmax', 'dayl', 'vp']]
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

    from os.path import basename
    daymet_dfs = [process_sampled_file(f) for f in tqdm(sampled_files, desc="Processing Daymet files")]
    daymet_dfs = [df for df in daymet_dfs if df is not None]
    if not daymet_dfs:
        raise ValueError(f"No valid Daymet data found for {basin_name}")
    daymet_df = (pd.concat(daymet_dfs)
                 .pivot_table(index='time', columns='variable_type', values='value')
                 .resample('D').mean()
                 .add_prefix('daymet_'))

    merged = daily[['pillow_swe', 'pillows_used', 'z']].merge(
        aso_df.to_pandas(), left_index=True, right_index=True, how='outer'
    ).merge(daymet_df, left_index=True, right_index=True, how='left')
    cols_to_interpolate = merged.columns.difference(['aso_swe'])
    merged[cols_to_interpolate] = merged[cols_to_interpolate].interpolate()

    merged['pillow_swe_corrected'] = merged['pillow_swe']
    mask = merged['aso_swe'].notna()
    merged.loc[mask, 'pillow_swe_corrected'] = merged.loc[mask, 'aso_swe']
    return cudf.from_pandas(merged)


def get_preprocessed_dataset(basin_name, insitu_gdf, save_path, overwrite=False):
    """
    Returns a preprocessed dataset for the given basin. Loads from save_path if available,
    otherwise processes the data and saves it.

    :param basin_name: Name of the basin.
    :type basin_name: str
    :param insitu_gdf: Insitu GeoDataFrame.
    :type insitu_gdf: geopandas.GeoDataFrame
    :param save_path: Path to save or load the dataset.
    :type save_path: str
    :param overwrite: If True, forces reprocessing.
    :type overwrite: bool
    :return: Preprocessed dataset as a pandas DataFrame.
    :rtype: pandas.DataFrame
    """
    if os.path.exists(save_path) and not overwrite:
        print(f"Loading preprocessed dataset for {basin_name} from {save_path}")
        df = pd.read_parquet(save_path)
        return df
    else:
        print(f"Processing basin {basin_name} and saving to {save_path}")
        df = process_basin(insitu_gdf, basin_name)
        if isinstance(df, cudf.DataFrame):
            df = df.to_pandas()
        df.to_parquet(save_path, index=True)
        return df
