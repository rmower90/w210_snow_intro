import os
import geopandas as gpd
import xarray as xr

def load_pillow_nc(insitu_dir):
    obs_data = []
    for file in sorted(os.listdir(insitu_dir)):
        if '.nc' in file:
            da = xr.open_dataarray(insitu_dir + file)
            obs_data.append(da)
    insitu_locations = None
    for file in sorted(os.listdir(insitu_dir)):
        if 'obs_summary.shp' in file:
            insitu_locations = gpd.read_file(insitu_dir + file)
            insitu_locations = insitu_locations.set_crs('EPSG:4326')
            insitu_locations.rename(columns={'elevation_': 'elevation_m'}, inplace=True)
    return obs_data, insitu_locations