# load_data.py
# lm_model.py
import glob
import datetime
import numpy as np
from sklearn import linear_model
import xarray as xr
import matplotlib.pyplot as plt
# import rioxarray as rxr
import os
import copy
import sys
import pandas as pd
from itertools import combinations
import sklearn.metrics as metrics
import json
import pickle
from sklearn.metrics import root_mean_squared_error,r2_score
import geopandas as gpd


def hello_world():
    return "hello world!"

def generate_unique_colors(n):
    """Generates n unique colors using matplotlib."""
    cmap = plt.get_cmap('hsv')  # You can use any colormap you like
    colors = [cmap(i/n) for i in range(n)]
    return colors


def load_observation_shape(aso_site_name,shape_crs,root_dir,isProjected = True):
    """
    Append xarray dataarrays of pillow data into a list
    Input:
      fpath - python string of relative filepath for observation shapefile.
      shape_crs - python string for epsg projection.
      isProjected - boolean to indicate whether shapefile should be projected.

    Output:
      obs_gdf - geopandas dataframe for observations.
    """
    obs_path = os.path.join(root_dir, "data", "insitu", aso_site_name, "qa", "obs_summary.shp")
    # obs_path = 'data/insitu/USCASJ/qa/obs_summary.shp'
    if isProjected:
        obs_gdf = gpd.read_file(obs_path).set_crs('EPSG:4326').to_crs(shape_crs)
    else:
        obs_gdf = gpd.read_file(obs_path).set_crs('EPSG:4326').to_crs(shape_crs)
    obs_gdf['colors'] = generate_unique_colors(len(obs_gdf))
    return obs_gdf

def load_baseline_features(root_dir,aso_site_name,obs_data = None):
    """
        Selects a baseline number of features to start to statistical models based
        on each having at least one valid flight per year.
        Input:
            summary_table_fpath - python relative string for pillow summary path.
        Output:
            preds - list of predictions for each cross-validated year.
    """
    ## 
    summary_table_fpath = os.path.join(root_dir, "data", "insitu", aso_site_name, "qa","total.csv")
    ## load table.
    df_summary_table = pd.read_csv(summary_table_fpath)
    ## convert time to datetime object.
    df_summary_table['time'] = pd.to_datetime(df_summary_table['time'])
    ## missing pillows.
    missing_pils = []
    ## check to see if there is mismatching information with full QA'd dataset.
    if obs_data is not None:
        for row,col in df_summary_table.iterrows():
          time = col['time']
          for pil_idx in range(0,len(obs_data)):
            pil = obs_data[pil_idx].name
            if pil not in df_summary_table.columns:
               missing_pils.append(pil_idx)
            else:
                val_obs = float(obs_data[pil_idx].sel({'time':time}))
                val_df = float(df_summary_table[df_summary_table['time'] == time][pil].values)
                if val_obs != val_df:
                    if (np.isnan(val_obs)) and (np.isnan(val_df)):
                        pass
                    else:
                        print(pil,time, 'qa data', val_obs, 'table data', val_df)
                        df_summary_table.loc[row,pil] = val_obs
                    # obs_data[pil_idx]
    ## notify user if there are duplicate dates
    if df_summary_table.time.nunique() != len(df_summary_table):
        print('DUPLICATE ROWS CONSIDER')
        sys.exit(0)

    ## create list of pillows.
    all_pils = df_summary_table.columns.to_list()
    all_pils.remove('time')
    all_pils.remove('aso_mean_bins_mm')
    ## group by year and sum to identify pillows without values in year.
    df_year = df_summary_table.groupby(df_summary_table.time.dt.year)[all_pils].sum()
    ## create list of pillows with at least one flight per year.
    pillow_w_flight_per_year = df_year.replace(0, np.nan).dropna(axis = 1,how = 'any').columns.to_list()
    pillows_cols = copy.deepcopy(pillow_w_flight_per_year)
    pillows_cols.append('time')
    slice_df = df_summary_table[pillows_cols]
    valid_time = slice_df.dropna(axis = 0, how = 'any').time.values
    slice_df = df_summary_table[df_summary_table['time'].isin(valid_time)].dropna(axis = 1, how = 'any')
    ## create baseline pillow list.
    baseline_pils = slice_df.columns.to_list()
    baseline_pils.remove('time')
    baseline_pils.remove('aso_mean_bins_mm')
    ## recreate list of datarrays.
    data_final = []
    for i in range(0,len(obs_data)):
       if i not in missing_pils:
          data_final.append(obs_data[i])
    return df_summary_table,slice_df,all_pils,baseline_pils,data_final

def load_pillow_nc(aso_site_name,root_dir):
    """
    Call CDEC API to download snow pillow data.
    Input:
      insitu_dir - python string of relative filepath for save insitu observations

    Output:
      obs_data - list of dataarrays for snow pillows.
      insitu_locations - geopandas object for locations of snow pillows.
    """

    obs_data = []

    insitu_dir = os.path.join(root_dir, "data", "insitu", aso_site_name, "qa/")

    ## create list of datarrays ##
    for file in sorted(os.listdir(insitu_dir)):
      if '.nc' in file:
        da = xr.open_dataarray(insitu_dir + file)
        obs_data.append(da)

    ## geopandas shape object ##
    for file in sorted(os.listdir(insitu_dir)):
      if ('obs_summary.shp' in file):
        insitu_locations = gpd.read_file(insitu_dir + file)
        insitu_locations = insitu_locations.set_crs('EPSG:4326')
        insitu_locations.rename(columns = {'elevation_':'elevation_m'},inplace = True)
    return obs_data,insitu_locations

def load_aso_tseries(aso_site_name,root_dir):
   aso_fpath = os.path.join(root_dir, "data", "aso", aso_site_name, "aso_tseries_1000ft.nc")
   return xr.load_dataarray(aso_fpath)

def load_dem_bins(aso_site_name,root_dir):
   dem_fpath = os.path.join(root_dir, "data", "dem", aso_site_name, "dem_50m_bins_1000ft.nc")
   return xr.load_dataarray(dem_fpath)

def load_mlr_basin_data(aso_site_name,root_dir,shape_crs = 'EPSG:32611'):
    """
    Load data relevant to MLR.
    Input:
    
    Output:
      basin_name - python string for basin.
      root_dir - python string of streamlit root directory.
    """
    
    # get observation locations.
    obs_gdf = load_observation_shape(aso_site_name,shape_crs,root_dir,isProjected = True)
    obs_gdf['pil_elev_f'] = obs_gdf['elevation_'] * 3.28084

    # load observations.
    obs_data,__ = load_pillow_nc(aso_site_name,root_dir)

    # get baseline tables.
    df_sum_total,slice_df,all_pils,baseline_pils,obs_data = load_baseline_features(root_dir,aso_site_name,obs_data)

    # load aso tseries.
    aso_tseries = load_aso_tseries(aso_site_name,root_dir)

    # load dem data.
    dem = load_dem_bins(aso_site_name,root_dir)

    return obs_gdf,obs_data,df_sum_total,slice_df,all_pils,baseline_pils,aso_tseries,dem

def select_vals_to_inputs(selected_basin,selected_elevation,selected_imputation,selected_season):
    """
    Load data relevant to MLR.
    Input:
    
    Output:
      basin_name - python string for basin.
      root_dir - python string of streamlit root directory.
    """
    # basin name to abbreviation.
    if selected_basin == 'San Joaquin':
        aso_site_name = 'USCASJ'
        start_wy = 2017
        end_wy = 2024
    else:
        aso_site_name = 'USCATM'
        start_wy = 2013 
        end_wy = 2024
    
    # elevation bin to index.
    if selected_elevation == '<7k':
        elev_band = 0
    elif selected_elevation == '7-8k':
        elev_band = 1
    elif selected_elevation == '8-9k':
        elev_band = 2
    elif selected_elevation == '9-10k':
        elev_band = 3
    elif selected_elevation == '10-11k':
        elev_band = 4
    elif selected_elevation == '11-12k':
        elev_band = 5
    elif selected_elevation == '12k':
        elev_band = 6
    elif selected_elevation == 'Total':
        elev_band = 7

    # imputation boolean.
    if selected_imputation == 'Drop NaNs':
        isImpute = False 
    elif selected_imputation == 'Predict NaNs':
        isImpute = True

    # seasonality booleans.
    if selected_season == 'Total Season':
        isSplit = False 
        isAccum = False 
    elif selected_season == 'Accumulation':
        isSplit = True 
        isAccum = True 
    elif selected_season == 'Melt':
        isSplit = True 
        isAccum = False 


    return aso_site_name,elev_band,isImpute,isSplit,isAccum,start_wy,end_wy


def select_vals_to_outputs(selected_basin):
    """
    Load data relevant to MLR.
    Input:
    
    Output:
      basin_name - python string for basin.
      root_dir - python string of streamlit root directory.
    """
    # basin name to abbreviation.
    if selected_basin == 'San Joaquin':
        aso_site_name = 'USCASJ'
    else:
        aso_site_name = 'USCATM'

    selected_dict = {'<7k':{'elevation_bin':'<7k',
                       'ymax_lim':100,
                       'text_adjust':20},
                '7-8k':{'elevation_bin':'7k-8k',
                       'ymax_lim':75,
                       'text_adjust':15},
                '8-9k':{'elevation_bin':'8k-9k',
                       'ymax_lim':50,
                       'text_adjust':10},
                '9-10k':{'elevation_bin':'9k-10k',
                       'ymax_lim':50,
                       'text_adjust':10},
                '10-11k':{'elevation_bin':'10k-11k',
                       'ymax_lim':50,
                       'text_adjust':10},
                '11-12k':{'elevation_bin':'11k-12k',
                       'ymax_lim':50,
                       'text_adjust':10},
                'Total':{'elevation_bin':'Total',
                       'ymax_lim':280,
                       'text_adjust':40},

    }


    return aso_site_name,selected_dict
   


   

    

