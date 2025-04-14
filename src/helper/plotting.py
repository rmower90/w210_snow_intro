# plotting.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import os
from sklearn.metrics import root_mean_squared_error


def predictions_v_observations(aso_mean_swe,bestFit,validation,aso_site_name,ax,
                              max_swe = 1500,features = None,showPlot = True):
    """
        Produces one to one plot of basin mean swe vs. predicted swe
        Input:
            aso_mean_swe - xarray dataarray of basin mean swe for aso flight dates.
            bestFit - np array of best fit predictions.
            validation - np array of cross-validation approach.
            aso_site_name - python string for aso basin name.
            saveFig - python boolean for saving figure
            max_swe - max swe used for limits on figure.
            features - list of features.
        Output:
            plot of best fit and cross validation.
    """
    ## calculate statistics
    rms_validation = np.sqrt(((aso_mean_swe - validation)**2).mean().values)
    r2_validation = np.corrcoef(aso_mean_swe.values,validation)[0,1]**2
    rms_best = np.sqrt(((aso_mean_swe - bestFit)**2).mean().values)
    r2_best = np.corrcoef(aso_mean_swe.values,bestFit)[0,1]**2
    ## plotting
    if showPlot:
        # plt.figure(dpi=300)
        ax.plot(validation, aso_mean_swe,'o', color = 'C1', label="Validation: r$^2$" + f'= {r2_validation:.3f}\nRMS error     = {rms_validation:.2f}')
        ax.plot(bestFit, aso_mean_swe,'x', color = 'C0', label = 'Best Fit: r$^2$' + f'    = {r2_best:.3f}\nRMS error     = {rms_best:.2f}')
        ax.set_ylabel("ASO Mean SWE [mm]",fontweight = 'bold',fontsize = 8)
        ax.set_xlabel("Predicted Basin Mean SWE [mm]",fontweight = 'bold',fontsize = 8)
        if features is not None:
            ax.set_title(f'Predictions \nn = {len(validation)}; k = {len(features)}',fontweight = 'bold',fontsize = 10)
        ax.legend()
        ax.set_ylim([0,max_swe])
        ax.set_xlim([0,max_swe])
        return ax
    else:
        return r2_validation,rms_validation,r2_best,rms_best

def residual_comparison(y,yhat,ax,mask_val = 50,showPlot = True):
    """
        Compares manually downloaded and API data for snow pillows.
        Input:
            y - xarray of mean swe with date dimension.
            yhat - numpy array of predicted mean swe.
            mask_val - integer for mask.
        Output:
            plots of residuals.
    """

    percent_resid = 100*((y - yhat ) / y)
    swe_msk = y >= mask_val
    resid = y-yhat

    if showPlot:
        # fig,ax = plt.subplots(1,2,figsize = (12,6))
        im = ax.scatter(percent_resid.date,resid,c = y,cmap = 'inferno', label = 'mean swe > 50mm')
        ax.set_title(f'Residuals',fontweight = 'bold',fontsize = 10)
        ax.set_ylabel('Residual [mm]\n mean swe - predicted',fontsize = 8,fontweight = 'bold')
        ax.set_xlabel('Date',fontsize = 8,fontweight = 'bold')
        limit_vals_resid = resid.values
        limit_vals_resid = limit_vals_resid[~np.isnan(limit_vals_resid)]
        ax.set_ylim(-1*(np.abs(limit_vals_resid).max()+10),
                   np.abs(limit_vals_resid).max()+10)
        plt.colorbar(im,ax=ax,label = 'ASO Mean SWE [mm]')
        ax.grid(linestyle = '--',color = 'lightgray')

        for tick in ax.get_xticklabels():
          tick.set_rotation(45)

    return ax

def limits_btw_2spatial(spatial_2D,obs_gdf,isMax = True,isX = True):
    """
        Gets plotting limits from two spatial datasets (geodataframe and xarray).
        Input:
            spatial_2D - xarray 2D array for spatial plot.
            obs_gdf - geopandas dataframe of observations.
            isMax - max limit boolean.
            isX - x or y boolean.

        Output:
            val - python float of limit.
    """

    if isX:

        if isMax: # xmax
            val1 = obs_gdf.bounds['maxx'].max()
            val2 = float(spatial_2D.x.max().values)
            if val1 > val2:
                return val1
            else:
                return val2
        else: # xmin
            val1 = obs_gdf.bounds['minx'].min()
            val2 = float(spatial_2D.x.min().values)
            if val1 > val2:
                return val2
            else:
                return val1
    else:
        if isMax: # ymax
            val1 = obs_gdf.bounds['maxy'].max()
            val2 = float(spatial_2D.y.max().values)
            if val1 > val2:
                return val1
            else:
                return val2
        else: # ymin
            val1 = obs_gdf.bounds['miny'].min()
            val2 = float(spatial_2D.y.min().values)
            if val1 > val2:
                return val2
            else:
                return val1


def combined_spatial_cdf_pillow_distributions(spatial_2D,obs_gdf,sorted_vals,
                                              vals_cunane_position,pillow_lst,ax,
                                              title_str = 'All',otherIngray = False):
    """
        Creates spatial plot and cdf plot of a variable.
        Input:
            spatial_2D - xarray 2D array for spatial plot.
            obs_gdf - geopandas dataframe of observations.
            sorted_vals - numpy array of sorted values.
            vals_cunane_position - numpy array of plotting position.
            title_str - python string for tile of plot.
            otherIngray - boolean to represent whether other pillows are shown in gray
                          on spatial and cdf plots.
            showPlot - boolean to show plot.

        Output:
            plots of ASO flights.
    """


    ## cdf plot
    ax.plot(sorted_vals,vals_cunane_position,'-', color = 'black', label='Elevation CDF')
    if otherIngray:
        all_stations = list(obs_gdf.id.values)
        not_selected = [i for i in all_stations if i not in pillow_lst]
        for pil in not_selected:
            ax.axvline(obs_gdf[obs_gdf['id'] == pil].pil_elev_f.values,color = 'gray',linestyle = '--',linewidth = 0.5)

    for pil in pillow_lst:
        ax.axvline(obs_gdf[obs_gdf['id'] == pil].pil_elev_f.values,color = obs_gdf[obs_gdf['id']==pil].colors.values[0],linestyle = '--',label = pil)
    plt.legend()
    ax.set_xlabel('Elevation [ft]',fontsize = 8,fontweight = 'bold')
    ax.set_ylabel('Cumulative Frequency',fontsize = 8,fontweight = 'bold')
    ax.set_title('Pillow Distribution',fontsize = 10,fontweight = 'bold')
    # plt.suptitle(f'{title_str} pillow distribution over basin topography',fontweight = 'bold',fontsize = 20)
    # plt.tight_layout()
    # plt.show()


def predictions_v_observations_tseries(aso_mean_swe,validation,pillow_data,ax2,obs_gdf,
                                       stations,start_wy,end_wy,aso_site_name,saveFig = False,
                                       max_swe = 1500):

    """
        Plot time series of pillows that are best predictors for ASO.
        Input:
            aso_mean_swe - xarray dataarray of basin mean swe for aso flight dates.
            validation - np array of cross-validation approach.
            pillow_data - xarray dataset with pillow data.
            stations - list of station integers identifying best predictors.
            start_wy - integer for starting water year.
            end_wy - integer for ending water year.
            max_swe - max swe used for limits on figure.
        Output:
            time series plot.
    """

    # plt.figure(dpi=300, figsize=(8,3))

    # ax2 = plt.gca()
    ax1 = plt.twinx(ax2)

    ax2.plot(aso_mean_swe.date, aso_mean_swe.data, marker="o", linestyle="", color="C0", label="ASO", zorder=3)
    ax2.plot(aso_mean_swe.date, validation, 'x', color="C1", label="Predicted", zorder=3)
    plt.ylabel("ASO Mean SWE [mm]",fontweight = 'bold',fontsize = 8)

    colors = [f'C{str(i+2)}' for i in range(0,len(stations))]

    for i,c in zip(stations, colors):
        plot_data = np.array(pillow_data[i].data, dtype="f")

        # ax1.plot(pillow_data[i].time, plot_data, label=pillow_data[i].name, color=c) #obs_gdf[obs_gdf['id']==pil].colors.values[0]
        ax1.plot(pillow_data[i].time, plot_data, label=pillow_data[i].name, color=obs_gdf[obs_gdf['id']==pillow_data[i].name].colors.values[0])


    ax1.set_ylabel("Pillow Observed SWE [mm]",fontweight = 'bold',fontsize = 8)
    ax2.set_ylabel("Basin Mean SWE [mm]",fontweight = 'bold',fontsize = 8)
    # ax2.set_ylim(0,max_swe)
    ax2.set_xlim(np.datetime64(f'{start_wy-1}-10-01'),np.datetime64(f'{end_wy}-08-01'))
    ax1.set_xlim(np.datetime64(f'{start_wy-1}-10-01'),np.datetime64(f'{end_wy}-08-01'))
    ax2.set_ylim(0,max_swe)
    ax1.set_ylim(0,3600)
    # plt.xlim(np.datetime64("2012-10-01"),np.datetime64("2019-09-01"))
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title('Observation and Prediction Timeseries',fontweight = 'bold',fontsize = 10)
    return

def combine_cross_validation_plots(aso_site_name,elev_band,isImpute,aso_tseries_2,predictions_bestfit,predictions_validation,
                                   max_swe_,stations2,dem_bin,obs_gdf,sorted_elev,elev_cunane_position,obs_data_6,
                                   start_wy,end_wy,table_dict,title_str):
    if elev_band == 0:
        elev_string = '7k'
    elif elev_band == 1:
        elev_string = '7-8k'
    elif elev_band == 2:
        elev_string = '8-9k'
    elif elev_band == 3:
        elev_string = '9-10k'
    elif elev_band == 4:
        elev_string = '10-11k'
    elif elev_band == 5:
        elev_string = '11-12k'
    elif elev_band == 6:
        elev_string = '12k'
    elif elev_band == 7:
        elev_string = 'Total'

    if isImpute:
      impute_string = "Predict_NaNs"
      impute_string = "LRSS"
    else:
      impute_string = "Drop_NaNs"

    fig = plt.figure(figsize = (10,8),dpi = 200)
    ax1 = plt.subplot2grid((3, 4), (0, 0),rowspan = 2,colspan=2)  # Spans all 3 columns in row 0
    ax2 = plt.subplot2grid((3, 4), (0, 2),rowspan = 1,colspan=2)  # Spans all 3 columns in row 0
    ax3 = plt.subplot2grid((3, 4), (1, 2))            # Single cell in row 2, column 0
    ax4 = plt.subplot2grid((3, 4), (1, 3))
    ax5 = plt.subplot2grid((3, 4), (2, 0),rowspan = 1,colspan=4)

    ax1 = predictions_v_observations(aso_tseries_2[:,elev_band],predictions_bestfit,predictions_validation,
                                    aso_site_name,ax1,max_swe = max_swe_,features = stations2,showPlot = True)
    
    ax3 = residual_comparison(aso_tseries_2[:,elev_band],predictions_validation,ax3,mask_val = 50,showPlot = True)


    ax4 = combined_spatial_cdf_pillow_distributions(dem_bin[-1,:,:],obs_gdf,sorted_elev,elev_cunane_position,[obs_data_6[i].name for i in stations2],ax4,title_str,otherIngray = True)

    ax5 = predictions_v_observations_tseries(aso_tseries_2[:,elev_band],predictions_validation,obs_data_6,ax5,obs_gdf,
                                       stations2,start_wy,end_wy,aso_site_name,saveFig = False,
                                       max_swe = 1500)


    df_cross = pd.DataFrame(table_dict).T.reset_index().rename(columns = {'index':'Fold',0:'Pillows',1:'RMSE'})

    df_cross = pd.DataFrame(table_dict).T.reset_index().rename(columns = {'index':'Fold',0:'Pillows',1:'RMSE'})

    cross_table = ax2.table(cellText=df_cross.values, colLabels=df_cross.columns,loc = 'upper center',cellLoc = 'center',colColours = ['gray']*len(df_cross.columns))
    cross_table.auto_set_font_size(False)
    cross_table.set_fontsize(6)
    cross_table.scale(1, 1.2)
    ax2.axis('off')

    num_rows = len(df_cross)
    for i in range(num_rows):
        table_cell = cross_table[i+1, 0]
        table_cell.set_facecolor('gray')

    ax2.set_title('Cross Validation Summary',fontweight = 'bold',fontsize = 10)

    plt.tight_layout()
    return fig


def cunnane_quantile_array(numbers):
    '''This function also computes the Cunnane plotting position given an array or list of numbers (rather than a pandas dataframe).
    It has two outputs, first the sorted numbers, second the Cunnane plotting position for each of those numbers.
    [Steven Pestana, spestana@uw.edu, Oct. 2020]'''

    # 1) sort the data, using the numpy sort function (np.sort())
    sorted_numbers = np.sort(numbers)

    # length of the list of numbers
    n = len(sorted_numbers)

    # make an empty array, of the same length. below we will add the plotting position values to this array
    cunnane_plotting_position = np.empty(n)

    # 2) compute the Cunnane plotting position for each number, using a for loop and the enumerate function
    for rank, number in enumerate(sorted_numbers):
        cunnane_plotting_position[rank] = ( (rank+1) - (2/5) ) / ( n + (1/5) )

    return sorted_numbers, cunnane_plotting_position
def terrain_cdf_distribution(data):
    """
    Sorts distribution to be plotted as cdf.
    Input:
      data - flattend-1D numpy array with values.

    Output:
      sorted_vals - numpy array of sorted values.
      cunane_position - numpy array of plotting position for values.
    """

    vals_nonnan = data[~np.isnan(data)]
    sorted_vals, cunane_position = cunnane_quantile_array(vals_nonnan)
    return sorted_vals,cunane_position


def create_difference_plots_benchmark_mlr_mult_dates(root_dir,aso_site_name,ax,date_str = '20250226',
                                                     elevation_bin = 'Total',ymax_lim = None,FirstPlot = True,text_adjust = 40):
    """
    Comparison bar plots of difference between ASO with predictions and benchmark.
    Input:
      root_dir - python string of streamlit root directory.
      aso_site_name - python string of ASO site name abbreviation.
      ax - matplotlib subplot axis.
      date_str - python prediction date string with format 'YEARMODAY'
      elevation_bin - python string of elevation bin.
      ymax_lim - integer for ymax plotting limit.
      FirstPlot - boolean indicating first plot in series.
      text_adjust - integer to adjust text upwards.


    Output:
      ax - matplotlib subplot axis.
      maximum yvalue.
    """
    file_fpath = os.path.join(root_dir, "data", "MLR_Comparison", aso_site_name, f"ASO_PREDICTION_COMP_{date_str}.csv")
    
    df = pd.read_csv(file_fpath)

    snodas = df[df['Elevation'] == elevation_bin]['SNODAS_Pred_THacreFt'].unique()
    uaswe =  df[df['Elevation'] == elevation_bin]['UASWE_Pred_THacreFt'].unique()
    aso = df[df['Elevation'] == elevation_bin]['ASO_Pred_THacreFt'].unique()
    mlr_dropna = df[(df['Elevation'] == elevation_bin) & (df['IsSplit'] == True) & (df['IsAccum'] == True) & (df['Training_Infer_NaNs'] == 'Drop NaNs')& (df['Prediction_QA'] == 3)]['MLR_Pred_THacreFt'].values
    mlr_predictna = df[(df['Elevation'] == elevation_bin) & (df['IsSplit'] == True) & (df['IsAccum'] == True) & (df['Training_Infer_NaNs'] == 'Predict NaNs')& (df['Prediction_QA'] == 3)]['MLR_Pred_THacreFt'].values

    diff_snodas = aso - snodas
    diff_uaswe = aso - uaswe
    diff_drop = aso - mlr_dropna
    diff_predict = aso - mlr_predictna


    ax.bar([1],np.abs(diff_drop),label = 'MLR - DropNa',color = 'C0')
    ax.bar([2],np.abs(diff_predict),label = 'MLR - PredictNa',color = 'C1')
    ax.bar([7],np.abs(diff_snodas),label = 'SNODAS',color = 'C4')
    ax.bar([8],np.abs(diff_uaswe),label = 'UASWE',color = 'C5')

    if FirstPlot:
      ax.set_ylabel('Net Water Difference\n[acre-Ft]',fontweight = 'bold',fontsize = 16)
    if date_str == '20250226':
      ax.set_title('February 26th, 2025',fontweight = 'bold',fontsize = 20)
    else:
      ax.set_title('March 25th, 2025',fontweight = 'bold',fontsize = 20)
    ax.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False) # labels along the bottom edge are off

    vals = np.array([float(np.abs(diff_drop)[0]),float(np.abs(diff_predict)[0]),float(np.abs(diff_snodas)[0]),float(np.abs(diff_uaswe)[0])])
    ax.grid(linestyle = '--',axis = 'y')
    if FirstPlot:
      ax.legend(prop={'size': 12})
    ax.set_xlim(0,9)
    if ymax_lim is None:
      ax.set_ylim(0,np.max(vals)+50)
    else:
      ax.set_ylim(0,ymax_lim+50)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(10)  # Set the font size (e.g., to 12)
        tick.set_fontweight('bold')

    if date_str == '20250226':
      values = [np.abs(diff_drop)+text_adjust,np.abs(diff_predict)+text_adjust]
    else:
      values = [np.abs(diff_drop)+text_adjust,np.abs(diff_predict)+text_adjust]
    values_str = [f'{float(np.abs(diff_drop)):.0f}\n({(float(np.abs(diff_drop)))/float(aso)*100:.1f}%)',f'{float(np.abs(diff_predict)):.0f}\n({(float(np.abs(diff_predict)))/float(aso)*100:.1f}%)']
    counter = 0
    for i in [1,2]:
      ax.text(i, values[counter], values_str[counter], ha='center', va='top',fontsize = 12)
      counter += 1

    if date_str == '20250226':
      values = [np.abs(diff_snodas)+text_adjust,np.abs(diff_uaswe)+text_adjust]
    else:
      values = [np.abs(diff_snodas)+text_adjust,np.abs(diff_uaswe)+text_adjust]
    values_str = [f'{float(np.abs(diff_snodas)):.0f}\n({(float(np.abs(diff_snodas)))/float(aso)*100:.1f}%)',f'{float(np.abs(diff_uaswe)):.0f}\n({(float(np.abs(diff_uaswe)))/float(aso)*100:.1f}%)']
    counter = 0
    for i in [7,8]:
      ax.text(i, values[counter], values_str[counter], ha='center', va='top',fontsize = 12)
      counter += 1
    return ax,np.max(vals)


def create_model_comparison(root_dir,aso_site_name,date_str):
  
  """
    create model comparison plots.
    Inputs:
        root_dir = python string for root directory.
        aso_site_name - python string of ASO basin abbreviation.
        date_str - string for flight date (format:'YYMODY')
  """
  file_fpath = os.path.join(root_dir, "data", "MLR_Comparison", aso_site_name, f"ASO_PREDICTION_COMP_{date_str}.csv")
  df = pd.read_csv(file_fpath)

  # mlr drop na.
  mlr_dropna = df[(df['IsSplit'] == True) & (df['IsAccum'] == True) & (df['Training_Infer_NaNs'] == 'Drop NaNs')& (df['Prediction_QA'] == 3)][['Elevation','MLR_Pred_THacreFt']]
  mlr_dropna.loc[mlr_dropna['MLR_Pred_THacreFt'] < 0, 'MLR_Pred_THacreFt'] = 0.0
  mlr_dropna_total = mlr_dropna[mlr_dropna['Elevation'] == 'Total']
  mlr_dropna_combined = mlr_dropna[mlr_dropna['Elevation'] != 'Total']

  # mlr predict na.
  mlr_predictna = df[(df['IsSplit'] == True) & (df['IsAccum'] == True) & (df['Training_Infer_NaNs'] == 'Predict NaNs')& (df['Prediction_QA'] == 3)][['Elevation','MLR_Pred_THacreFt']]
  mlr_predictna.loc[mlr_predictna['MLR_Pred_THacreFt'] < 0, 'MLR_Pred_THacreFt'] = 0.0
  mlr_predictna_total = mlr_predictna[mlr_predictna['Elevation'] == 'Total']
  mlr_predictna_combined = mlr_predictna[mlr_predictna['Elevation'] != 'Total']

  # aso.
  aso = df[(df['IsSplit'] == True) & (df['IsAccum'] == True) & (df['Training_Infer_NaNs'] == 'Drop NaNs')& (df['Prediction_QA'] == 3)][['Elevation','ASO_Pred_THacreFt']]
  aso_total = aso[aso['Elevation'] == 'Total']
  aso_combined = aso[aso['Elevation'] != 'Total']

  # snodas.
  snodas = df[(df['IsSplit'] == True) & (df['IsAccum'] == True) & (df['Training_Infer_NaNs'] == 'Drop NaNs')& (df['Prediction_QA'] == 3)][['Elevation','SNODAS_Pred_THacreFt']]
  snodas_total = snodas[snodas['Elevation'] == 'Total']
  snodas_combined = snodas[snodas['Elevation'] != 'Total']

  # uaswe.
  uaswe = df[(df['IsSplit'] == True) & (df['IsAccum'] == True) & (df['Training_Infer_NaNs'] == 'Drop NaNs')& (df['Prediction_QA'] == 3)][['Elevation','UASWE_Pred_THacreFt']]
  uaswe_total = uaswe[uaswe['Elevation'] == 'Total']
  uaswe_combined = uaswe[uaswe['Elevation'] != 'Total']

  # create kpi table.
  rmse_dropna = root_mean_squared_error(mlr_dropna_combined['MLR_Pred_THacreFt'],aso_combined['ASO_Pred_THacreFt'])
  rmse_predictna = root_mean_squared_error(mlr_predictna_combined['MLR_Pred_THacreFt'],aso_combined['ASO_Pred_THacreFt'])
  rmse_snodas = root_mean_squared_error(snodas_combined['SNODAS_Pred_THacreFt'],aso_combined['ASO_Pred_THacreFt'])
  rmse_uaswe = root_mean_squared_error(uaswe_combined['UASWE_Pred_THacreFt'],aso_combined['ASO_Pred_THacreFt'])

  percent_drop = float((np.abs(mlr_dropna_total['MLR_Pred_THacreFt'].values - aso_total['ASO_Pred_THacreFt'].values)/aso_total['ASO_Pred_THacreFt'].values * 100)[0])
  percent_predict = float((np.abs(mlr_predictna_total['MLR_Pred_THacreFt'].values - aso_total['ASO_Pred_THacreFt'].values)/aso_total['ASO_Pred_THacreFt'].values * 100)[0])
  percent_snodas = float((np.abs(snodas_total['SNODAS_Pred_THacreFt'].values - aso_total['ASO_Pred_THacreFt'].values)/aso_total['ASO_Pred_THacreFt'].values * 100)[0])
  percent_uaswe = float((np.abs(uaswe_total['UASWE_Pred_THacreFt'].values - aso_total['ASO_Pred_THacreFt'].values)/aso_total['ASO_Pred_THacreFt'].values * 100)[0])


  kpi_df = pd.DataFrame({'Model':['MLR - DropNa','MLR - PredictNa','SNODAS','UASWE'],
                        'Accuracy [%]':[f'{percent_drop:.1f}',f'{percent_predict:.1f}',f'{percent_snodas:.1f}',f'{percent_uaswe:.1f}'],
                        'RMSE [Th acreFt]':[int(rmse_dropna),int(rmse_predictna),int(rmse_snodas),int(rmse_uaswe)]}
  )
  # plotting.
  fig = plt.figure(figsize = (10,8),dpi = 200)

  ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=2)
  ax2 = plt.subplot2grid((2, 3), (0, 2))
  ax3 = plt.subplot2grid((2, 3), (1, 0),colspan = 3)

  # Plot directly on ax1 and keep the returned Line2D objects
  line1, = ax1.plot(mlr_dropna_combined['Elevation'], mlr_dropna_combined['MLR_Pred_THacreFt'],
                  label='MLR - DropNa', linestyle='--', color='C0')
  line2, = ax1.plot(mlr_predictna_combined['Elevation'], mlr_predictna_combined['MLR_Pred_THacreFt'],
                  label='MLR - PredictNa', linestyle='--', color='C1')
  line3, = ax1.plot(snodas_combined['Elevation'], snodas_combined['SNODAS_Pred_THacreFt'],
                  label='SNODAS', linestyle='--', color='C4')
  line4, = ax1.plot(uaswe_combined['Elevation'], uaswe_combined['UASWE_Pred_THacreFt'],
                  label='UASWE', linestyle='--', color='C5')
  line5, = ax1.plot(aso_combined['Elevation'], aso_combined['ASO_Pred_THacreFt'],
                  label='ASO', linewidth=3, color='black', linestyle='-')

  # Labeling
  ax1.set_ylabel('SWE Volumne\n[Thousand acre-Ft]', fontweight='bold')
  ax1.set_title('SWE vs. Elevation', fontweight='bold')
  ax1.set_ylim(0,300)
  ax1.set_xlabel('Elevation Bins [ft]',fontweight = 'bold')
# ax1.grid(linestyle = '--',color = 'gray')

  # Create two legends
  legend1 = ax1.legend(handles=[line5], loc='upper left', title='Reference')
  legend1.get_title().set_fontweight('bold')
  legend2 = ax1.legend(handles=[line1, line2, line3, line4], loc='upper right', title='Model Output')
  legend2.get_title().set_fontweight('bold')

  # Add the first legend again
  ax1.add_artist(legend1)

  bar1 = ax2.bar([0],mlr_dropna_total['MLR_Pred_THacreFt'].values,color = 'C0',label = 'MLR - DropNa')
  bar2 = ax2.bar([1],mlr_predictna_total['MLR_Pred_THacreFt'].values,color = 'C1',label = 'MLR - PredictNa')
  bar3 = ax2.bar([2],snodas_total['SNODAS_Pred_THacreFt'].values,color = 'C4',label = 'SNODAS')
  bar4 = ax2.bar([3],uaswe_total['UASWE_Pred_THacreFt'].values,color = 'C5',label = 'UASWE')
  bar5 = ax2.bar([5],aso_total['ASO_Pred_THacreFt'].values,color = 'black',label = 'ASO')

  ax2.set_ylim(0,1750)
  ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

  ax2.set_ylabel('SWE Volumne\n[Thousand acre-Ft]', fontweight='bold')

  legend3 = ax2.legend(handles=[bar5], loc='lower right', title='Reference')
  legend3.get_title().set_fontweight('bold')
  legend4 = ax2.legend(handles=[bar1, bar2, bar3, bar4], loc='upper right', title='Model Output')
  legend4.get_title().set_fontweight('bold')

  # Add the first legend again (ASO/Reference)
  ax2.add_artist(legend3)

  ax2.set_title('Total Basin SWE Volume', fontweight='bold')


  cross_table = ax3.table(cellText=kpi_df.values, colLabels=kpi_df.columns,loc = 'upper center',cellLoc = 'center',colColours = ['gray']*len(kpi_df.columns))
  cross_table.auto_set_font_size(False)
  cross_table.set_fontsize(12)
  cross_table.scale(1, 4)
  ax3.axis('off')

  num_rows = len(kpi_df)
  for i in range(num_rows):
      table_cell = cross_table[i+1, 0]
      if i == 0:
        table_cell.set_facecolor('C0')
      elif i == 1:
        table_cell.set_facecolor('C1')
      elif i == 2:
        table_cell.set_facecolor('C4')
      elif i == 3:
        table_cell.set_facecolor('C5')
  ax3.set_title('KPI Results',fontweight = 'bold')

  if date_str == '20250226':
    plt.suptitle('San Joaquin Model Comparison\nFebruary 26th, 2025',fontweight = 'bold',fontsize = 24)
  else:
    plt.suptitle('San Joaquin Model Comparison\nMarch 25th, 2025',fontweight = 'bold',fontsize = 24)
  plt.tight_layout()
  return fig
    
