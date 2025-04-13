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

def run_mlr_cross_validation(aso_site_name,root_dir,aso_tseries_1,elev_band,isSplit,isImpute,
                             isAccum,df_sum_total,all_pils,obs_data,baseline_pils,start_wy,end_wy,
                             isMean = False,isCombination = False):
    ## code
    title_str = str(aso_tseries_1.elev[elev_band].values)



    if isSplit:
        if isImpute:
            ## update missing observations with average from flight date.
            if isMean:
                obs_data_5_,pils_removed,df_summary_impute = impute_pillow_mean(df_sum_total,all_pils,obs_data,obs_threshold = 0.50)
            else:
                obs_data_5_,pils_removed,df_summary_impute = impute_pillow_prediction(root_dir,aso_site_name,df_sum_total,all_pils,obs_data,obs_threshold = 0.50)

            ## split up data into accumlation and melt.
            df_split = process_melt_accum_thresh(aso_site_name,
                                                 root_dir,
                                                 df_summary_impute,
                                                 elev_band,
                                                 isAccum)

            predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6,table_dict = run_cross_val_selection(obs_data_5_,df_split,aso_tseries_1,pils_removed,start_wy,end_wy,
                                                                                                                     elev_band,isCombination = isCombination,showOutput = False,isMelt = isSplit)
        else:
            ## split up data into accumlation and melt.
            df_split = process_melt_accum_thresh(aso_site_name,
                                                 root_dir,
                                                 df_sum_total,
                                                 elev_band,
                                                 isAccum)

            predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6,table_dict = run_cross_val_selection(obs_data,df_split,aso_tseries_1,baseline_pils,start_wy,end_wy,
                                                                                                                     elev_band,isCombination = isCombination,showOutput = False,isMelt = isSplit)
    else:
        if isImpute:
            ## update missing observations with average from flight date.
            if isMean:
                obs_data_5_,pils_removed,df_summary_impute = impute_pillow_mean(df_sum_total,all_pils,obs_data,obs_threshold = 0.50)
            else:
                obs_data_5_,pils_removed,df_summary_impute = impute_pillow_prediction(root_dir,aso_site_name,df_sum_total,all_pils,obs_data,obs_threshold = 0.50)

            print('YES')
            predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6,table_dict = run_cross_val_selection(obs_data_5_,df_summary_impute,aso_tseries_1,pils_removed,start_wy,end_wy,
                                                                                                                     elev_band,isCombination = isCombination,showOutput = False,isMelt = isSplit)
        else:
            predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6,table_dict = run_cross_val_selection(obs_data,df_sum_total,aso_tseries_1,baseline_pils,start_wy,end_wy,
                                                                                                                     elev_band,isCombination = isCombination,showOutput = False,isMelt = isSplit)
    selected_pillow = [obs_data_6[i].name for i in range(0,len(obs_data_6)) if i in stations2]

    # plot basin mean v predictions ##
    if float(np.max(aso_tseries_2[:,elev_band].values)) < 600:
        max_swe_ = 600
    else:
        max_swe_ = 1600

    s = ","
    for k,v in table_dict.items():
        if type(v[0]) == list:
            v[0] = s.join([obs_data_6[i].name for i in v[0]])
            v[1] = int(v[1])

    return predictions_bestfit,predictions_validation,aso_tseries_2,obs_data_6,table_dict,selected_pillow,max_swe_,title_str,stations2


def process_melt_accum_thresh(aso_site_name,root_dir,df_sum_total,elev_bin,isAccum):
    """
    Slice data based on accumulation / melt threshold for each year.
    Input:
      thresh_fpath - relative filepath of threshold csv.
      df_sum_total - pandas dataframe of summary data.
      elev_bin - integer of elevation bin.
      isAccum - boolean for indicating accumulation or melt analysis.

    Output:
      df_accum - accumulation dataframe.
      df_melt - melt dataframe.
    """
    thresh_fpath = os.path.join(root_dir, "data", "insitu", aso_site_name, "qa","melt_threshold.csv")
    ## load melt threshold dataframe ##
    melt_thresh_df = pd.read_csv(thresh_fpath)
    melt_thresh_df['threshold_1'] = pd.to_datetime(melt_thresh_df['threshold_1'])
    melt_thresh_df['threshold_2'] = pd.to_datetime(melt_thresh_df['threshold_2'])
    melt_thresh_df['threshold_best'] = pd.to_datetime(melt_thresh_df['threshold_best'])

    ## slice data ##
    df_thresh_elev = melt_thresh_df[melt_thresh_df['elev_bin'] == elev_bin][['water_year','threshold_best']].reset_index().drop(columns = ['index'])

    ## add wateryear to df_sum_total for joining
    df_sum_total['water_year'] = df_sum_total.time.dt.year

    ## merge and threshold data
    df_merge = df_sum_total.merge(df_thresh_elev, on='water_year', how='left')
    if isAccum:
        df_accum = df_merge[df_merge['time'] < df_merge['threshold_best']].reset_index().drop(columns = ['water_year','threshold_best','index'])
        return df_accum
    else:
        df_melt = df_merge[df_merge['time'] >= df_merge['threshold_best']].reset_index().drop(columns = ['water_year','threshold_best','index'])
        return df_melt
    

def run_cross_val_selection(obs_data,df_sum_total,aso_tseries,all_pillows,start_wy,end_wy,
                            elev_band = -1,isCombination = True,showOutput = True,isMelt = False):
    """
        Runs cross-validation approach of linear regression based on year used 
        to run testing.
        Input:
            obs_data - list of xarray pillow observations.
            df_sum_total - pandas dataframe for observations, features, and labels WITH nans.
            aso_tseries_1 - xarray object for aso mean swe based on elevation.
            all_pillows - list of pillows to use.
            start_wy - integer for start water year.
            end_wy - integer for end water year.
            elev_band - elevation band to run model (note: -1 is entire domain)
            isCombination - boolean to determine which approach to use.
            isMelt - boolean to indicate additional threshold of dates for melt and accum.
            showOutput - boolean to indicate show output.
        Output:
            preds - list of predictions for each cross-validated year.
    """
    ## slice observation data ##
    obs_data_6 = [obs_data[i] for i in range(0,len(obs_data)) if obs_data[i].name in all_pillows]

    ## slice for melt or accum ##
    if isMelt:
        aso_tseries = aso_tseries[aso_tseries.date.isin(df_sum_total.time.values)]

    ## identify missing times ##
    missing_times = df_sum_total[df_sum_total[all_pillows].isnull().sum(axis=1) > 0].time.values

    ## slice aso mean swe ##
    aso_tseries_2 = aso_tseries[~aso_tseries.date.isin(missing_times)]

    ## run summary ##
    summary_data_total = summarize_data(obs_data_6, aso_tseries_2[:,elev_band])

    predictions_bestfit = summarize(summary_data_total,plotFig = showOutput,saveFig = False, axis_max = 1600)
    if showOutput:
        print(f'All features correlation: {np.corrcoef(predictions_bestfit, summary_data_total[-1])[0,1]**2:.3f}')
    ## run station selection based on cross-validation
    nstations = summary_data_total.shape[0]-1
    predictions_validation,best_stations,table_dict = cross_val_loyo_pred_select(aso_tseries_2[:,elev_band], summary_data_total[:-1,:],
                                                                      start_wy,end_wy,showOutput = showOutput)
    predictions_validation = np.array(predictions_validation)
    predictions_validation[predictions_validation<0]=0
    if showOutput:
        print('')
        print(f'Validation station correlation: {np.corrcoef(predictions_validation,summary_data_total[-1])[0,1]**2:.3f}')
        print('Select best stations')
    stations2 = identify_best_stations(best_stations,aso_tseries_2,elev_band,summary_data_total,isCombination,showOutput = showOutput)
    predictions_bestfit = cross_val_loo(aso_tseries_2[:,elev_band], summary_data_total[stations2,:])
    predictions_bestfit = np.array(predictions_bestfit)
    predictions_bestfit[predictions_bestfit<0]=0

    # print(f'Best fit station correlation: {np.corrcoef(predictions_validation,summary_data_total[-1])[0,1]**3:.3f}')
    # return predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6
    return predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6,table_dict

def summarize_data(pillow_data, aso_tseries):
    """
        Processes pillow data to match ASO time series using nearest date. 
        Input:
            pillow_data - list object of xarray dataarray objects representing
                          timeseries of each snow pillow.m
            time_series - dataarray object representing ASO time series of 
                          basin averaged SWE (mm)
        Output:
            full_data - numpy 2D array with number of pillows represented by number of
                        rows and aso dates represented by columns. Values within the
                        matrix represent the snow pillow values closest to each ASO 
                        flight. The last row data[-1,:] represents the actual ASO 
                        averaged values.
    """
    full_data = np.zeros((len(pillow_data)+1,len(aso_tseries)))

    for i,t in enumerate(aso_tseries):
        for j,p in enumerate(pillow_data):
            full_data[j,i] = p.sel(time=t.date, method="nearest")
            
## set the aso observations in the last row ##
        full_data[len(pillow_data),i] = t.values

    return full_data


def summarize(data,plotFig = True,saveFig=False,axis_max = 1600):
    """
        Runs linear regression on snow pillows to predict ASO basin mean SWE. 
        Utilizes Sklearn's linear_model.LinearRegression() to make the predictions. 
        Input:
            data - numpy 2D array of snow pillow observations and ASO basin mean SWE
                   in milimeters. The last row data[-1,:] represents the actual ASO 
                   averaged values.
        Output:
            res - numpy 1D array represented the predictions of ASO basin mean SWE for 
                  each data on linear regression model. 
    """
    ## create the linear regression model ##
    lm = linear_model.LinearRegression()
    
    ## fit model with data ##
    lm.fit(data[:-1,:].T,data[-1,:])
    
    ## predict aso values ##
    res = lm.predict(data[:-1,:].T)
    
    ## calculate statistics
    rms = np.sqrt(((res-data[-1])**2).mean())
    r2 = np.corrcoef(res,data[-1])[0,1]**2

    ## plot ASO vs. predicted SWE [mm] ##
    if plotFig:
        plt.clf()
        plt.plot(res,data[-1],'x',label = 'best fit: r$^2$' +f'={r2:.3f},RMS error={rms:.3f}')
        plt.ylabel("ASO SWE [mm]")
        plt.xlabel("predicted SWE [mm]")
        plt.legend()
        plt.xlim([0,axis_max])
        plt.ylim([0,axis_max])


        # plt.title("r$^2$  "+str(r2)[:5]+"  RMS error = "+str(rms)[:5])
        plt.title(f"Best Fit")
        plt.tight_layout()
        plt.show()
        if saveFig:
            plt.savefig("summary.png")
    # plt.xlim([0,625])
    # plt.ylim([0,575])
    return res

def identify_best_stations(best_stations,aso_tseries_2,elev_band,summary_data_total,
                           isCombination = True,showOutput = True):
    """
        Identifies best stations from cross-validation folds using two approaches.
        First, run combination of all selected pillows and select best stations
        based on adjusted R2. Second, start with pillow that has the most occurences in
        across all folds and add on the subsequent pillows of most occurence based
        on R2 value.
        Input:
            best_stations - list lists of selected pillows from cross-validation.
            summary_data_total - numpy array containing features and labels.
            aso_tseries_2 - xarray object for aso mean swe based on elevation.
            all_pillows - list of pillows to use.
            elev_band - elevation band to run model (note: -1 is entire domain).
            isCombination - boolean to determine which approach to use.
            showOutput - boolean to print output.
        Output:
            best_pillows - list of predictions for each cross-validated year.

    """
    ## instantiate max adjusted r2 variable.
    r2adj_max = 0
    ## use combination approach
    if isCombination:
        station_lst = list(set(sum(best_stations, [])))
        for i in range(1,len(station_lst)+1):
            if showOutput: print(i)
            comb = list(combinations(station_lst, i))
            for val in comb:
                predictions_validation = cross_val_loo(aso_tseries_2[:,elev_band], summary_data_total[list(val),:])
                predictions_validation = np.array(predictions_validation)
                predictions_validation[predictions_validation<0]=0
                r2 = np.corrcoef(predictions_validation,summary_data_total[-1])[0,1]**2
                n_ = len(predictions_validation)
                k_ = len(list(val))
                adj_r2 = 1 - ((1-r2) * (n_-1)/(n_-k_-1))
                if showOutput: print(list(val),adj_r2)
                if adj_r2 > r2adj_max:
                    best_pils = list(val)
                    r2adj_max = adj_r2
        if showOutput: print(best_pils,r2adj_max)
    else:
    ## use most repeat pillows 
        dic_ = {}
        for num in sum(best_stations, []):
            if num not in dic_:
                dic_[num] = 1
            else:
                dic_[num] += 1
        val_based = {k: v for k, v in sorted(dic_.items(), key=lambda item: item[1], reverse=True)}
        counter = 0
        val_lst = []
        for k,v in val_based.items():
            val_lst.append(k)
            predictions_validation = cross_val_loo(aso_tseries_2[:,elev_band], summary_data_total[val_lst,:])
            predictions_validation = np.array(predictions_validation)
            predictions_validation[predictions_validation<0]=0
            r2 = np.corrcoef(predictions_validation,summary_data_total[-1])[0,1]**2
            n_ = len(predictions_validation)
            k_ = len(val_lst)
            adj_r2 = 1 - ((1-r2) * (n_-1)/(n_-k_-1))
            if showOutput: print(val_lst,r2)
            if adj_r2 > r2adj_max:
                best_pils = copy.deepcopy(val_lst)
                r2adj_max = adj_r2
            else:
                val_lst.remove(k)
        if showOutput: print(best_pils,r2adj_max)


    return best_pils


def cross_val_loyo_pred_select(aso, index, start_year, end_year,
                               add_points=None, max_params=None,showOutput = True):
    """
        Runs cross-validation approach of linear regression based on year used 
        to run testing.
        Input:
            aso - datarray object representing ASO basin SWE.
            index - 2D numpy array representing the snow pillow features.
            start_year - integer representing starting year of ASO flights. 
            end_year - integer representing ending year of ASO flights. 
        Output:
            preds - list of predictions for each cross-validated year.
    """
    preds = []
    param_lst = []
    table_dict = {}
    ## loop through and test each year of data in cross validation approach ##
    for y in range(start_year,end_year+1):
        # print('')
        # print('')
        # print('YEAR',y)
        ## check to see if any flights are observed in year ##
        if len([True for i in aso.date if y == int(i.date.dt.year.values)]) > 0:
            test = []
            testy = []
            train = []
            trainy = []

            for i in range(len(aso)):
                if int(aso[i].date.dt.year.values) == y:
                    ## pull out pillow values for date ##
                    values = index[:,i]
                    ## set nonfinite values to 0 ##
                    values[~np.isfinite(values)] = 0
                    ## append features ##
                    test.append(values)
                    ## append labels ##
                    testy.append(aso.values[i])
                else:
                    values = index[:,i]
                    values[~np.isfinite(values)] = 0
                    train.append(values)
                    trainy.append(aso.values[i])

            if add_points is not None:
                for point in add_points:
                    train.append(point[0])
                    trainy.append(point[1])

            # print('trainx',np.array(train))
            # print('')
            # print('trainy',np.array(trainy))
            # print('')
            # print('testx',np.array(test))
            # print('')
            # print('testy',np.array(testy))                
            testy, params,error,corr = compute_linear(np.array(train), np.array(trainy), np.array(test), np.array(testy), max_params=max_params)
            table_dict[y] = [[int(i) for i in params],int(error)]
            if showOutput: print(f"{y}: {params}, {error:.2f}, {corr:.2f}")
            param_lst.append(params)

            if testy is not None:
                preds.extend(testy)
            else:
                preds.extend(np.zeros(len(test)))
        else:
            # pass
            if showOutput: print(f'No flights for: {y}')
            table_dict[y] = [np.nan,np.nan]
            
    return preds,param_lst,table_dict

def cross_val_loyo_pred_select_by_threshold(aso, index, pillow_names, start_year, end_year,
                               add_points=None, max_params=None,
                               min_num_features=None,min_num_obs=None,
                               fixedPillows = None):
    """
        Runs cross-validation approach of linear regression based on year used 
        to run testing. Handles NaNs by selecting fixed pillows when provided 
        and threshold values for number of valid dates and features.
        Input:
            aso - datarray object representing ASO basin SWE.
            index - 2D numpy array representing the snow pillow features.
            start_year - integer representing starting year of ASO flights. 
            end_year - integer representing ending year of ASO flights.
            add_points - list of points to add to training (default = None).
            max_params - integer for maximum number of features.
            min_num_features - integer for minimum threshold of features.
            min_num_obs - integer for minimum threshold of observations.
            fixedPillows - python list of pillow names (e.g. ['TUM','AGP'])
        Output:
            np.array(flatten(yhat_best_lst)) - array of validation predictions for each cross-validated year. NOTE: naming is wrong for variable.
            best_stn_lst - list of lists of best stations for each year used in validation. NOTE: naming is wrong for variable.
            combined_best_stns - list of best stations derived from unique values in best_stn_lst.
            combined_best_dates - numpy array of dates used in cross-validation training.
            combined_worst_dates - numpy array of dates never used in cross-validation training.
            np.array(flatten(yhat_valid_lst)) - array of best fit predictions. NOTE: naming is wrong for variable.
            valid_stns_lst - list of lists of best stations for each year used in validation. NOTE: naming is wrong for variable.
            worst_stns - list of stations not used in best fit.

    """
    preds = []
    param_lst = []
    orig_pillow_index = np.arange(0,index.shape[0])
    yhat_best_lst = []
    best_stn_lst = []
    best_dates_lst = []
    yhat_valid_lst = []
    valid_stn_lst = []

    ## loop through and test each year of data in cross validation approach ##
    for y in range(start_year,end_year+1):
        print('')
        print('')
        print(f'YEAR: {y}')
        ## check to see if any flights are observed in year ##
        if len([True for i in aso.date if y == int(i.date.dt.year.values)]) > 0:
            test = []
            testy = []
            train = []
            trainy = []
            trainDate = []
            testDate = []

            for i in range(len(aso)):
                if int(aso[i].date.dt.year.values) == y:
                    ## pull out pillow values for date ##
                    values = index[:,i]
                    ## set nonfinite values to 0 ##
                    # values[~np.isfinite(values)] = 0
                    ## append features ##
                    test.append(values)
                    ## append labels ##
                    testy.append(aso.values[i])
                    testDate.append(aso[i].date.values)
                else:
                    values = index[:,i]
                    # values[~np.isfinite(values)] = 0
                    train.append(values)
                    trainy.append(aso.values[i])
                    trainDate.append(aso[i].date.values)

            if add_points is not None:
                for point in add_points:
                    train.append(point[0])
                    trainy.append(point[1]) 
            
   

            ## find and remove stations with nan values in test dataset ##
            valid_stations_bool,train,test = drop_test_nans(np.array(train),np.array(test))


            ## find best fit based on threshold of nan values
            y_hat_bestfit,best_names,best_dates,y_hat_valid,valid_names = parameter_select_by_threshold(train,np.array(trainy),test,np.array(testy),
                                  orig_pillow_index[valid_stations_bool],pillow_names[valid_stations_bool],
                                  np.array(trainDate),min_num_features,min_num_obs,max_params,fixedPillows)
            
            yhat_best_lst.append(y_hat_bestfit)
            best_stn_lst.append(best_names)
            best_dates_lst.append(best_dates)
            yhat_valid_lst.append(y_hat_valid)
            valid_stn_lst.append(valid_names)

        else:
            print(f'no valid flights for {y}')
            return
    
    combined_best_dates = np.array(list(set(flatten(best_dates_lst))))
    combined_worst_dates = [i for i in aso.date.values if i not in combined_best_dates]
    combined_valid_stns = np.array(list(set(flatten(valid_stn_lst))))
    combined_best_stns = np.array(list(set(flatten(best_stn_lst))))
    worst_stns = [i for i in pillow_names if i not in combined_best_stns]

    return np.array(flatten(yhat_best_lst)),best_stn_lst,combined_best_stns,\
           combined_best_dates,combined_worst_dates,np.array(flatten(yhat_valid_lst)),\
           valid_stn_lst,combined_valid_stns,worst_stns
                

def subset_by_station(station_arry,station_names,train_X,train_Y,trainDate):
    ## find indices of stations
    fixed_station_indices = [i for i in range(0,len(station_names)) if station_names[i] in (np.intersect1d(station_arry,station_names))]
    ## boolean of valid dates
    bool_date_indexer = (np.isnan(train_X[:,fixed_station_indices]).sum(axis = 1) == 0)
    ## reindex based on boolean indexer 
    train_X_ = train_X[bool_date_indexer,:]
    train_Y_ = train_Y[bool_date_indexer]
    trainDate_ = trainDate[bool_date_indexer]

    return train_X_,train_Y_,trainDate_


def parameter_select_by_threshold(train_X,train_Y,test_X,test_Y,station_indices,
                                  station_names,trainDate,min_num_features,
                                  min_num_obs,max_params,fixedPillows):

    if fixedPillows is None:
        pass
    else:
        train_X,train_Y,trainDate = subset_by_station(np.array(fixedPillows),station_names,train_X,train_Y,trainDate)
        min_num_obs = 10
    
    ## create array with numpy of valid flights per station
    num_valid_obs = (~np.isnan(train_X)).sum(axis = 0)
    ## best error ##
    best_error = np.sum(test_Y**2)

    ## loop through threshold array to create linear regression
    for threshold in np.arange(num_valid_obs.min(),num_valid_obs.max()+1,1):
        ## find additional pillows to remove based on threshold ##
        additional_pillows_to_remove = list(np.where(num_valid_obs < threshold)[0])
        ## find valid pillow indices with threshold ##
        pillows_thresh = [i for i in range(0,train_X.shape[1]) if i not in additional_pillows_to_remove]
        ## all station names ##
        all_station_names = [station_names[i] for i in range(0,len(station_names)) if i in pillows_thresh]
        ## slice train_X
        arr_tX = train_X[:,pillows_thresh]
        ## find valid dates based on threshold ##
        valid_date_indexes = list(np.where((~np.isnan(arr_tX)).all(axis=1))[0])
        ## slice train_X ##
        arr_tX = arr_tX[valid_date_indexes,:]
        ## slice train_Y ##
        arr_tY = train_Y[valid_date_indexes]

        ## slice test_X
        arr_vX = test_X[:,pillows_thresh]

        ## run regression ##
        if ((arr_tX.shape[0] >= min_num_obs) and (arr_tX.shape[1] >= min_num_features)):
            y_hat_fit, params, error,corr = compute_linear(arr_tX, arr_tY, arr_vX, test_Y, max_params=max_params)
            ## best station names ##
            best_station_names = [all_station_names[i] for i in range(0,len(all_station_names)) if i in params]

            print('threshold',threshold,'features',arr_tX.shape[1],'observations',arr_tX.shape[0],'error', error)
            print('params',params)
            if error <= best_error:
                best_threshold = threshold
                y_hat_bestfit = y_hat_fit
                best_error = error
                valid_names = all_station_names
                best_names = best_station_names
                best_obs = arr_tX.shape[0]
                best_dates = [trainDate[i] for i in range(0,len(trainDate)) if i in valid_date_indexes]

                lm = get_model(np.arange(0,arr_tX.shape[1]), arr_tX, arr_tY)
                y_hat_valid = lm.predict(arr_vX)

    
    print('')
    print('BEST:\nthreshold',best_threshold,'features',len(valid_names),'observations',best_obs,'error', best_error)
    print('stn names',best_names)
    print('')

    return y_hat_bestfit,best_names,best_dates,y_hat_valid,valid_names



def drop_test_nans(train,test):
    ## get station indices of nans for test flights ##
    valid_stations = ~np.isnan(test).any(axis=0)
    return valid_stations,train[:,valid_stations],test[:,valid_stations]

    
def compute_linear(x, y, x_hat=None, y_hat=None, max_params=5):
    """
        Runs linear regression to produce predictions for each tested year.
        Input:
            x - numpy array for training features
            y - numpy array for training labels
            x_hat - numpy array for testing features.
            y_hat - numpy array for testing labels.
            max_params - integer ??????
        Output:
            res - numpy array of predictions
            p - list of parameters or most important snow observations
    """
    ## if no data exist for testing year, run linear regression on training ##
    if (y_hat is None) or (x_hat is None): 
        lm = linear_model.LinearRegression()
        lm.fit(x,y)
        p = np.arange(x.shape[1])
    else:
    ## find best model ##
        lm, p, error,corr = fit_best_model(x,y, x_hat,y_hat, max_params=max_params)
        # print('')
        # print('lm',lm)
        # print('')
        # print('p',p)
        # print('')
        # print('error',error)
        if lm is None:
            return None, None


    if x_hat is None:
        res = lm.predict(x[:,p])
    else:
        res = lm.predict(x_hat[:,p])
        # print('')
        # print(f'res',res)
        # print('')
        # print('error',error)

    if (y_hat is None) or (x_hat is None):
        return res
    else:
        return res, p, error,corr
        
        
def add_parameter(param,x,y):
    """
        Runs linear regression with an increasing number of snow observations 
        and chooses the optimimum number based on R2
        Input:
            x - numpy array for training features
            y - numpy array for training labels
            param - list of chosen stations or snow observations
        Output:
            res - numpy array of predictions
            p - list of parameters or most important snow observations
    """
    corr = np.zeros(x.shape[1])
    ## loop over the number of observations in training dataset ##
    for i in range(x.shape[1]):
        if i in param:
            pass
        else:
            test_param = param.copy()
            test_param.append(i)
            ## make predictions with new number of observations ##
            lm = get_model(test_param, x, y)
            newy = lm.predict(x[:,test_param])
            ## calculate r2 ##
            corr[i] = np.corrcoef(newy,y)[0,1]**2
            if not np.isfinite(corr[i]):
                corr[i] = 0
    ## find the station that contributes most to r2 value ##
    best_param = np.argmax(corr)
    test_param = param.copy()
    test_param.append(best_param)
    return test_param
    
def get_model(param, x, y):
    """
        Fit a linear regression model
        Input:
            x - numpy array for training features
            y - numpy array for training labels
            param - list of indexes to select from training data
        Output:
            lm - linear regression model
    """
    lm = linear_model.LinearRegression()
    lm.fit(x[:, param],y)
    return lm

def impute_pillow_prediction(root_dir,aso_site_name,df_sum_total,all_pils,obs_data_5,obs_threshold = 0.50):
    """
        Fit a linear regression model
        Input:
            df_sum_total - summary pandas dataframe with features and labels.
            all_pils - list of all pillow ids.
            obs_data_5 - list of xarray datarrays for pillow observations.
            obs_threshold - float of fraction of missing observations for pillow removal.
        Output:
            obs_data_5_ - new list of xarray datarrays for pillow observations with filled values.
            pils_removed - updated list of features.
            df_summary_impute - updated summary datatable.
    """
    ## find pillows with observations more than threshold.
    drop_bool = ((df_sum_total[all_pils].isnull().sum(axis=0) / len(df_sum_total)) < obs_threshold).values
    ## array of new pillows with observations more than threshold
    pils_removed = np.array(all_pils)[drop_bool]
    ## subset summary table to remove pillows with minimal observations.
    df_dropped_pils = df_sum_total[pils_removed]
    ## create impute filepath.
    impute_df_fpath = os.path.join(root_dir, "data", "insitu", aso_site_name, "qa","pillow_impute_MLR.csv")

    if not os.path.exists(impute_df_fpath):

        for pil in df_dropped_pils.columns:
            df_new = pd.DataFrame(df_dropped_pils[pil])
# iterate over observations.
            for row,col in df_new.iterrows():
                val = col[pil]
                # find nans
                if np.isnan(val):
                    valid_pillows = df_dropped_pils.iloc[row][~df_dropped_pils.iloc[row].isna()].index.tolist()
                    target_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == pil][0])
                    target_da = obs_data_5[target_id]
                
                    first_corr = {}
                    feature_list = []
                    for feat in valid_pillows:
                        feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                        feat_da = obs_data_5[feat_id]
                        mask_target = ~target_da.isnull()
                        mask_feat = ~feat_da.isnull()
                        target_vals = target_da.where(mask_target).where(mask_feat).values
                        feat_vals = feat_da.where(mask_target).where(mask_feat).values

                        target_vals = target_vals[~np.isnan(target_vals)]
                        feat_vals = feat_vals[~np.isnan(feat_vals)]
                        r2 = np.corrcoef(target_vals, feat_vals)[0,1]**2
                        k = 1
                        n = len(feat_vals)
                        adjr2= 1 - ((1-r2) * (n-1)/(n-k-1))

                        first_corr[feat] = adjr2

                    # find max correlated pillow.
                    best_corr_pillow = max(first_corr, key=first_corr.get)
                    best_corr_adjr2 = first_corr[best_corr_pillow]
                    feature_list.append(best_corr_pillow)
            
            
                    bestfeat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == best_corr_pillow][0])
                    # remove pillow from list.
                    valid_pillows.remove(best_corr_pillow)
                    blah = True
                    train_df = pd.merge(target_da.to_dataframe(),obs_data_5[bestfeat_id].to_dataframe(),on = 'time')
                    second_corr = {}
                    for feat in valid_pillows:
                        feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                        train_df_ = pd.merge(train_df,obs_data_5[feat_id].to_dataframe(),on = 'time')
                        train_df_ = train_df_[train_df_.index.year >= 2013]
                        train_df_ = train_df_.dropna(axis = 0)
                        feat_cols = train_df_.columns.to_list()
                        feat_cols.remove(pil)
                        target_vals = train_df_[pil].values
                        feat_vals = train_df_[feat_cols].values
                        r2 = np.corrcoef(target_vals, feat_vals.T)[0,1]**2
                        k = len(feat_cols)
                        n = len(feat_vals)
                        adjr2= 1 - ((1-r2) * (n-1)/(n-k-1))
                    
                        second_corr[feat] = adjr2

                    # find max correlated pillow.
                    second_corr_pillow = max(second_corr, key=second_corr.get)
                    second_corr_adjr2 = second_corr[second_corr_pillow]
                    if second_corr_adjr2 > best_corr_adjr2:
                        feature_list.append(second_corr_pillow)
                
                        second_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == second_corr_pillow][0])
                        # remove pillow from list.
                        valid_pillows.remove(second_corr_pillow)
                        blah = True
                        train_df = pd.merge(train_df,obs_data_5[second_id].to_dataframe(),on = 'time')
                        third_corr = {}
                        for feat in valid_pillows:
                            feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                            train_df_ = pd.merge(train_df,obs_data_5[feat_id].to_dataframe(),on = 'time')
                            train_df_ = train_df_[train_df_.index.year >= 2013]
                            train_df_ = train_df_.dropna(axis = 0)
                            feat_cols = train_df_.columns.to_list()
                            feat_cols.remove(pil)
                            target_vals = train_df_[pil].values
                            feat_vals = train_df_[feat_cols].values
                            r2 = np.corrcoef(target_vals, feat_vals.T)[0,1]**2
                            k = len(feat_cols)
                            n = len(feat_vals)
                            adjr2= 1 - ((1-r2) * (n-1)/(n-k-1))

                            third_corr[feat] = adjr2

                        # find max correlated pillow.
                        third_corr_pillow = max(third_corr, key=third_corr.get)
                        third_corr_adjr2 = third_corr[third_corr_pillow]
                        if third_corr_adjr2 > second_corr_adjr2:
                            feature_list.append(third_corr_pillow)

                    # train regression
                    final_train_df_ = obs_data_5[target_id].to_dataframe()
                    for feat in feature_list:
                        feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                        final_train_df_ = pd.merge(final_train_df_,obs_data_5[feat_id].to_dataframe(),on = 'time')

                    final_train_df = final_train_df_[final_train_df_.index.year >= 2013]
                    final_train_df = final_train_df.dropna(axis = 0)

                    lm = linear_model.LinearRegression()
                    lm.fit(final_train_df.values[:,1:],final_train_df.values[:,0])

                    res = lm.predict(df_dropped_pils.iloc[row][feature_list].values.reshape(1,-1))
                    if res < 0:
                        res = 0
                    df_dropped_pils.loc[row, pil] = res




        df_summary_impute = copy.deepcopy(df_dropped_pils)
        pillows_ = df_dropped_pils.columns.to_list()

        df_summary_impute['time'] = df_sum_total.time.values
        df_summary_impute['aso_mean_bins_mm'] = df_sum_total.aso_mean_bins_mm.values
        df_summary_impute = df_summary_impute[['time','aso_mean_bins_mm'] + pillows_]
        # ouput table.
        df_summary_impute.to_csv(impute_df_fpath,index = False)

    # load impute table.
    df_summary_impute = pd.read_csv(impute_df_fpath)
    df_summary_impute['time'] = pd.to_datetime(df_summary_impute['time'])
    ## create copy of observations.
    obs_data_5_ = copy.deepcopy(obs_data_5)
    ## fill observation data arrays with imputed values.
    for pil_id in pils_removed:
        ## create dataframe with time and pillow.
        df_impute = df_summary_impute[['time',pil_id]]
        ## identify pillow index.
        pil_idx = [i for i in range(0,len(obs_data_5_)) if obs_data_5_[i].name == pil_id][0]
        for row,col in df_impute.iterrows():
            time = col['time']
            val = col[pil_id]
            ## if values are same pass, else update value
            if (obs_data_5_[pil_idx].sel({'time':time}) == val).values == True:
                pass
            else:
                obs_data_5_[pil_idx].loc[dict(time=time)] = val
    return obs_data_5_,list(pils_removed),df_summary_impute

def impute_pillow_mean(df_sum_total,all_pils,obs_data_5,obs_threshold = 0.50):
    """
        Fit a linear regression model
        Input:
            df_sum_total - summary pandas dataframe with features and labels.
            all_pils - list of all pillow ids.
            obs_data_5 - list of xarray datarrays for pillow observations.
            obs_threshold - float of fraction of missing observations for pillow removal.
        Output:
            obs_data_5_ - new list of xarray datarrays for pillow observations with filled values.
            pils_removed - updated list of features.
            df_summary_impute - updated summary datatable.
    """
    ## find pillows with observations more than threshold.
    drop_bool = ((df_sum_total[all_pils].isnull().sum(axis=0) / len(df_sum_total)) < obs_threshold).values
    ## array of new pillows with observations more than threshold
    pils_removed = np.array(all_pils)[drop_bool]
    ## subset summary table to remove pillows with minimal observations.
    df_dropped_pils = df_sum_total[pils_removed]
    ## fill nans with mean across rows.
    df_dropped_pils = df_sum_total[pils_removed]
    df_summary_impute = df_dropped_pils.T.fillna(df_dropped_pils.mean(axis=1)).T
    df_summary_impute['time'] = df_sum_total.time.values
    df_summary_impute['aso_mean_bins_mm'] = df_sum_total.aso_mean_bins_mm.values

    ## create copy of observations.
    obs_data_5_ = copy.deepcopy(obs_data_5)
    ## fill observation data arrays with imputed values.
    for pil_id in pils_removed:
        ## create dataframe with time and pillow.
        df_impute = df_summary_impute[['time',pil_id]]
        ## identify pillow index.
        pil_idx = [i for i in range(0,len(obs_data_5_)) if obs_data_5_[i].name == pil_id][0]
        for row,col in df_impute.iterrows():
            time = col['time']
            val = col[pil_id]
            ## if values are same pass, else update value
            if (obs_data_5_[pil_idx].sel({'time':time}) == val).values == True:
                pass
            else:
                obs_data_5_[pil_idx].loc[dict(time=time)] = val
    return obs_data_5_,list(pils_removed),df_summary_impute
        
        
def fit_best_model(x,y, x_hat,y_hat, max_params=None):
    """
        Runs linear regression to produce predictions for each tested year.
        Input:
            x - numpy array for training features
            y - numpy array for training labels
            x_hat - numpy array for testing features.
            y_hat - numpy array for testing labels.
            max_params - integer ??????
        Output:
            res - numpy array of predictions
            p - numpy array of number of observations.
    """
    new_parameters = []
    parameters = []
    # error = np.sum(y_hat**2)
    error = 10000000
    best_error = error
    # print('starting best error', best_error)
    # print('previous starting best error', np.sum(y_hat**2))
    # print('max_params',max_params)

    if max_params is None:
        max_params = x.shape[0]-1

    ## select optimimum model based on maximum r2 values provided by
    ## subsequently adding more observations ##
    while (error <= best_error) and (len(parameters) < max_params):
        best_error = error
        parameters = new_parameters
        ## add new station that improves r2 ##
        new_parameters = add_parameter(parameters, x,y)
        ## calculate improvement as a result of new station ##
        lm = get_model(new_parameters, x, y)
        error = np.sqrt(np.mean((lm.predict(x_hat[:,new_parameters]) - y_hat)**2))
        # corr = r2_score(y_hat,lm.predict(x_hat[:,new_parameters]))
        # r2 =  np.corrcoef(y_hat, lm.predict(x_hat[:,new_parameters]))[0,1]**2
        # print('R2',r2)
        # print(f'{new_parameters},{error:.2f}')

    if len(parameters)==0:
        print('HERE')
        return None, None, None
        # raise ValueError("0 parameters not allowed")
    
    ## get linear regression model for optimum observations ##
    lm = get_model(parameters, x,y)
    corr = np.corrcoef(y_hat, lm.predict(x_hat[:,parameters]))[0,1]**2

    return lm, parameters, best_error,corr
    
def cross_val_loo(aso, index, add_points=None):
    """
        Runs linear regression given the stations selected from  
        cross-validation analysis.
        Input:
            aso - datarray object representing ASO basin SWE.
            index - 2D numpy array representing the snow pillow features. 
        Output:
            output - array of predictions for each cross-validated year.
    """
    output = []
    for y in range(len(aso)):
        test = []
        train = []
        trainy = []

        for i in range(len(aso)):
            if i == y:
                test.append(index[:,i])
            else:
                train.append(index[:,i])
                trainy.append(aso.values[i])

        if add_points is not None:
            for point in add_points:
                train.append(point[0])
                trainy.append(point[1])

        testy = compute_linear(np.array(train), np.array(trainy), np.array(test))

        output.extend(testy)
    return output
    
def find_best_aso_points(aso, index, stn_list, aso_site_name,
                         add_points=None,ntests=10, low_snow = 0,
                         max_flights=15,title=None, max_params=None, 
                         always_use=None,saveFig=False,figName=None):
    """
        Runs linear regression given the stations selected from  
        cross-validation analysis.
        Input:
            aso - datarray object representing ASO basin SWE.
            index - 2D numpy array representing the snow pillow features.
            stn_list - python list of pillow station indexes.
            aso_site_name - python string of aso basin code.
            add_points - list with two entries to add to regression model. 
                         Typically used to force the regression model to go through point 0,0.
                         Entry one: list of pillow values for the number of stations used. 
                         entry two: mean aso value.
                         ex. [[0,0,0,0],0] 

            ntests - python integer for number of tests.
            low_snow - python float for setting conditional on flights to be used.
            max_flights - python integer for maximum number of flights.
            title - python string for plot title.
            max_params - maximum number of parameters for regression.
            always_use - list of flight indexes to use.
            saveFig - python boolean for saving figure png. 
            figName - python string for figure name. Does not include ".png"
        Output:
            output - array of predictions for each cross-validated year.
    """
    stns = index[stn_list,:].T
    ## create low snow boolean ##
    lowsnow = aso >low_snow
    ## array of index of flights to use ##
    only_use = np.arange(len(aso)) #[aso > 20]
    errcount = 0
    flight_err = np.zeros(len(only_use))
    flight_r = np.zeros(len(only_use))
    flight_n = np.zeros(len(only_use))

    ## create empty lists of regressions statitistics ##
    r = [] # correlation
    mn = [] # worst fit
    mx = [] # best fit
    lr = []
    r_list = []
    rmed = [] # median fit

    ## loop from number of stations to max flights ##
    for i in range(len(stn_list)+1, max_flights):
        r_accum = 0
        l_accum = 0
        lcount = 0
        cmn = 1
        cmx = 0

        ## loop over number of tests ##
        for n in range(ntests):
            points = only_use[random_points(i, len(only_use), require=always_use)]
            if add_points is not None:
                yhat = compute_linear(np.vstack([add_points[0],stns[points,:]]), 
                                      np.hstack([add_points[1],aso.values[points]]), 
                                      np.array(stns), 
                                      max_params=max_params)
            else:
                yhat = compute_linear(np.array(stns[points,:]), 
                                      np.array(aso.values[points]), 
                                      np.array(stns), 
                                      max_params=max_params)
            yhat[yhat < 0] = 0

            r2 = np.corrcoef(yhat, aso)[0,1]**2
            r_accum += r2
            cmx = max(cmx, r2)
            cmn = min(cmn, r2)
            flight_r[points] += r2
            flight_n[points] += 1
            if r2>0.8:
                flight_err += (yhat - aso)**2
                errcount += 1
            
            r2 = np.corrcoef(yhat[lowsnow], aso[lowsnow])[0,1]**2
            if np.isfinite(r2):
                l_accum += r2
                lcount += 1
                r_list.append(r2)


        rmed.append(np.median(r_list))
        r_list = []
        r.append(r_accum / ntests)
        lr.append(l_accum / lcount)
        mn.append(cmn)
        mx.append(cmx)

    
    flight_err = np.sqrt(flight_err/errcount)
    print("r^2 = ",str(r[-1])[:5])

    ## plotting ##
    plt.figure(figsize=(14,3), dpi=200)
    plt.subplot(1,2,1)
    n = np.arange(len(r)) + len(stn_list)+1
    plt.plot(n, rmed,'x',label="Median Fit", color="C2")
    plt.plot(n, mn, label="Worst Fit", color="C0")
    plt.plot(n, mx, label="Best Fit", color="C1")
    plt.xlim(2, None) #max_flights - len(stn_list)+3)
    plt.xlabel("Number of Flights used")
    plt.ylabel("Correlation (r$^2$)")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.scatter(yhat, aso, c=flight_r/flight_n, marker='o', cmap="viridis_r")
    plt.colorbar()
    plt.xlabel("MLR predicted Basin SWE [mm]")
    plt.ylabel("ASO Basin SWE [mm]")
    plt.title(title)
    plt.show()


    ## save fig ##
    if saveFig == True:
        fig,ax = plt.subplots(dpi=300)
        plt.plot(n, rmed,'x',label="Median Fit", color="C2")
        plt.plot(n, mn, label="Worst Fit", color="C0")
        plt.plot(n, mx, label="Best Fit", color="C1")
        plt.xlim(2, None) #max_flights - len(stn_list)+3)
        plt.xlabel("Number of Flights used")
        plt.ylabel("Correlation (r$^2$)")
        plt.legend()
        if not os.path.exists(f'./data/figures/{aso_site_name}'):
            os.makedirs(f'./data/figures/{aso_site_name}')
        if figName is None:
            plt.savefig(f'./data/figures/{aso_site_name}/04_corr_v_flights.png',dpi=300)
        else:
            plt.savefig(f'./data/figures/{aso_site_name}/{figName}.png',dpi=300)

    
    return yhat,flight_r/flight_n

def random_points(n, size, require=None):
    """
    Runs linear regression given the stations selected from  
    cross-validation analysis.
    Input:
        n - number of stations.
        size - number of valid aso flights.
        require - python list of pillow station indexes.
    Output:
        output - array of predictions for each cross-validated year.
    """
    r = []
    points_to_select = list(range(size))
    if require is not None:
        for i in range(len(require)):
            if type(require[i]) is int:
                required_flight = require[len(require)-i-1]
            else:
                required_flights = require[len(require)-i-1]
                required_flight = required_flights[np.random.randint(0,len(required_flights))]

            r.append(points_to_select.pop(required_flight))
        n -= len(r)
        size -= len(r)
        
    for i in range(n):
        this = np.random.randint(0,size)
        r.append(points_to_select.pop(this))
        size -= 1

    return r




   

    

