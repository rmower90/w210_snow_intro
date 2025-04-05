"""
Model Module
~~~~~~~~~~~~

This module provides a function for training two models (cuML Linear Regression and XGBoost)
on combined data from all basins. The data is split into training (<cutoff_year) and test (>=cutoff_year)
sets, and features are constructed from available columns. Model hyperparameters are loaded from a YAML
configuration file. Optionally, the trained models are saved.

:author: Your Name
:date: YYYY-MM-DD
"""

import os
import numpy as np
import pandas as pd
import cudf
import xgboost as xgb
from cuml.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib  # for saving/loading models
import plotly.graph_objects as go
import yaml

def train_all_models(data, cutoff_year=2023, n_splits=5, save_path=None):
    """
    Trains two models (cuML Linear Regression and XGBoost) on combined data from all basins.
    Data is split into training (<cutoff_year) and test (>=cutoff_year) sets.

    The target is 'pillow_swe_corrected' if available; otherwise 'pillow_swe'.
    Predictor features include 'pillow_swe' (as a predictor), 'z' (if available), and all columns 
    starting with 'daymet_'. SWE measurements are excluded from the predictors.

    Model hyperparameters are loaded from the YAML configuration file 'config/modeling.yml'.

    If a save_path is provided, the Linear Regression model is saved as a pickle file and the
    XGBoost model is saved using its native save method.

    :param data: The input dataset (pandas DataFrame or similar).
    :type data: pandas.DataFrame or cudf.DataFrame
    :param cutoff_year: Year to split training and test sets.
    :type cutoff_year: int
    :param n_splits: Number of splits for time series cross-validation.
    :type n_splits: int
    :param save_path: Path prefix to save the trained models.
    :type save_path: str or None
    :return: Tuple containing train_df, test_df, features, target_col, models, predictions, and metrics.
    :rtype: tuple
    """
    # Load modeling configuration from YAML file.
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'modeling.yml')
    with open(config_path, 'r') as f:
        modeling_config = yaml.safe_load(f)
    xgb_params = modeling_config.get('xgb_params', {})

    # Ensure data is a pandas DataFrame with a datetime index.
    if hasattr(data, "to_pandas"):
        data = data.to_pandas()
    data.index = pd.to_datetime(data.index)

    target_col = 'pillow_swe_corrected' if 'pillow_swe_corrected' in data.columns else 'pillow_swe'
    train_data = data[data.index.year < cutoff_year]
    test_data  = data[data.index.year >= cutoff_year]

    # Build features list: include 'pillow_swe' (as a predictor) and 'z' (if exists) and all 'daymet_' columns.
    features = ['pillow_swe']
    if 'z' in data.columns:
        features.append('z')
    features += [c for c in data.columns if c.startswith('daymet_')]
    features = [f for f in features if f not in ['pillow_swe_corrected']]

    train_df = train_data.dropna(subset=features + [target_col])
    test_df  = test_data.dropna(subset=features + [target_col])

    # Convert to cuDF for Linear Regression.
    train_df_cudf = cudf.from_pandas(train_df)
    test_df_cudf  = cudf.from_pandas(test_df)

    X_train = train_df_cudf[features]
    y_train = train_df_cudf[target_col]
    X_test  = test_df_cudf[features]
    y_test  = test_df_cudf[target_col]

    # Train cuML Linear Regression model.
    model_lin = LinearRegression()
    model_lin.fit(X_train, y_train)
    pred_lin_test = model_lin.predict(X_test)

    # Prepare pandas data for XGBoost.
    X_train_pd = train_df[features]
    y_train_pd = train_df[target_col]
    X_test_pd  = test_df[features]
    y_test_pd  = test_df[target_col]

    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_train_pd, y_train_pd)
    pred_xgb_test = model_xgb.predict(X_test_pd)

    def compute_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        return rmse, r2

    rmse_lin, r2_lin = compute_metrics(y_test.to_pandas(), pred_lin_test.to_pandas())
    rmse_xgb, r2_xgb = compute_metrics(y_test_pd, pred_xgb_test)

    metrics = {
        "lin": {"test_rmse": rmse_lin, "test_r2": r2_lin},
        "xgb": {"test_rmse": rmse_xgb, "test_r2": r2_xgb}
    }

    models = {"lin": model_lin, "xgb": model_xgb}
    predictions = {"lin": {"test": pred_lin_test},
                   "xgb": {"test": pred_xgb_test}}

    # Save the models if a save_path is provided.
    if save_path is not None:
        joblib.dump(model_lin, save_path + "_lin.pkl")
        model_xgb.save_model(save_path + "_xgb.json")

    return train_df, test_df, features, target_col, models, predictions, metrics
