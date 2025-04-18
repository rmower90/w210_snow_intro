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

from src.mo_sno.preprocessing import get_intermediate_insitu_data, get_preprocessed_dataset

warnings.filterwarnings("ignore")
pio.templates.default = 'plotly_dark'

# Elevation band binning function (ordinal)
def compute_elev_band(z_meters):
    elev_ft = z_meters * 3.28084
    bins = [7000, 8000, 9000, 10000, 11000, 12000]
    return np.digitize(elev_ft, bins)

def adjust_swe_bias_loess(pillow_swe, aso_swe, loess_frac=0.1, alpha=0.1):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    common_idx = pillow_swe.dropna().index.intersection(aso_swe.dropna().index)
    if len(common_idx) == 0:
        print("No overlapping data; returning original pillow_swe.")
        bias_corrected = pillow_swe.copy()
    else:
        bias_factor = aso_swe.loc[common_idx].mean() / pillow_swe.loc[common_idx].mean()
        print(f"Computed bias factor: {bias_factor:.3f}")
        bias_corrected = pillow_swe * bias_factor

    x = bias_corrected.index.map(pd.Timestamp.toordinal)
    smoothed = lowess(bias_corrected.values, x, frac=loess_frac, return_sorted=False)
    smoothed_series = pd.Series(smoothed, index=bias_corrected.index)
    blended = alpha * smoothed_series + (1 - alpha) * pillow_swe
    return blended

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return rmse, r2

def train_all_models(data, cutoff_year=2023, save_path=None):
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    target_col = 'pillow_swe_corrected' if 'pillow_swe_corrected' in data.columns else 'pillow_swe'

    # Drop z and z_sq if present
    data.drop(columns=[c for c in ['z', 'z_sq'] if c in data.columns], inplace=True)

    # Add elevation band as an ordinal feature
    data['elev_band'] = compute_elev_band(data['z']) if 'z' in data.columns else 0

    # Define features
    features = ['elev_band'] + [c for c in data.columns if c.startswith('daymet_') or c == 'pillow_swe']
    features = [f for f in features if f != target_col]

    # Train/test split
    train_df = data[data.index.year < cutoff_year].dropna(subset=features + [target_col])
    print(f"Training data shape: {train_df.head(50)}")
    test_df = data[data.index.year >= cutoff_year].dropna(subset=features + [target_col])

    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    model_lin = LinearRegression(positive=True)
    model_lin.fit(X_train, y_train)
    pred_lin = model_lin.predict(X_test)
    pred_lin_train = model_lin.predict(X_train)

    monotonic = [1 if f == 'elev_band' else 0 for f in features]
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.15,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 500,
        'random_state': 42,
    }
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_train, y_train)
    pred_xgb = model_xgb.predict(X_test)
    pred_xgb_train = model_xgb.predict(X_train)

    rmse_lin, r2_lin = compute_metrics(y_test, pred_lin)
    rmse_xgb, r2_xgb = compute_metrics(y_test, pred_xgb)

    metrics = {
        "lin": {"test_rmse": rmse_lin, "test_r2": r2_lin},
        "xgb": {"test_rmse": rmse_xgb, "test_r2": r2_xgb}
    }

    if save_path:
        joblib.dump(model_lin, save_path + '_lin.pkl')
        model_xgb.save_model(save_path + '_xgb.json')

    return train_df, test_df, features, target_col, {'lin': model_lin, 'xgb': model_xgb}, \
           {'lin': pred_lin, 'xgb': pred_xgb, 'lin_train': pred_lin_train, 'xgb_train': pred_xgb_train}, metrics

if __name__ == '__main__':
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(ROOT, "data", "mo_data")
    model_dir = os.path.join(ROOT, "models")
    intermediate_path = os.path.join(data_dir, "intermediate_insitu.parquet")
    insitu_gdf = get_intermediate_insitu_data(intermediate_path, overwrite=False)
    basins = ["San_Joaquin", "Tuolumne"]

    for basin in basins:
        print(f"\nProcessing and training for basin: {basin}")
        save_path = os.path.join(data_dir, f"saved_{basin}_preprocessed.parquet")
        df = get_preprocessed_dataset(basin, insitu_gdf, save_path, overwrite=True)
        df['basin'] = basin
        df['pillow_swe_corrected'] = adjust_swe_bias_loess(
            df['pillow_swe'], df['aso_swe'], loess_frac=0.021, alpha=0.9
        )
        df.to_parquet(save_path, index=True)

        print("Training models...")
        model_path = os.path.join(model_dir, f"{basin}_model")
        train_df, test_df, features, target_col, models, predictions, metrics = train_all_models(
            df, cutoff_year=2023, save_path=model_path
        )
        print("Model metrics:")
        print(metrics)