# ------------------ Streamlit App ------------------

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import traceback

from src.mo_sno.models import train_all_models, adjust_swe_bias_loess
from src.mo_sno.preprocessing import get_intermediate_insitu_data, get_preprocessed_dataset
from src.mo_sno.plotting import (
    create_timeseries_plot_train,
    create_timeseries_plot,
    create_scatter_plot,
    plot_feature_importances
)

# --- Feature labels ---
FEATURE_LABELS = {
    "pillow_swe": "Modeled SWE (mm)",
    "daymet_dayl": "Daylength (sec)",
    "daymet_prcp": "Precipitation (mm)",
    "daymet_tmax": "Max Temperature (°C)",
    "daymet_tmin": "Min Temperature (°C)",
    "elev_band": "Elevation Band (Ordinal)"
}

data_dir = "data/mo_data"
model_dir = "models"
basin_acreages = {
    "San_Joaquin": 10163200,
    "Tuolumne": 1253120
}
basins = list(basin_acreages.keys())

elevation_band_labels = [
    "< 7000 ft", "7000–8000 ft", "8000–9000 ft", "9000–10000 ft",
    "10000–11000 ft", "11000–12000 ft", "> 12000 ft"
]

st.set_page_config(page_title="Mo Data Dashboard", layout="wide")
tab_train, tab_predict, tab_analytics = st.tabs(["Model Training", "Model Prediction", "Analytics"])

# ----------------------------- TAB 1: TRAINING -----------------------------
with tab_train:
    st.title("Model Training")
    st.write("Train Linear Regression and XGBoost models for each basin.")
    if st.button("🚀 Run Training Pipeline"):
        with st.spinner("Loading and preprocessing data..."):
            insitu_gdf = get_intermediate_insitu_data(
                os.path.join(data_dir, "intermediate_insitu.parquet"), overwrite=False)
            for basin in basins:
                st.subheader(f"Basin: {basin}")
                df_path = os.path.join(data_dir, f"saved_{basin}_preprocessed.parquet")
                df = get_preprocessed_dataset(basin, insitu_gdf, df_path, overwrite=True)
                df['pillow_swe_corrected'] = adjust_swe_bias_loess(
                    df['pillow_swe'], df['aso_swe'], loess_frac=0.021, alpha=0.9
                )
                df.to_parquet(df_path, index=True)
                model_path = os.path.join(model_dir, f"{basin}_model")
                train_df, test_df, features, target_col, models, predictions, metrics = train_all_models(
                    df, cutoff_year=2024, save_path=model_path
                )
                st.write("Model Metrics:")
                st.write(metrics)
                st.plotly_chart(
                    plot_feature_importances(models['xgb'], features, f"{basin} - XGBoost Feature Importance"),
                    use_container_width=True
                )
                st.plotly_chart(
                    plot_feature_importances(models['lin'], features, f"{basin} - Linear Regression Feature Importance"),
                    use_container_width=True
                )
                st.plotly_chart(
                    create_timeseries_plot_train(train_df, predictions['lin_train'],
                                                 predictions['xgb_train'], target_col, basin),
                    use_container_width=True
                )
                st.plotly_chart(
                    create_timeseries_plot(test_df, predictions['lin'],
                                             predictions['xgb'], target_col, basin),
                    use_container_width=True
                )
                st.plotly_chart(
                    create_scatter_plot(test_df, predictions['lin'],
                                        predictions['xgb'], target_col, basin),
                    use_container_width=True
                )

# ----------------------------- TAB 2: PREDICTION -----------------------------
with tab_predict:
    st.title("Model Prediction")
    selected_basin = st.selectbox("Select Basin", options=basins)

    st.header("Select Elevation Band")
    selected_band = st.selectbox("Elevation Band (ft)", options=elevation_band_labels)
    elev_band_index = elevation_band_labels.index(selected_band)

    st.header("Enter Feature Values")
    col1, col2, col3 = st.columns(3)
    pillow_swe = col1.number_input("pillow_swe (Modeled SWE, mm)", value=439.58)
    daymet_dayl = col2.number_input("daymet_dayl (Daylength, sec)", value=34120.93)
    daymet_prcp = col3.number_input("daymet_prcp (Precipitation, mm)", value=381.0)

    col4, col5 = st.columns(2)
    daymet_tmax = col4.number_input("daymet_tmax (Max Temp, °C)", value=22.22)
    daymet_tmin = col5.number_input("daymet_tmin (Min Temp, °C)", value=6.66)

    input_data = {
        "pillow_swe": [pillow_swe],
        "elev_band": [elev_band_index],
        "daymet_dayl": [daymet_dayl],
        "daymet_prcp": [daymet_prcp],
        "daymet_tmax": [daymet_tmax],
        "daymet_tmin": [daymet_tmin]
    }
    input_df = pd.DataFrame(input_data)
    st.write("Input Data:")
    st.dataframe(input_df)

    conversion_factor = 0.00328084

    if st.button("Run Model"):
        try:
            lin_model_path = os.path.join(model_dir, f"{selected_basin}_model_lin.pkl")
            model_lin = joblib.load(lin_model_path)
            expected_features = model_lin.feature_names_in_
            aligned_input = input_df.reindex(columns=expected_features, fill_value=0.0)
            pred_lin = model_lin.predict(aligned_input)
            total_volume_lin = pred_lin[0] * conversion_factor * basin_acreages[selected_basin]
            st.success(f"Linear Regression Prediction: {pred_lin[0]:.2f} mm SWE\n"
                       f"Equivalent Volume: {total_volume_lin:,.0f} acre-feet")
        except Exception as e:
            st.error(f"Linear model failed: {e}")
            st.exception(traceback.format_exc())

        try:
            xgb_model_path = os.path.join(model_dir, f"{selected_basin}_model_xgb.json")
            booster = xgb.Booster()
            booster.load_model(xgb_model_path)
            booster_feature_names = booster.feature_names
            if booster_feature_names is None:
                booster_feature_names = list(input_df.columns)
            aligned_input_xgb = input_df.reindex(columns=booster_feature_names, fill_value=0.0)
            dmat = xgb.DMatrix(aligned_input_xgb, feature_names=booster_feature_names)
            pred_xgb = booster.predict(dmat)
            total_volume_xgb = pred_xgb[0] * conversion_factor * basin_acreages[selected_basin]
            st.success(f"XGBoost Prediction: {pred_xgb[0]:.2f} mm SWE\n"
                       f"Equivalent Volume: {total_volume_xgb:,.0f} acre-feet")
        except Exception as e:
            st.error(f"XGBoost model failed: {e}")
            st.exception(traceback.format_exc())

# ----------------------------- TAB 3: ANALYTICS -----------------------------
with tab_analytics:
    st.title("Analytics (Coming Soon)")
    st.info("This section will include project summary stats, basin comparisons, and export options.")
