import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import traceback
from src.mo_sno.models import train_all_models, add_derived_variables, add_degree_days_and_accumulated_precip, adjust_swe_bias_loess
from src.mo_sno.preprocessing import get_intermediate_insitu_data, get_preprocessed_dataset
from src.mo_sno.plotting import (
    create_timeseries_plot_train,
    create_timeseries_plot,
    create_scatter_plot,
    plot_feature_importances
)

data_dir = "data/mo_data"
model_dir = "models"
intermediate_path = os.path.join(data_dir, "intermediate_insitu.parquet")
basins = ["San_Joaquin", "Tuolumne"]

st.set_page_config(page_title="Mo Data Dashboard", layout="wide")

FEATURE_LABELS = {
    "pillow_swe": "Modeled SWE (mm)",
    "z": "Elevation (m)",
    "daymet_dayl": "Daylength (sec)",
    "daymet_prcp": "Precipitation (mm)",
    "daymet_tmax": "Max Temperature (°C)",
    "daymet_tmin": "Min Temperature (°C)",
    "daymet_mean_temp": "Mean Temperature (°C)",
    "daymet_snowfall_index": "Snowfall Index",
    "daymet_daily_degree_day": "Daily Degree Day",
    "daymet_degree_days_7d": "7-day Degree Days",
    "daymet_accumulated_precip_7d": "Accumulated Precip (7-day, mm)"
}

DEFAULT_VALUES = {k: 0.0 for k in FEATURE_LABELS}
DEFAULT_VALUES.update({
    "pillow_swe": 490.64,
    "z": 2931.0,
    "daymet_dayl": 34120.93,
    "daymet_tmax": -1.27,
    "daymet_tmin": -8.93,
    "daymet_mean_temp": -5.10,
    "daymet_degree_days_7d": 10.35,
    "daymet_accumulated_precip_7d": 223.70
})

desired_dtypes = {
    "pillow_swe": "float64",
    "z": "float64",
    "daymet_dayl": "float32",
    "daymet_prcp": "float32",
    "daymet_tmax": "float32",
    "daymet_tmin": "float32",
    "daymet_mean_temp": "float32",
    "daymet_snowfall_index": "float64",
    "daymet_daily_degree_day": "float64",
    "daymet_degree_days_7d": "float64",
    "daymet_accumulated_precip_7d": "float64"
}

# Tabs
tab_train, tab_predict, tab_analytics = st.tabs(["Model Training", "Model Prediction", "Analytics"])

# ----------------------------- TAB 1: TRAINING -----------------------------
with tab_train:
    st.title("Model Training")
    st.write("Train Linear Regression and XGBoost models for each basin.")

    if st.button("🚀 Run Training Pipeline"):
        with st.spinner("📦 Loading and preprocessing data..."):
            insitu_gdf = get_intermediate_insitu_data(intermediate_path, overwrite=False)

            for basin in basins:
                st.subheader(f"🔄 {basin}")
                df_path = os.path.join(data_dir, f"saved_{basin}_preprocessed.parquet")
                df = get_preprocessed_dataset(basin, insitu_gdf, df_path, overwrite=False)
                df = add_derived_variables(df)
                df = add_degree_days_and_accumulated_precip(df)
                df['pillow_swe_corrected'] = adjust_swe_bias_loess(
                    df['pillow_swe'], df['aso_swe'], loess_frac=0.021, alpha=0.9
                )
                df.to_parquet(df_path, index=True)

                model_path = os.path.join(model_dir, f"{basin}_model")
                train_df, test_df, features, target_col, models, predictions, metrics = train_all_models(
                    df, cutoff_year=2023, save_path=model_path
                )

                st.plotly_chart(
                    plot_feature_importances(models['xgb'], features, f"{basin} Feature Importance - XGBoost"),
                    use_container_width=True
                )
                st.plotly_chart(
                    plot_feature_importances(models['lin'], features, f"{basin} Feature Importance - Linear Regression"),
                    use_container_width=True
                )
                st.plotly_chart(
                    create_timeseries_plot_train(
                        train_df,
                        predictions['lin_train'],
                        predictions['xgb_train'],
                        target_col,
                        basin
                    ),
                    use_container_width=True
                )
                st.plotly_chart(
                    create_timeseries_plot(test_df, predictions['lin'], predictions['xgb'], target_col, basin),
                    use_container_width=True
                )
                st.plotly_chart(
                    create_scatter_plot(test_df, predictions['lin'], predictions['xgb'], target_col, basin),
                    use_container_width=True
                )

# ----------------------------- TAB 2: PREDICTION -----------------------------
with tab_predict:
    st.title("Mo Data Prediction")
    selected_basin = st.selectbox("Select Basin", options=basins)

    st.header("Enter Feature Values")
    cols = st.columns(3)
    feature_values = {}
    for i, (var, label) in enumerate(FEATURE_LABELS.items()):
        col = cols[i % 3]
        feature_values[var] = col.number_input(
            label=label,
            value=float(DEFAULT_VALUES[var]),
            step=0.1,
            format="%.2f",
            key=var
        )

    st.header("Entered Feature Values")
    display_df = pd.DataFrame({
        "Feature": list(FEATURE_LABELS.values()),
        "Value": [float(feature_values[k]) for k in FEATURE_LABELS.keys()]
    })
    st.dataframe(display_df, use_container_width=True)

    if st.button("Run Model"):
        input_df = pd.DataFrame([feature_values]).astype(desired_dtypes)

        try:
            model_lin_path = f"models/{selected_basin}_model_lin.pkl"
            model_lin = joblib.load(model_lin_path)

            expected_features = model_lin.feature_names_in_
            aligned_input = input_df.reindex(columns=expected_features, fill_value=0.0)

            if not (aligned_input.columns == expected_features).all():
                st.warning("⚠️ Feature columns are misaligned. Predictions may be unreliable.")

            pred_lin = model_lin.predict(aligned_input)
            st.success(f"Linear Regression Prediction: {pred_lin[0]:.2f}")

            st.subheader("🔢 Linear Model Coefficients")
            coeff_df = pd.DataFrame({
                "Feature": expected_features,
                "Coefficient": model_lin.coef_
            })
            st.dataframe(coeff_df)

        except Exception as e:
            st.error(f"Linear model failed: {e}")
            st.exception(traceback.format_exc())

        try:
            model_xgb_path = f"models/{selected_basin}_model_xgb.json"
            model_xgb = xgb.XGBRegressor()
            model_xgb.load_model(model_xgb_path)
            aligned_input_xgb = input_df.reindex(columns=model_xgb.get_booster().feature_names, fill_value=0.0)
            pred_xgb = model_xgb.predict(aligned_input_xgb)
            st.success(f"XGBoost Prediction: {pred_xgb[0]:.2f}")
        except Exception as e:
            st.error(f"XGBoost model failed: {e}")

# ----------------------------- TAB 3: ANALYTICS -----------------------------
with tab_analytics:
    st.title("Analytics (Coming Soon)")
    st.info("This will contain project summary stats, basin comparisons, and export options.")
