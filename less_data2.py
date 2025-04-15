import math
import os
import sys
import io
import numpy as np
import pandas as pd
import streamlit as st
from bokeh.embed import components
from bokeh.events import Tap
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    DatetimeTickFormatter,
    HoverTool,
    Range1d,
    TapTool,
    WMTSTileSource,
    Toggle,
)
from bokeh.plotting import figure
from bokeh.io import show
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon, MultiPolygon
from bokeh.tile_providers import get_provider, Vendors

# helper modules
import helper.lm_model as lm_model
import helper.load_data as load_data
import helper.plotting as plotting

# Set page configuration only once
st.set_page_config(page_title="Less Data Dashboard", layout="wide")

# --- Helper Function ---
def coor_conv(df, lon="longitude", lat="latitude"):
    k = 6378137
    df["x"] = df[lon] * (k * np.pi / 180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360)) * k
    return df

# --- Tabs ---
tab_overview, tab_data, tab_train, tab_inference, tab_conclusions = st.tabs(
    ["Overview", "Data", "Training", "Inference", "Conclusions"]
)

# ========================
# Overview Tab
# ========================
with tab_overview:
    st.title("Overview")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("## Hypothesis 1 - Less Data")
        st.write(
            """***Less Data*** paradigm:
- Uses snow pillows to predict mean ASO SWE across elevational groupings.
- **Pros:** Data familiarity, interpretability, lower computational cost, flexibility with missing data.
- **Cons:** Limited training dates, potential data quality issues.
- **Assumptions:** Minimal features can capture the necessary signal without introducing noise."""
        )
    with col2:
        # Assume root_dir is one directory above this script’s folder.
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        image_fpath = os.path.join(root_dir, "images", "Overview", "less_is_more.png")
        st.image(image_fpath, width=600)
    st.write("SWE Estimator - Less Data Approach")
    st.subheader("Overview")
    st.write("Welcome to the interactive data exploration platform. Explore the dataset, examine the training process, and evaluate the model results.")


# ========================
# Data Tab
# ========================
with tab_data:
    st.title("Data")
    st.write("This section displays an interactive map overlaid with a time series chart. Select a basin below to explore the relevant snow pillow data and historic ASO flight measurements.")
    
    col_left, col_right = st.columns(2)
    with col_left:
        selected_basin = st.selectbox("Select Basin:", ("San Joaquin", "Toulumne"))
    
    # Depending on the basin selection, load the appropriate data
    if selected_basin == "San Joaquin":
        # --- Load San Joaquin data ---
        sj_pillow_df = pd.read_csv(os.path.join("data", "snow_pillows", "locations", "sj_pillow_locations.csv"))
        sj_pillow_readings_df = pd.read_csv(
            os.path.join("data", "snow_pillows", "measurements", "sj_pillow_qa_table.csv")
        ).fillna(0)
        # Format numeric values
        for col in sj_pillow_readings_df.columns:
            if col != "time":
                sj_pillow_readings_df[col] = sj_pillow_readings_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) else x
                )
        basin_gdf = gpd.read_file(os.path.join("data", "shape", "USCASJ", "USCASJ.shp")).to_crs("EPSG:3857")
        patches_data = {"xs": [], "ys": []}
        for geom in basin_gdf.geometry:
            if isinstance(geom, Polygon):
                x, y = geom.exterior.xy
                patches_data["xs"].append(list(x))
                patches_data["ys"].append(list(y))
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    patches_data["xs"].append(list(x))
                    patches_data["ys"].append(list(y))
        sj_pillow_df = coor_conv(sj_pillow_df)
        
        # --- Create map plot ---
        tile_provider = get_provider(Vendors.ESRI_IMAGERY)
        scatter = figure(
            title="SJ Snow Pillows",
            tools="tap,pan,wheel_zoom,reset,lasso_select",
            x_axis_type="mercator",
            y_axis_type="mercator",
            width=700,
            height=500,
            x_range=(-13428333, -13090833),
            y_range=(4363056, 4671115),
        )
        scatter.add_tile(tile_provider)
        scatter.patches(xs="xs", ys="ys", source=ColumnDataSource(patches_data), fill_color=None, line_color="blue", line_width=2)
        source = ColumnDataSource(data=dict(
            x=sj_pillow_df["x"],
            y=sj_pillow_df["y"],
            snow_pillow=sj_pillow_df["id"],
        ))
        # Prepare time series data dictionary
        pillow_readings = {"x": pd.to_datetime(sj_pillow_readings_df["time"].tolist())}
        for col in sj_pillow_readings_df.columns[1:]:
            pillow_readings[col] = sj_pillow_readings_df[col].tolist()
        spr = ColumnDataSource(data=pillow_readings)
        colors = ['red', 'black', 'green', 'orange', 'purple', 'gray']
        source.data["color"] = [colors[i % len(colors)] for i in range(len(sj_pillow_df))]
        scatter.circle(x="x", y="y", size=10, source=source, color="color", line_color="black", line_width=1)
        scatter.text(x="x", y="y", text="snow_pillow", source=source,
                     text_color="black", text_font_size="8pt", x_offset=5, y_offset=-2)
        
        # --- Create time series chart ---
        line_source = ColumnDataSource(data=dict())
        line_fig = figure(
            title="Line Chart for Selected Snow Pillow",
            width=700,
            height=500,
            x_axis_type="datetime",
            y_range=Range1d(start=0, end=3500),
        )
        line_fig.xaxis.formatter = DatetimeTickFormatter(days="%d %b %Y", months="%b %Y", years="%Y")
        
        # CustomJS callback (note: no nested CustomJS call now)
        callback_code = """
        const indices = cb_obj.indices;
        if (indices.length === 0) return;
        let selectedSnowPillows = [];
        let selectedData = {x: spr['x']};
        for (let i = 0; i < indices.length; i++) {
            const index = indices[i];
            const data = source.data;
            const pillow = data['snow_pillow'][index];
            selectedSnowPillows.push(pillow);
        }
        for (let i = 0; i < selectedSnowPillows.length; i++) {
            selectedData[selectedSnowPillows[i]] = spr[selectedSnowPillows[i]];
        }
        line_source.data = selectedData;
        line_source.change.emit();
        """
        select_callback = CustomJS(
            args=dict(source=source, line_source=line_source, spr=pillow_readings),
            code=callback_code,
        )
        source.selected.js_on_change("indices", select_callback)
        
        # Draw a line for each pillow
        pillow_ids = sj_pillow_df["id"].tolist()
        color_mapping = dict(zip(pillow_ids, source.data["color"]))
        for pillow in pillow_ids:
            renderer = line_fig.line(
                x="x", y=pillow, source=line_source, name=pillow, color=color_mapping[pillow], line_width=2
            )
            hover = HoverTool(
                renderers=[renderer],
                tooltips=[("Pillow ID", pillow), ("Date", "@x{%F}"), ("Units: mm", f"@{pillow}")],
                formatters={"@x": "datetime"},
                mode="mouse",
            )
            line_fig.add_tools(hover)
        
        # Load ASO flight data for SJ
        sj_aso_df = pd.read_csv(os.path.join("data", "aso", "USCASJ", "uscasj_aso_sum.csv"))
        sj_aso_df["aso_mean_bins_mm"] = sj_aso_df["aso_mean_bins_mm"].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else x
        )
        points_data = {
            "time": pd.to_datetime(sj_aso_df["time"]),
            "value": sj_aso_df["aso_mean_bins_mm"],
            "image_url": [f"data/aso/USCASJ/images/plot{i}.png" for i in range(len(sj_aso_df))],
        }
        points_source = ColumnDataSource(points_data)
        aso_circle = line_fig.circle(
            x="time", y="value", source=points_source, size=10, color="blue", line_color="black", line_width=1,
            legend_label="ASO Flights"
        )
        aso_hover = HoverTool(
            renderers=[aso_circle],
            tooltips=[("Date", "@time{%F}"), ("Value (mm)", "@value")],
            formatters={"@time": "datetime"},
            mode="mouse",
        )
        line_fig.add_tools(aso_hover)
        
        st.bokeh_chart(column(scatter, line_fig), use_container_width=True)
        flight_dates = ["Dates"] + [text.replace("-", "/") for text in sj_aso_df["time"]]
    
    else:
        # --- Tuolumne branch ---
        tm_pillow_df = pd.read_csv(os.path.join("data", "snow_pillows", "locations", "tm_pillow_locations.csv"))
        tm_pillow_readings_df = pd.read_csv(
            os.path.join("data", "snow_pillows", "measurements", "tm_pillow_qa_table.csv")
        ).fillna(0)
        for col in tm_pillow_readings_df.columns:
            if col != "time":
                tm_pillow_readings_df[col] = tm_pillow_readings_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) else x
                )
        basin_gdf = gpd.read_file(os.path.join("data", "shape", "USCATM", "USCATM.shp")).to_crs("EPSG:3857")
        patches_data = {"xs": [], "ys": []}
        for geom in basin_gdf.geometry:
            if isinstance(geom, Polygon):
                x, y = geom.exterior.xy
                patches_data["xs"].append(list(x))
                patches_data["ys"].append(list(y))
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    patches_data["xs"].append(list(x))
                    patches_data["ys"].append(list(y))
        tm_pillow_df = coor_conv(tm_pillow_df)
        
        tile_provider = get_provider(Vendors.ESRI_IMAGERY)
        scatter = figure(
            title="TM Snow Pillows",
            tools="tap,pan,wheel_zoom,reset,lasso_select",
            x_axis_type="mercator",
            y_axis_type="mercator",
            width=700,
            height=500,
            x_range=(-13428333, -13090833),
            y_range=(4363056, 4671115),
        )
        scatter.add_tile(tile_provider)
        scatter.patches(xs="xs", ys="ys", source=ColumnDataSource(patches_data), fill_color=None, line_color="red", line_width=2)
        source = ColumnDataSource(data=dict(
            x=tm_pillow_df["x"],
            y=tm_pillow_df["y"],
            snow_pillow=tm_pillow_df["id"],
        ))
        pillow_readings = {"x": pd.to_datetime(tm_pillow_readings_df["time"].tolist())}
        for col in tm_pillow_readings_df.columns[1:]:
            pillow_readings[col] = tm_pillow_readings_df[col].tolist()
        spr = ColumnDataSource(data=pillow_readings)
        colors = ['red', 'black', 'green', 'orange', 'purple', 'gray']
        source.data["color"] = [colors[i % len(colors)] for i in range(len(tm_pillow_df))]
        scatter.circle(x="x", y="y", size=10, source=source, color="color", line_color="black", line_width=1)
        scatter.text(x="x", y="y", text="snow_pillow", source=source,
                     text_color="black", text_font_size="8pt", x_offset=5, y_offset=-2)
        
        line_source = ColumnDataSource(data=dict())
        line_fig = figure(
            title="Line Chart for Selected Snow Pillow",
            width=700,
            height=500,
            x_axis_type="datetime",
            y_range=Range1d(start=0, end=3500),
        )
        line_fig.xaxis.formatter = DatetimeTickFormatter(days="%d %b %Y", months="%b %Y", years="%Y")
        
        callback_code = """
        const indices = cb_obj.indices;
        if (indices.length === 0) return;
        let selectedSnowPillows = [];
        let selectedData = {x: spr['x']};
        for (let i = 0; i < indices.length; i++) {
            const index = indices[i];
            const data = source.data;
            const pillow = data['snow_pillow'][index];
            selectedSnowPillows.push(pillow);
        }
        for (let i = 0; i < selectedSnowPillows.length; i++) {
            selectedData[selectedSnowPillows[i]] = spr[selectedSnowPillows[i]];
        }
        line_source.data = selectedData;
        line_source.change.emit();
        """
        select_callback = CustomJS(
            args=dict(source=source, line_source=line_source, spr=pillow_readings),
            code=callback_code,
        )
        source.selected.js_on_change("indices", select_callback)
        
        pillow_ids = tm_pillow_df["id"].tolist()
        color_mapping = dict(zip(pillow_ids, source.data["color"]))
        for pillow in pillow_ids:
            renderer = line_fig.line(
                x="x", y=pillow, source=line_source, name=pillow, color=color_mapping[pillow], line_width=2
            )
            hover = HoverTool(
                renderers=[renderer],
                tooltips=[("Pillow ID", pillow), ("Date", "@x{%F}"), ("Units: mm", f"@{pillow}")],
                formatters={"@x": "datetime"},
                mode="mouse",
            )
            line_fig.add_tools(hover)
            
        tm_aso_df = pd.read_csv(os.path.join("data", "aso", "USCATM", "uscatm_aso_sum.csv"))
        tm_aso_df["aso_mean_bins_mm"] = tm_aso_df["aso_mean_bins_mm"].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else x
        )
        points_data = {
            "time": pd.to_datetime(tm_aso_df["time"]),
            "value": tm_aso_df["aso_mean_bins_mm"],
            "image_url": [f"data/aso/USCATM/images/plot{i}.png" for i in range(len(tm_aso_df))]
        }
        points_source = ColumnDataSource(points_data)
        aso_circle = line_fig.circle(
            x="time", y="value", source=points_source, size=10, color="blue", line_color="black", line_width=1,
            legend_label="ASO Flights"
        )
        aso_hover = HoverTool(
            renderers=[aso_circle],
            tooltips=[("Date", "@time{%F}"), ("Value (mm)", "@value")],
            formatters={"@time": "datetime"},
            mode="mouse",
        )
        line_fig.add_tools(aso_hover)
        st.bokeh_chart(column(scatter, line_fig), use_container_width=True)
        flight_dates = ["Dates"] + [text.replace("-", "/") for text in tm_aso_df["time"]]
    
    # --- Flight Date Selection ---
    col_flight, col_image = st.columns(2)
    with col_flight:
        aso_flight_date = st.selectbox("Select ASO flight date:", flight_dates)
    if aso_flight_date != "Dates":
        folder = "USCASJ" if selected_basin == "San Joaquin" else "USCATM"
        # Adjust the index as needed (here subtract 1 because of the placeholder "Dates")
        index = flight_dates.index(aso_flight_date) - 1
        st.image(os.path.join("data", "aso", folder, "images", f"plot{index}.png"))
    else:
        st.image(os.path.join("data", "aso", "blank.png"))

# ========================
# Training Tab
# ========================
with tab_train:
    st.title("Training")
    st.write(
        """For ***Less Data*** we created multiple linear regression models with cross-validation to reduce overfitting.  
Select the options below to view each model’s training performance."""
    )
    image_folder = "images"
    basin_options_train = ["San Joaquin", "Toulumne"]
    selected_basin_train = st.selectbox("Select Basin", basin_options_train, index=0)
    elevation_options = ["<7k", "7-8k", "8-9k", "9-10k", "10-11k", "11-12k", ">12k", "Total"]
    selected_elevation = st.selectbox("Select Elevation", elevation_options, index=7)
    impute_options = ["Drop NaNs", "Predict NaNs"]
    selected_imputation = st.selectbox("Select Imputation Strategy", impute_options, index=0)
    season_options = ["Total Season", "Accumulation", "Melt"]
    selected_season = st.selectbox("Select Season Segmentation", season_options, index=0)

    # Convert selections to inputs for the model.
    aso_site_name, elev_band, isImpute, isSplit, isAccum, start_wy, end_wy = load_data.select_vals_to_inputs(
        selected_basin_train, selected_elevation, selected_imputation, selected_season
    )
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    obs_gdf, obs_data, df_sum_total, slice_df, all_pils, baseline_pils, aso_tseries_1, dem_bin = load_data.load_mlr_basin_data(
        aso_site_name, root_dir
    )
    sorted_elev, elev_cunane_position = plotting.terrain_cdf_distribution(dem_bin[-1, :, :].flatten())
    predictions_bestfit, predictions_validation, aso_tseries_2, obs_data_6, table_dict, selected_pillow, max_swe_, title_str, stations2 = lm_model.run_mlr_cross_validation(
        aso_site_name,
        root_dir,
        aso_tseries_1,
        elev_band,
        isSplit,
        isImpute,
        isAccum,
        df_sum_total,
        all_pils,
        obs_data,
        baseline_pils,
        start_wy,
        end_wy,
    )
    fig = plotting.combine_cross_validation_plots(
        aso_site_name,
        elev_band,
        isImpute,
        aso_tseries_2,
        predictions_bestfit,
        predictions_validation,
        max_swe_,
        stations2,
        dem_bin,
        obs_gdf,
        sorted_elev,
        elev_cunane_position,
        obs_data_6,
        start_wy,
        end_wy,
        table_dict,
        title_str,
    )
    st.pyplot(fig)

# ========================
# Inference Tab
# ========================
with tab_inference:
    st.title("Results and Evaluation")
    st.write(
        "Comparing model inference results in the San Joaquin basin. First we compare models for each elevation interval, then evaluate KPI metrics."
    )
    basin_options_inf = ["San Joaquin"]
    selected_basin_inf = st.selectbox("Basin", basin_options_inf, index=0)
    elevation_options_inf = ["<7k", "7-8k", "8-9k", "9-10k", "10-11k", "11-12k", ">12k", "Total"]
    selected_elevation_inf = st.selectbox("Elevation", elevation_options_inf, index=7)
    aso_site_name, selected_dict = load_data.select_vals_to_outputs(selected_basin_inf)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True, dpi=200)
    ax[1], _ = plotting.create_difference_plots_benchmark_mlr_mult_dates(
        root_dir,
        aso_site_name,
        ax[1],
        date_str="20250325",
        elevation_bin=selected_dict[selected_elevation_inf]["elevation_bin"],
        ymax_lim=selected_dict[selected_elevation_inf]["ymax_lim"],
        FirstPlot=False,
        text_adjust=selected_dict[selected_elevation_inf]["text_adjust"],
    )
    ax[0], _ = plotting.create_difference_plots_benchmark_mlr_mult_dates(
        root_dir,
        aso_site_name,
        ax[0],
        date_str="20250226",
        elevation_bin=selected_dict[selected_elevation_inf]["elevation_bin"],
        ymax_lim=selected_dict[selected_elevation_inf]["ymax_lim"],
        FirstPlot=True,
        text_adjust=selected_dict[selected_elevation_inf]["text_adjust"],
    )
    plt.suptitle(f"{selected_basin_inf} - {selected_elevation_inf} Mean SWE Comparison", fontweight="bold", fontsize=24)
    plt.tight_layout()
    st.pyplot(fig)
    st.write("## Model Comparison")
    date_options = ["2025-02-26", "2025-03-25"]
    selected_test_date = st.selectbox("Test Dates", date_options, index=1)
    fig = plotting.create_model_comparison(root_dir, aso_site_name, date_str=selected_test_date.replace("-", ""))
    st.pyplot(fig)

# ========================
# Conclusions Tab
# ========================
with tab_conclusions:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("## Hypothesis 1 - Less Data")
        st.write(
            """
Based on our final flight (2025-03-25) in the **San Joaquin** basin:
- **Drop NaNs:** Fails Accuracy KPI (within 10% of ASO), better than SNODAS, worse than UASWE.
- **Predict NaNs:** Passes Accuracy KPI, better than SNODAS, and shows promise across elevations.
            """
        )
    with col2:
        image_fpath = os.path.join(root_dir, "images", "Overview", "less_is_more.png")
        st.image(image_fpath, width=500)
    
    # Additional ASO flight image selection within conclusions.
    col3, col4 = st.columns(2)
    with col3:
        if selected_basin == "San Joaquin":
            flight_dates = [text.replace("-", "/") for text in sj_aso_df["time"]]
        else:
            flight_dates = [text.replace("-", "/") for text in tm_aso_df["time"]]
        aso_flight_date = st.selectbox("Select ASO flight date:", flight_dates)
    with col4:
        if aso_flight_date:
            folder = "USCASJ" if selected_basin == "San Joaquin" else "USCATM"
            index = flight_dates.index(aso_flight_date)
            st.image(os.path.join("data", "aso", folder, "images", f"plot{index}.png"))
        else:
            st.image(os.path.join("data", "aso", "blank.png"))
    
    st.title("Modeling & Training Validation")
    elevation_options_val = ["<7k", "7-8k", "8-9k", "9-10k", "10-11k", "11-12k", ">12k", "Total"]
    selected_elevation_val = st.selectbox("Select Elevation", elevation_options_val)
    basin_options_val = ["San Joaquin", "Toulumne"]
    selected_basin_val = st.selectbox("Select Basin", basin_options_val)
    png_path = os.path.join(
        "images",
        "USCASJ" if selected_basin_val == "San Joaquin" else "USCATM",
        "MLR",
        "7k" if selected_elevation_val == "<7k" else "12k" if selected_elevation_val == ">12k" else selected_elevation_val,
        "validation.png",
    )
    st.image(png_path)
    st.subheader("Results and Evaluation")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        option1 = st.selectbox("Select ASO flight date:", ("2025/03/25", "2025/02/26"))
    with colB:
        option2 = st.selectbox("Select Elevation:", ("<7k", "7-8k", "8-9k", "9-10k", "10-11k", "11-12k", ">12k", "Total", "Combined"))
    image_path = os.path.join("data", "MLR_Comparison", "7k" if option2 == "<7k" else "12k" if option2 == ">12k" else option2, f"{option1.replace('/', '_')}.png")
    st.image(image_path)
