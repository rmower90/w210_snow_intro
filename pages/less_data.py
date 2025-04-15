import os
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, DatetimeTickFormatter, HoverTool, Range1d
from bokeh.layouts import column
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models import WMTSTileSource
from bokeh.embed import components
import pandas as pd
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import io
from shapely.geometry import Polygon, MultiPolygon

# Import helper scripts
import src.helper.lm_model as lm_model
import src.helper.load_data as load_data
import src.helper.plotting as plotting

st.set_page_config(page_title="Less Data Dashboard", layout="wide")

# Tabs
tab_overview, tab_data, tab_train, tab_inference, tab_conclusions = st.tabs(
    ["Overview", "Data", "Training", "Results", "Conclusions"]
)

# Directories: get paths relative to project root.
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, ".."))

##############################
# Section 1 - Overview Tab
##############################
with tab_overview:
    st.title("Overview")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("## Hypothesis 1 - Less Data")
        st.write("***Less Data*** paradigm:")
        st.write(
            """
            - Uses snow pillows to predict mean ASO SWE across elevational groupings.
            - Pros:
                - Data familiarity.
                - Interpretability.
                - Not computationally expensive.
                - Flexibility based on missingness and pillow data quality.
            - Cons:
                - Does not optimize all relevant information.
                - Training is limited to dates of ASO flights.
                - Snow pillow limitations and data quality issues.
            - Assumptions:
                - Aggregate ASO mean SWE is most accurate estimate to the "truth".
                - Minimal features are capable of making accurate predictions.
                - More features introduces noise, data quality concerns, and less accurate predictions.
            """
        )
    with col2:
        image_fpath = os.path.join(root_dir, "images", "Overview", "less_is_more.png")
        st.image(image_fpath, width=600)

##############################
# Section 2 - Data Tab
##############################
with tab_data:
    st.title("Data")
    st.write(
        "Use the interactive chart below displaying relevant snow pillows based on the selected basin. Selecting a specific snow pillow reveals its measurements on the accompanying time series chart. The chart is overlaid with historic ASO flight measurements, illustrating relationships between snow pillow data and ASO flight data."
    )

    col3, col4 = st.columns(2)
    with col3:
        selected_1 = st.selectbox(
            'Select ASO flight date:',
            ('San Joaquin', 'Toulumne')
        )

    if selected_1 == 'San Joaquin':
        # Load San Joaquin data
        sj_pillow_df = pd.read_csv('data/snow_pillows/locations/sj_pillow_locations.csv')
        sj_pillow_readings_df = pd.read_csv('data/snow_pillows/measurements/sj_pillow_qa_table.csv')
        sj_pillow_readings_df = sj_pillow_readings_df.fillna(0)
        basin_gdf = gpd.read_file('data/shape/USCASJ/USCASJ.shp').to_crs('EPSG:3857')
        
        # Flatten geometries and extract xs and ys.
        patches_data = {'xs': [], 'ys': []}
        for geom in basin_gdf.geometry:
            if isinstance(geom, Polygon):
                x, y = geom.exterior.xy
                patches_data['xs'].append(list(x))
                patches_data['ys'].append(list(y))
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    patches_data['xs'].append(list(x))
                    patches_data['ys'].append(list(y))
        
        # Format readings columns (skip time column)
        for col in sj_pillow_readings_df.columns:
            if col != 'time':
                sj_pillow_readings_df[col] = sj_pillow_readings_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
        
        # Coordinate conversion function
        def coor_conv(df, lon="longitude", lat="latitude"):
            k = 6378137
            df["x"] = df[lon] * (k * np.pi/180.0)
            df["y"] = np.log(np.tan((90 + df[lat]) * np.pi /360)) * k
            return
        
        coor_conv(sj_pillow_df)

        # Create map plot (scatter)
        scatter = figure(
            title="SJ Snow Pillows", tools="tap,pan,wheel_zoom,reset,lasso_select",
            x_axis_type="mercator", y_axis_type="mercator",
            width=700, height=500, x_range=(-13428333, -13090833), y_range=(4363056, 4671115)
        )
        # Tile for map plot (using ESRI imagery)
        tile_provider = get_provider(Vendors.ESRI_IMAGERY)
        scatter.add_tile(tile_provider)
        # Add basin polygon layer
        scatter.patches(xs='xs', ys='ys', source=ColumnDataSource(patches_data),
                        fill_color=None, line_color='blue', line_width=2)
        
        # Data source for snow pillows
        source = ColumnDataSource(data=dict(
            x=sj_pillow_df["x"],
            y=sj_pillow_df["y"],
            snow_pillow=sj_pillow_df['id']
        ))
        # Add random colors to each point.
        colors = ['red', 'black', 'green', 'orange', 'purple', 'gray']
        source.data['color'] = [colors[i % len(colors)] for i in range(len(sj_pillow_df))]
        scatter.circle(x='x', y='y', size=10, source=source, color='color',
                       line_color='black', line_width=1)
        # Add text labels
        text_glyph = scatter.text(x='x', y='y', text='snow_pillow', source=source,
                                  text_color='black', text_font_size='8pt', x_offset=5, y_offset=-2)
        text_glyph.selection_glyph = None
        text_glyph.nonselection_glyph = None
        text_glyph.muted_glyph = None

        # Create pillow readings dictionary.
        pillow_readings = {'x': pd.to_datetime(sj_pillow_readings_df["time"].tolist())}
        for col in sj_pillow_readings_df.columns[1:]:
            pillow_readings[col] = sj_pillow_readings_df[col].tolist()
        # Create a ColumnDataSource for pillow readings.
        spr = ColumnDataSource(data=pillow_readings)

        # Initialize line plot data source with expected keys.
        init_data = {'x': pillow_readings['x']}
        for pillow_id in sj_pillow_df['id']:
            init_data[pillow_id] = [np.nan] * len(pillow_readings['x'])
        line_source = ColumnDataSource(data=init_data)

        # Create line plot figure.
        line = figure(
            title="Line Chart for Selected Snow Pillow", width=700, height=500,
            x_axis_type='datetime', y_range=Range1d(start=0, end=3500)
        )
        line.xaxis.formatter = DatetimeTickFormatter(
            days="%d %b %Y", months="%b %Y", years="%Y"
        )

        # CustomJS callback to update line_source on selection.
        select_SnowPillow = CustomJS(args=dict(source=source, line_source=line_source, spr=spr), code="""
            const indices = cb_obj.indices;
            if (indices.length === 0) return;
            let selectedData = { x: spr.data['x'] };
            let selectedSnowPillows = [];
            for (let i = 0; i < indices.length; i++) {
                let index = indices[i];
                let pillowName = source.data['snow_pillow'][index];
                selectedSnowPillows.push(pillowName);
            }
            for (let i = 0; i < selectedSnowPillows.length; i++) {
                selectedData[selectedSnowPillows[i]] = spr.data[selectedSnowPillows[i]];
            }
            line_source.data = selectedData;
            line_source.change.emit();
        """)
        source.selected.js_on_change("indices", select_SnowPillow)

        # Create time series line plots for each pillow.
        lines = {}
        for name in sj_pillow_df['id']:
            lines[name] = line.line('x', name, source=line_source,
                                    name=name, color=source.data['color'][list(sj_pillow_df['id']).index(name)],
                                    line_width=2)
        # Add Hover tool for each line.
        for name, renderer in lines.items():
            hover = HoverTool(
                renderers=[renderer],
                tooltips=[
                    ("Pillow ID", name),
                    ("Date", "@x{%F}"),
                    ("Units: mm", f"@{name}")
                ],
                formatters={'@x': 'datetime'},
                mode='mouse'
            )
            line.add_tools(hover)

        # Load ASO flight data.
        sj_aso_df = pd.read_csv('data/aso/USCASJ/uscasj_aso_sum.csv')
        sj_aso_df['aso_mean_bins_mm'] = sj_aso_df['aso_mean_bins_mm'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
        points_data = {
            'time': pd.to_datetime(sj_aso_df['time']),
            'value': sj_aso_df['aso_mean_bins_mm'],
            'image_url': [f'data/aso/USCASJ/images/plot{i}.png' for i in range(len(sj_aso_df))]
        }
        points_source = ColumnDataSource(points_data)
        # Add hover for ASO flights.
        aso_hover = HoverTool(
            renderers=[line.circle('time', 'value', source=points_source, size=10,
                                     color='blue', line_color='black', line_width=1, legend_label='ASO Flights')],
            tooltips=[("Date", "@time{%F}"), ("Value (mm)", "@value")],
            formatters={'@time': 'datetime'},
            mode='mouse'
        )
        line.add_tools(aso_hover)

        # Display plots.
        col5, col6 = st.columns(2)
        with col5:
            st.bokeh_chart(column(scatter, line), use_container_width=False)

    else:
        # Load Tuolumne data (similar structure as San Joaquin)
        tm_pillow_df = pd.read_csv('data/snow_pillows/locations/tm_pillow_locations.csv')
        tm_pillow_readings_df = pd.read_csv('data/snow_pillows/measurements/tm_pillow_qa_table.csv')
        tm_pillow_readings_df = tm_pillow_readings_df.fillna(0)
        basin_gdf = gpd.read_file('data/shape/USCATM/USCATM.shp').to_crs('EPSG:3857')
        
        patches_data = {'xs': [], 'ys': []}
        for geom in basin_gdf.geometry:
            if isinstance(geom, Polygon):
                x, y = geom.exterior.xy
                patches_data['xs'].append(list(x))
                patches_data['ys'].append(list(y))
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    patches_data['xs'].append(list(x))
                    patches_data['ys'].append(list(y))
        
        for col in tm_pillow_readings_df.columns:
            if col != 'time':
                tm_pillow_readings_df[col] = tm_pillow_readings_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
        
        def coor_conv(df, lon="longitude", lat="latitude"):
            k = 6378137
            df["x"] = df[lon] * (k * np.pi/180.0)
            df["y"] = np.log(np.tan((90 + df[lat]) * np.pi /360)) * k
            return
        
        coor_conv(tm_pillow_df)

        scatter = figure(
            title="TM Snow Pillows", tools="tap,pan,wheel_zoom,reset,lasso_select",
            x_axis_type="mercator", y_axis_type="mercator",
            width=700, height=500, x_range=(-13428333, -13090833), y_range=(4363056, 4671115)
        )
        tile_provider = get_provider(Vendors.ESRI_IMAGERY)
        scatter.add_tile(tile_provider)
        scatter.patches(xs='xs', ys='ys', source=ColumnDataSource(patches_data),
                        fill_color=None, line_color='red', line_width=2)
        
        source = ColumnDataSource(data=dict(
            x=tm_pillow_df["x"],
            y=tm_pillow_df["y"],
            snow_pillow=tm_pillow_df['id']
        ))
        colors = ['red', 'black', 'green', 'orange', 'purple', 'gray']
        source.data['color'] = [colors[i % len(colors)] for i in range(len(tm_pillow_df))]
        scatter.circle(x='x', y='y', size=10, source=source, color='color',
                       line_color='black', line_width=1)
        text_glyph = scatter.text(x='x', y='y', text='snow_pillow', source=source,
                                  text_color='black', text_font_size='8pt', x_offset=5, y_offset=-2)
        text_glyph.selection_glyph = None
        text_glyph.nonselection_glyph = None
        text_glyph.muted_glyph = None

        pillow_readings = {'x': pd.to_datetime(tm_pillow_readings_df["time"].tolist())}
        for col in tm_pillow_readings_df.columns[1:]:
            pillow_readings[col] = tm_pillow_readings_df[col].tolist()
        spr = ColumnDataSource(data=pillow_readings)

        init_data = {'x': pillow_readings['x']}
        for pillow_id in tm_pillow_df['id']:
            init_data[pillow_id] = [np.nan] * len(pillow_readings['x'])
        line_source = ColumnDataSource(data=init_data)

        line = figure(
            title="Line Chart for Selected Snow Pillow", width=700, height=500,
            x_axis_type='datetime', y_range=Range1d(start=0, end=3500)
        )
        line.xaxis.formatter = DatetimeTickFormatter(
            days="%d %b %Y", months="%b %Y", years="%Y"
        )

        select_SnowPillow = CustomJS(args=dict(source=source, line_source=line_source, spr=spr), code="""
            const indices = cb_obj.indices;
            if (indices.length === 0) return;
            let selectedData = { x: spr.data['x'] };
            let selectedSnowPillows = [];
            for (let i = 0; i < indices.length; i++) {
                let index = indices[i];
                let pillowName = source.data['snow_pillow'][index];
                selectedSnowPillows.push(pillowName);
            }
            for (let i = 0; i < selectedSnowPillows.length; i++) {
                selectedData[selectedSnowPillows[i]] = spr.data[selectedSnowPillows[i]];
            }
            line_source.data = selectedData;
            line_source.change.emit();
        """)
        source.selected.js_on_change("indices", select_SnowPillow)

        lines = {}
        for name in tm_pillow_df['id']:
            lines[name] = line.line('x', name, source=line_source,
                                    name=name, color=source.data['color'][list(tm_pillow_df['id']).index(name)],
                                    line_width=2)
        for name, renderer in lines.items():
            hover = HoverTool(
                renderers=[renderer],
                tooltips=[
                    ("Pillow ID", name),
                    ("Date", "@x{%F}"),
                    ("Units: mm", f"@{name}")
                ],
                formatters={'@x': 'datetime'},
                mode='mouse'
            )
            line.add_tools(hover)

        tm_aso_df = pd.read_csv('data/aso/USCATM/uscatm_aso_sum.csv')
        tm_aso_df['aso_mean_bins_mm'] = tm_aso_df['aso_mean_bins_mm'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
        points_data = {
            'time': pd.to_datetime(tm_aso_df['time']),
            'value': tm_aso_df['aso_mean_bins_mm'],
            'image_url': [f'data/aso/USCATM/images/plot{i}.png' for i in range(len(tm_aso_df))]
        }
        points_source = ColumnDataSource(points_data)
        aso_hover = HoverTool(
            renderers=[line.circle('time', 'value', source=points_source, size=10,
                                     color='blue', line_color='black', line_width=1, legend_label='ASO Flights')],
            tooltips=[("Date", "@time{%F}"), ("Value (mm)", "@value")],
            formatters={'@time': 'datetime'},
            mode='mouse'
        )
        line.add_tools(aso_hover)

        st.bokeh_chart(column(scatter, line), use_container_width=False)

    st.write("Select a date to visualize a LiDAR derived map from a historic flight map.")

    col5, col6 = st.columns(2)
    with col5:
        flight_dates = ["Dates"]
        if selected_1 == 'San Joaquin':
            sj_flights = [text.replace("-", "/") for text in sj_aso_df['time']]
            flight_dates.extend(sj_flights)
        else:
            tm_flights = [text.replace("-", "/") for text in tm_aso_df['time']]
            flight_dates.extend(tm_flights)
        aso_flight_date = st.selectbox('Select ASO flight date:', flight_dates)
    if aso_flight_date != "Dates":
        basin_code = "USCASJ" if selected_1 == "San Joaquin" else "USCATM"
        # Adjust index (-1) because flight_dates[0] is "Dates"
        st.image(f'data/aso/{basin_code}/images/plot{flight_dates.index(aso_flight_date)-1}.png')
    else:
        st.image('data/aso/blank.png')

##############################
# Section 3 - Training Data
##############################
with tab_train:
    st.title("Training")
    st.write(
        """
        For ***Less Data*** we created multiple linear regression models using a cross-validation approach to reduce overfitting and optimize the selected pillows used in each model. The user can make the selections below based on **Basin**, **Elevation Interval**, **Imputation Strategy**, and **Season Segmentation** to interpret each model's training performance.
        """
    )

    image_folder = "images"
    basin_options = ['San Joaquin', 'Toulumne']
    selected_basin = st.selectbox('Select Basin', basin_options, index=0)
    elevation_options = ['<7k','7-8k','8-9k','9-10k','10-11k','11-12k','>12k','Total']
    selected_elevation = st.selectbox('Select Elevation', elevation_options, index=7)
    impute_options = ['Drop NaNs','Predict NaNs']
    selected_imputation = st.selectbox('Select Imputation Strategy', impute_options, index=0)
    meltSeason_options = ['Total Season','Accumulation','Melt']
    selected_season = st.selectbox('Select Season Segmentation', meltSeason_options, index=0)

    aso_site_name, elev_band, isImpute, isSplit, isAccum, start_wy, end_wy = load_data.select_vals_to_inputs(
        selected_basin, selected_elevation, selected_imputation, selected_season
    )
    obs_gdf, obs_data, df_sum_total, slice_df, all_pils, baseline_pils, aso_tseries_1, dem_bin = load_data.load_mlr_basin_data(aso_site_name, root_dir)
    sorted_elev, elev_cunane_position = plotting.terrain_cdf_distribution(dem_bin[-1, :, :].values.flatten())
    predictions_bestfit, predictions_validation, aso_tseries_2, obs_data_6, table_dict, selected_pillow, max_swe_, title_str, stations2 = lm_model.run_mlr_cross_validation(
        aso_site_name, root_dir, aso_tseries_1, elev_band, isSplit, isImpute, isAccum, df_sum_total, all_pils, obs_data, baseline_pils, start_wy, end_wy
    )
    fig = plotting.combine_cross_validation_plots(
        aso_site_name, elev_band, isImpute, aso_tseries_2, predictions_bestfit, predictions_validation,
        max_swe_, stations2, dem_bin, obs_gdf, sorted_elev, elev_cunane_position, obs_data_6,
        start_wy, end_wy, table_dict, title_str
    )
    st.pyplot(fig)

##############################
# Section 4 - Inference/Results
##############################
with tab_inference:
    st.title("Results and Evaluation")
    st.write(
        "Here we are comparing model inference results in the San Joaquin. We compare the best ***MLR Drop NaNs*** and ***Predict NaNs*** models to our label (***ASO***) and benchmarks (***SNODAS*** and ***UASWE***). The flights occurred on February 26th, 2025 and March 25th, 2025. Future work will extend a similar comparison to the Tuolumne basin."
    )
    st.write("- First, we compare the models for each elevation interval, ***separately***.")
    st.write("- Next, we compare the models across elevation intervals and generate statistics relevant to our ***KPI*** metrics.")
    st.write("## Elevation Comparison")

    basin_options_2 = ['San Joaquin']
    selected_basin_2 = st.selectbox('Basin', basin_options_2, index=0)
    elevation_options_2 = ['<7k','7-8k','8-9k','9-10k','10-11k','11-12k','>12k','Total']
    selected_elevation_2 = st.selectbox('Elevation', elevation_options_2, index=7)
    aso_site_name, selected_dict = load_data.select_vals_to_outputs(selected_basin_2)
    fig, ax = plt.subplots(1, 2, figsize=(10,5), sharey=True, dpi=200)
    ax[1], _ = plotting.create_difference_plots_benchmark_mlr_mult_dates(
        root_dir, aso_site_name, ax[1], date_str='20250325',
        elevation_bin=selected_dict[selected_elevation_2]['elevation_bin'],
        ymax_lim=selected_dict[selected_elevation_2]['ymax_lim'],
        FirstPlot=False, text_adjust=selected_dict[selected_elevation_2]['text_adjust']
    )
    ax[0], _ = plotting.create_difference_plots_benchmark_mlr_mult_dates(
        root_dir, aso_site_name, ax[0], date_str='20250226',
        elevation_bin=selected_dict[selected_elevation_2]['elevation_bin'],
        ymax_lim=selected_dict[selected_elevation_2]['ymax_lim'],
        FirstPlot=True, text_adjust=selected_dict[selected_elevation_2]['text_adjust']
    )
    plt.suptitle(f'{selected_basin_2} - {selected_elevation_2} Mean SWE Comparison', fontweight='bold', fontsize=24)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("## Model Comparison")
    date_options = ['2025-02-26','2025-03-25']
    selected_test_date = st.selectbox('Test Dates', date_options, index=1)
    fig = plotting.create_model_comparison(root_dir, aso_site_name, date_str=selected_test_date.replace('-', ''))
    st.pyplot(fig)

##############################
# Section 5 - Conclusions Tab
##############################
with tab_conclusions:
    st.write("Conclusions")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("## Hypothesis 1 - Less Data")
        st.write(
            """
            Based on the results of our models for the final flight (2025-03-25) in the ***San Joaquin***:
            - Drop NaNs
                - Fails Accuracy KPI (within 10% of ASO for Total Mean SWE).
                - Better than SNODAS.
                - Worse than UASWE.
            - Predict NaNs
                - Passes Accuracy KPI (within 10% of ASO for Total Mean SWE).
                - Better than SNODAS.
                - Worse than UASWE for Total Accuracy. ***However***, shows promise of performing better across elevational gradients.
            """
        )
    with col2:
        image_fpath = os.path.join(root_dir, "images", "Overview", "less_is_more.png")
        st.image(image_fpath, width=500)
