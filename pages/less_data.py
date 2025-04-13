import math
import os

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
)
from bokeh.plotting import figure, show
from bokeh.io import show
from bokeh.models import Toggle
import sys
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import io
from shapely.geometry import Polygon, MultiPolygon

# import helper scripts
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import lm_model as lm_model
import load_data as load_data
import plotting as plotting

st.set_page_config(page_title="Less Data Dashboard", layout="wide")

# Tabs
tab_overview, tab_data, tab_train, tab_inference,tab_conclusions = st.tabs(["Overview", "Data", "Training","Inference","Conclusions"])

# directories.
# Get path relative to the project root
base_dir = os.path.dirname(os.path.abspath(__file__))  # /.../pages
root_dir = os.path.abspath(os.path.join(base_dir, "..")) # /.../ (one level up)
from bokeh.tile_providers import get_provider, Vendors

st.set_page_config(page_title="Less Data Dashboard", layout="wide")
## Section 1 - EDA Less Data Overview & Motivation
# st.title("SWE Estimator - Less Data Approach")
# st.write("# Overview")
# st.write("This page explores the data and models for the ***Hypothesis 1 - Less Data***")
# st.write("Welcome to the interactive data exploration platform. Below are dynamic charts and intuitive menus designed for delving into the dataset. Explore the intricacies of the model training process and evaluate the results with ease. Gain insights and a deeper understanding of the analysis.")

with tab_overview:
    st.title("Overview")

    col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

    with col1:
        st.markdown("## Hypothesis 1 - Less Data")
        st.write("***Less Data*** paradigm:")
        st.write("""
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
                """)

    with col2:
        image_fpath = os.path.join(root_dir, "images", "Overview", "less_is_more.png")
        st.image(image_fpath, width=600)

## Section 2 - Basin Data Exploration
with tab_data:
    st.title("Data")
    # st.write("Basin Data Exploration")
    st.write("Use the interactive chart below displaying relevant snow pillows dependant on the selected basin. Selecting a specific snow pillow reveals its measurements on the accompanying time series chart. The chart is overlaid with historic ASO flight measurements, illustrating relationships between snow pillow data and ASO flight data.")

    col3, col4 = st.columns(2)
    # Add a selectbox to each column
    with col3:
        selected_1 = st.selectbox(
            'Select ASO flight date:',
            ('San Joaquin', 'Toulumne')
        )
    if selected_1 == 'San Joaquin':
        # Load snow pillow data from the CSV file
        sj_pillow_df = pd.read_csv('data/snow_pillows/locations/sj_pillow_locations.csv')
        sj_pillow_readings_df = pd.read_csv('data/snow_pillows/measurements/sj_pillow_qa_table.csv')
        sj_pillow_readings_df = sj_pillow_readings_df.fillna(0)
        basin_gdf = gpd.read_file('data/shape/USCASJ/USCASJ.shp').to_crs('EPSG:3857')
        # Flatten geometries and extract xs and ys
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
        for i in sj_pillow_readings_df.columns:
            if i != 'time':
                sj_pillow_readings_df[i] = sj_pillow_readings_df[i].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
st.title("SWE Estimatator - Less Data Approach")
st.subheader("Overview")
st.write(
    "Welcome to the interactive data exploration platform. Below are dynamic charts and intuitive menus designed for delving into the dataset. Explore the intricacies of the model training process and evaluate the results with ease. Gain insights and a deeper understanding of the analysis."
)

## Section 2 - Basin Data Exploration
st.title("Basin Data Exploration")
st.write(
    "Use the interactive chart below displaying relevant snow pillows dependant on the selected basin. Selecting a specific snow pillow reveals its measurements on the accompanying time series chart. The chart is overlaid with historic ASO flight measurements, illustrating relationships between snow pillow data and ASO flight data."
)

col3, col4 = st.columns(2)
# Add a selectbox to each column
with col3:
    selected_1 = st.selectbox("Select Basin:", ("San Joaquin", "Toulumne"))
    if selected_1 == "San Joaquin":
        # Load snow pillow data from the CSV file
        sj_pillow_df = pd.read_csv(
            "data/snow_pillows/locations/sj_pillow_locations.csv"
        )
        sj_pillow_readings_df = pd.read_csv(
            "data/snow_pillows/measurements/sj_pillow_qa_table.csv"
        )
        sj_pillow_readings_df = sj_pillow_readings_df.fillna(0)
        for i in sj_pillow_readings_df.columns:
            if i != "time":
                sj_pillow_readings_df[i] = sj_pillow_readings_df[i].apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) else x
                )

        # Covert coordinates to Mercarto Projection
        def coor_conv(df, lon="longitude", lat="latitude"):
            k = 6378137
            df["x"] = df[lon] * (k * np.pi/180.0)
            df["y"] = np.log(np.tan((90 + df[lat]) * np.pi /360)) * k
            df["x"] = df[lon] * (k * np.pi / 180.0)
            df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360)) * k
            return

        # Convert
        coor_conv(sj_pillow_df)

        # Create map plot
        scatter = figure(title="SJ Snow Pillows", tools="tap,pan,wheel_zoom,reset,lasso_select", 
                    x_axis_type="mercator", y_axis_type="mercator",
                width=700, height=500,  x_range=(-13428333, -13090833), y_range=(4363056, 4671115))

        # Tile for map plot (Option 1)
        url = "http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png"
        #scatter.add_tile(WMTSTileSource(url=url))

        # Tile for map plot (Option 2)
        tile_provider = get_provider(Vendors.ESRI_IMAGERY)
        scatter.add_tile(tile_provider)

        # --- Add basin polygon ---
        scatter.patches(xs='xs', ys='ys', source=ColumnDataSource(patches_data), 
                    fill_color=None, line_color='blue', line_width=2)

        # Data
        source = ColumnDataSource(data=dict(x=sj_pillow_df["x"], y=sj_pillow_df["y"], snow_pillow=sj_pillow_df['id']))

        # Create pillow readings dictionary
        pillow_readings = {'x': pd.to_datetime(sj_pillow_readings_df["time"].tolist())}
        for i in range(1,sj_pillow_readings_df.shape[1]):
            pillow_readings[f'{sj_pillow_readings_df.columns[i]}'] = sj_pillow_readings_df[f'{sj_pillow_readings_df.columns[i]}'].tolist()
        scatter = figure(
            title="SJ Snow Pillows",
            tools="tap,pan,wheel_zoom,reset,lasso_select",
            x_axis_type="mercator",
            y_axis_type="mercator",
            width=650,
            height=500,
            x_range=(-13428333, -13090833),
            y_range=(4363056, 4671115),
        )

        # Tile for map plot (Option 1)
        url = "http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png"
        # scatter.add_tile(WMTSTileSource(url=url))

        # Tile for map plot (Option 2)
        tile_provider = get_provider(Vendors.CARTODBPOSITRON)
        scatter.add_tile(tile_provider)

        # Data
        source = ColumnDataSource(
            data=dict(
                x=sj_pillow_df["x"], y=sj_pillow_df["y"], snow_pillow=sj_pillow_df["id"]
            )
        )

        # Create pillow readings dictionary
        pillow_readings = {"x": pd.to_datetime(sj_pillow_readings_df["time"].tolist())}
        for i in range(1, sj_pillow_readings_df.shape[1]):
            pillow_readings[f"{sj_pillow_readings_df.columns[i]}"] = (
                sj_pillow_readings_df[f"{sj_pillow_readings_df.columns[i]}"].tolist()
            )

        spr = ColumnDataSource(data=pillow_readings)

        # Add random colors to dots to make the points more distinct
        colors = ['red', 'black', 'green', 'orange', 'purple','gray']
        source.data['color'] = [colors[i % len(colors)] for i in range(len(sj_pillow_df))]
        scatter.circle(x='x', y='y', size=10, source=source, color='color',line_color='black', line_width=1)
        color_mapping = dict(zip(source.data['snow_pillow'], source.data['color']))

        # Add text labels to points
        text_glyph = scatter.text(x='x', y='y', text='snow_pillow', source=source, text_color='black', text_font_size='8pt', x_offset=5, y_offset=-2)
        colors = ["red", "black", "green", "orange", "purple", "gray"]
        source.data["color"] = [
            colors[i % len(colors)] for i in range(len(sj_pillow_df))
        ]
        scatter.circle(
            x="x",
            y="y",
            size=10,
            source=source,
            color="color",
            line_color="black",
            line_width=1,
        )
        color_mapping = dict(zip(source.data["snow_pillow"], source.data["color"]))

        # Add text labels to points
        text_glyph = scatter.text(
            x="x",
            y="y",
            text="snow_pillow",
            source=source,
            text_color="black",
            text_font_size="8pt",
            x_offset=5,
            y_offset=-2,
        )
        text_glyph.selection_glyph = None
        text_glyph.nonselection_glyph = None
        text_glyph.muted_glyph = None

        # Create line plot
        line_source = ColumnDataSource(data=dict())
        line = figure(title="Line Chart for Selected Snow Pillow", width=700, height=500, x_axis_type='datetime', y_range=Range1d(start=0, end=3500))
        line.xaxis.formatter = DatetimeTickFormatter(
            days="%d %b %Y",  # Format for day-level ticks
            months="%b %Y",   # Format for month-level ticks
            years="%Y"        # Format for year-level ticks
        )

        # Modify the callback to update the p chart
        select_SnowPillow = CustomJS(args=dict(source=source, line_source = line_source, spr = pillow_readings), code="""
        line = figure(
            title="Line Chart for Selected Snow Pillow",
            width=1200,
            height=500,
            x_axis_type="datetime",
            y_range=Range1d(start=0, end=3500),
        )
        line.xaxis.formatter = DatetimeTickFormatter(
            days="%d %b %Y",  # Format for day-level ticks
            months="%b %Y",  # Format for month-level ticks
            years="%Y",  # Format for year-level ticks
        )

        # Modify the callback to update the p chart
        select_SnowPillow = CustomJS(
            args=dict(source=source, line_source=line_source, spr=pillow_readings),
            code="""
            const indices = cb_obj.indices;
            if (indices.length === 0) return;
            let selectedSnowPillows = [];
            let x_values = [];
            let y_values = [];
            let selectedData = {x: spr['x']};
            for (let i = 0; i < indices.length; i++) {
                const index = indices[i];
                const data = source.data;
                const selectedSnowPillow = data['snow_pillow'][index];
                selectedSnowPillows.push(selectedSnowPillow);
                console.log(selectedSnowPillows)
            }
            for (let i = 0;i < selectedSnowPillows.length; i++) {
                selectedData[`${selectedSnowPillows[i]}`] = spr[`${selectedSnowPillows[i]}`]
            }
            line_source.data = selectedData;
            line_source.change.emit();
        """)
        """,
        )
        source.selected.js_on_change("indices", select_SnowPillow)

        # Create time series line plots
        lines = {}
        pillow_name_list = sj_pillow_df['id']
        for name in pillow_name_list:
            if name != 'time':
                lines[name] = line.line('x', name, source=line_source, name=name, color=color_mapping[name], line_width=2)
        pillow_name_list = sj_pillow_df["id"]
        for name in pillow_name_list:
            if name != "time":
                lines[name] = line.line(
                    "x",
                    name,
                    source=line_source,
                    name=name,
                    color=color_mapping[name],
                    line_width=2,
                )

        # Add Hover tooltip for each line item
        for name, renderer in lines.items():
            hover = HoverTool(
                renderers=[renderer],  # Apply this hover tool only to the specific line
                tooltips=[
                    ("Pillow ID", name), 
                    ("Date", "@x{%F}"),  # Display the date in YYYY-MM-DD format
                    ("Units: mm", f"@{name}")  # Display the value for the specific line
                ],
                formatters={
                    '@x': 'datetime',  # Use 'datetime' formatter for the x value

                },
                mode='mouse'  # Show tooltip for the closest data point to the mouse
                    ("Pillow ID", name),
                    ("Date", "@x{%F}"),  # Display the date in YYYY-MM-DD format
                    (
                        "Units: mm",
                        f"@{name}",
                    ),  # Display the value for the specific line
                ],
                formatters={
                    "@x": "datetime",  # Use 'datetime' formatter for the x value
                },
                mode="mouse",  # Show tooltip for the closest data point to the mouse
            )
            line.add_tools(hover)

        # ASO flight data
        sj_aso_df = pd.read_csv('data/aso/USCASJ/uscasj_aso_sum.csv')
        sj_aso_df['aso_mean_bins_mm'] = sj_aso_df['aso_mean_bins_mm'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
    
        # Plot points for ASO flights
        points_data = {
            'time': pd.to_datetime(sj_aso_df['time']),  # Example dates
            'value': sj_aso_df['aso_mean_bins_mm'],  # Example values
            'image_url': [f'data/aso/USCASJ/images/plot{i}.png' for i in range(0,len(sj_aso_df))]
        }
        points_source = ColumnDataSource(points_data)
    
        # Hover tooltip for ASO plot points
        aso_hover = HoverTool(
            renderers=[line.circle('time', 'value', source=points_source, size=10, color='blue', line_color='black', line_width=1, legend_label='ASO Flights')],
            tooltips=[
                ("Date", "@time{%F}"),  # Display the date in YYYY-MM-DD format
                ("Value (mm)", "@value")  # Display the value
            ],
            formatters={
                '@time': 'datetime',  # Use 'datetime' formatter for the time value
        sj_aso_df = pd.read_csv("data/aso/USCASJ/uscasj_aso_sum.csv")
        sj_aso_df["aso_mean_bins_mm"] = sj_aso_df["aso_mean_bins_mm"].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else x
        )

        # Plot points for ASO flights
        points_data = {
            "time": pd.to_datetime(sj_aso_df["time"]),  # Example dates
            "value": sj_aso_df["aso_mean_bins_mm"],  # Example values
            "image_url": [
                f"data/aso/USCASJ/images/plot{i}.png" for i in range(0, len(sj_aso_df))
            ],
        }
        points_source = ColumnDataSource(points_data)

        # Hover tooltip for ASO plot points
        aso_hover = HoverTool(
            renderers=[
                line.circle(
                    "time",
                    "value",
                    source=points_source,
                    size=10,
                    color="blue",
                    line_color="black",
                    line_width=1,
                    legend_label="ASO Flights",
                )
            ],
            tooltips=[
                ("Date", "@time{%F}"),  # Display the date in YYYY-MM-DD format
                ("Value (mm)", "@value"),  # Display the value
            ],
            formatters={
                "@time": "datetime",  # Use 'datetime' formatter for the time value
            },
            mode="mouse",  # Show tooltip for the closest data point to the mouse
        )
        # Add the hover tool to the line figure
        line.add_tools(aso_hover)
        st.bokeh_chart(column(scatter, line), use_container_width=True)
    else:
        # Load snow pillow data from the CSV file
        tm_pillow_df = pd.read_csv(
            "data/snow_pillows/locations/tm_pillow_locations.csv"
        )
        tm_pillow_readings_df = pd.read_csv(
            "data/snow_pillows/measurements/tm_pillow_qa_table.csv"
        )
        tm_pillow_readings_df = tm_pillow_readings_df.fillna(0)
        for i in tm_pillow_readings_df.columns:
            if i != "time":
                tm_pillow_readings_df[i] = tm_pillow_readings_df[i].apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) else x
                )

        # Covert coordinates to Mercarto Projection
        def coor_conv(df, lon="longitude", lat="latitude"):
            k = 6378137
            df["x"] = df[lon] * (k * np.pi / 180.0)
            df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360)) * k
            return

        # Convert
        coor_conv(tm_pillow_df)

        # Create map plot
        scatter = figure(
            title="TM Snow Pillows",
            tools="tap,pan,wheel_zoom,reset,lasso_select",
            x_axis_type="mercator",
            y_axis_type="mercator",
            width=650,
            height=500,
            x_range=(-13428333, -13090833),
            y_range=(4363056, 4671115),
        )

        # Add the hover tool to the line figure
        line.add_tools(aso_hover)

        # Display the updated plots
        col5, col6 = st.columns(2)
        with col5:
            st.bokeh_chart(column(scatter, line), use_container_width=False)

    else:
        # Load snow pillow data from the CSV file
        tm_pillow_df = pd.read_csv('data/snow_pillows/locations/tm_pillow_locations.csv')
        tm_pillow_readings_df = pd.read_csv('data/snow_pillows/measurements/tm_pillow_qa_table.csv')
        tm_pillow_readings_df = tm_pillow_readings_df.fillna(0)
        basin_gdf = gpd.read_file('data/shape/USCATM/USCATM.shp').to_crs('EPSG:3857')
        # Flatten geometries and extract xs and ys
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
    
        for i in tm_pillow_readings_df.columns:
            if i != 'time':
                tm_pillow_readings_df[i] = tm_pillow_readings_df[i].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)

        # Covert coordinates to Mercarto Projection
        def coor_conv(df, lon="longitude", lat="latitude"):
            k = 6378137
            df["x"] = df[lon] * (k * np.pi/180.0)
            df["y"] = np.log(np.tan((90 + df[lat]) * np.pi /360)) * k
            return

        # Convert
        coor_conv(tm_pillow_df)

        # Create map plot
        scatter = figure(title="TM Snow Pillows", tools="tap,pan,wheel_zoom,reset,lasso_select", 
                x_axis_type="mercator", y_axis_type="mercator",
            width=700, height=500,  x_range=(-13428333, -13090833), y_range=(4363056, 4671115))

        # --- Add basin polygon ---
        scatter.patches(xs='xs', ys='ys', source=ColumnDataSource(patches_data), 
                fill_color=None, line_color='red', line_width=2)

        # Tile for map plot (Option 1)
        url = "http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png"
        #scatter.add_tile(WMTSTileSource(url=url))

        # Tile for map plot (Option 2)
        tile_provider = get_provider(Vendors.ESRI_IMAGERY)
        scatter.add_tile(tile_provider)

        # Data
        source = ColumnDataSource(data=dict(x=tm_pillow_df["x"], y=tm_pillow_df["y"], snow_pillow=tm_pillow_df['id']))

        # Create pillow readings dictionary
        pillow_readings = {'x': pd.to_datetime(tm_pillow_readings_df["time"].tolist())}
        for i in range(1,tm_pillow_readings_df.shape[1]):
            pillow_readings[f'{tm_pillow_readings_df.columns[i]}'] = tm_pillow_readings_df[f'{tm_pillow_readings_df.columns[i]}'].tolist()

        # Tile for map plot (Option 1)
        url = "http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png"
        # scatter.add_tile(WMTSTileSource(url=url))

        # Tile for map plot (Option 2)
        tile_provider = get_provider(Vendors.CARTODBPOSITRON)
        scatter.add_tile(tile_provider)

        # Data
        source = ColumnDataSource(
            data=dict(
                x=tm_pillow_df["x"], y=tm_pillow_df["y"], snow_pillow=tm_pillow_df["id"]
            )
        )

        # Create pillow readings dictionary
        pillow_readings = {"x": pd.to_datetime(tm_pillow_readings_df["time"].tolist())}
        for i in range(1, tm_pillow_readings_df.shape[1]):
            pillow_readings[f"{tm_pillow_readings_df.columns[i]}"] = (
                tm_pillow_readings_df[f"{tm_pillow_readings_df.columns[i]}"].tolist()
            )

        spr = ColumnDataSource(data=pillow_readings)

        # Add random colors to dots to make the points more distinct
        colors = ['red', 'black', 'green', 'orange', 'purple','gray']
        source.data['color'] = [colors[i % len(colors)] for i in range(len(tm_pillow_df))]
        scatter.circle(x='x', y='y', size=10, source=source, color='color',line_color='black', line_width=1)
        color_mapping = dict(zip(source.data['snow_pillow'], source.data['color']))

        # Add text labels to points
        text_glyph = scatter.text(x='x', y='y', text='snow_pillow', source=source, text_color='black', text_font_size='8pt', x_offset=5, y_offset=-2)
        colors = ["red", "black", "green", "orange", "purple", "gray"]
        source.data["color"] = [
            colors[i % len(colors)] for i in range(len(tm_pillow_df))
        ]
        scatter.circle(
            x="x",
            y="y",
            size=10,
            source=source,
            color="color",
            line_color="black",
            line_width=1,
        )
        color_mapping = dict(zip(source.data["snow_pillow"], source.data["color"]))

        # Add text labels to points
        text_glyph = scatter.text(
            x="x",
            y="y",
            text="snow_pillow",
            source=source,
            text_color="black",
            text_font_size="8pt",
            x_offset=5,
            y_offset=-2,
        )
        text_glyph.selection_glyph = None
        text_glyph.nonselection_glyph = None
        text_glyph.muted_glyph = None

        # Create line plot
        line_source = ColumnDataSource(data=dict())
        line = figure(title="Line Chart for Selected Snow Pillow", width=700, height=500, x_axis_type='datetime', y_range=Range1d(start=0, end=3500))
        line.xaxis.formatter = DatetimeTickFormatter(
            days="%d %b %Y",  # Format for day-level ticks
            months="%b %Y",   # Format for month-level ticks
            years="%Y"        # Format for year-level ticks
        )

        # Modify the callback to update the p chart
        select_SnowPillow = CustomJS(args=dict(source=source, line_source = line_source, spr = pillow_readings), code="""
        const indices = cb_obj.indices;
        if (indices.length === 0) return;
        let selectedSnowPillows = [];
        let x_values = [];
        let y_values = [];
        let selectedData = {x: spr['x']};
        for (let i = 0; i < indices.length; i++) {
            const index = indices[i];
            const data = source.data;
            const selectedSnowPillow = data['snow_pillow'][index];
            selectedSnowPillows.push(selectedSnowPillow);
            console.log(selectedSnowPillows)
        }
        for (let i = 0;i < selectedSnowPillows.length; i++) {
            selectedData[`${selectedSnowPillows[i]}`] = spr[`${selectedSnowPillows[i]}`]
        }
        line_source.data = selectedData;
        line_source.change.emit();
    """)
        source.selected.js_on_change("indices", select_SnowPillow)
    
        # Create time series line plots
        lines = {}
        pillow_name_list = tm_pillow_df['id']
        for name in pillow_name_list:
            if name != 'time':
                lines[name] = line.line('x', name, source=line_source, name=name, color=color_mapping[name], line_width=2)

        # Add Hover tooltip for each line item
        for name, renderer in lines.items():
            hover = HoverTool(
                renderers=[renderer],  # Apply this hover tool only to the specific line
                tooltips=[
                    ("Pillow ID", name), 
                    ("Date", "@x{%F}"),  # Display the date in YYYY-MM-DD format
                    ("Units: mm", f"@{name}")  # Display the value for the specific line
                ],
                formatters={
                    '@x': 'datetime',  # Use 'datetime' formatter for the x value

                },
                mode='mouse'  # Show tooltip for the closest data point to the mouse
            )
            line.add_tools(hover)

        # ASO flight data
        tm_aso_df = pd.read_csv('data/aso/USCATM/uscatm_aso_sum.csv')
        tm_aso_df['aso_mean_bins_mm'] = tm_aso_df['aso_mean_bins_mm'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)

        # Plot points for ASO flights
        points_data = {
            'time': pd.to_datetime(tm_aso_df['time']),  # Example dates
            'value': tm_aso_df['aso_mean_bins_mm'],  # Example values
            'image_url': [f'data/aso/USCATM/images/plot{i}.png' for i in range(0,len(tm_aso_df))]
        }
        points_source = ColumnDataSource(points_data)
    
        # Hover tooltip for ASO plot points
        aso_hover = HoverTool(
            renderers=[line.circle('time', 'value', source=points_source, size=10, color='blue',line_color='black', line_width=1, legend_label='ASO Flights')],
            tooltips=[
                ("Date", "@time{%F}"),  # Display the date in YYYY-MM-DD format
                ("Value (mm)", "@value")  # Display the value
            ],
            formatters={
                '@time': 'datetime',  # Use 'datetime' formatter for the time value
        line = figure(
            title="Line Chart for Selected Snow Pillow",
            width=1200,
            height=500,
            x_axis_type="datetime",
            y_range=Range1d(start=0, end=3500),
        )
        line.xaxis.formatter = DatetimeTickFormatter(
            days="%d %b %Y",  # Format for day-level ticks
            months="%b %Y",  # Format for month-level ticks
            years="%Y",  # Format for year-level ticks
        )

        # Modify the callback to update the p chart
        select_SnowPillow = CustomJS(
            args=dict(source=source, line_source=line_source, spr=pillow_readings),
            code="""
            const indices = cb_obj.indices;
            if (indices.length === 0) return;
            let selectedSnowPillows = [];
            let x_values = [];
            let y_values = [];
            let selectedData = {x: spr['x']};
            for (let i = 0; i < indices.length; i++) {
                const index = indices[i];
                const data = source.data;
                const selectedSnowPillow = data['snow_pillow'][index];
                selectedSnowPillows.push(selectedSnowPillow);
                console.log(selectedSnowPillows)
            }
            for (let i = 0;i < selectedSnowPillows.length; i++) {
                selectedData[`${selectedSnowPillows[i]}`] = spr[`${selectedSnowPillows[i]}`]
            }
            line_source.data = selectedData;
            line_source.change.emit();
        """,
        )
        source.selected.js_on_change("indices", select_SnowPillow)

        # Create time series line plots
        lines = {}
        pillow_name_list = tm_pillow_df["id"]
        for name in pillow_name_list:
            if name != "time":
                lines[name] = line.line(
                    "x",
                    name,
                    source=line_source,
                    name=name,
                    color=color_mapping[name],
                    line_width=2,
                )

        # Add Hover tooltip for each line item
        for name, renderer in lines.items():
            hover = HoverTool(
                renderers=[renderer],  # Apply this hover tool only to the specific line
                tooltips=[
                    ("Pillow ID", name),
                    ("Date", "@x{%F}"),  # Display the date in YYYY-MM-DD format
                    (
                        "Units: mm",
                        f"@{name}",
                    ),  # Display the value for the specific line
                ],
                formatters={
                    "@x": "datetime",  # Use 'datetime' formatter for the x value
                },
                mode="mouse",  # Show tooltip for the closest data point to the mouse
            )
            line.add_tools(hover)

        # ASO flight data
        tm_aso_df = pd.read_csv("data/aso/USCATM/uscatm_aso_sum.csv")
        tm_aso_df["aso_mean_bins_mm"] = tm_aso_df["aso_mean_bins_mm"].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else x
        )

        # Plot points for ASO flights
        points_data = {
            "time": pd.to_datetime(tm_aso_df["time"]),  # Example dates
            "value": tm_aso_df["aso_mean_bins_mm"],  # Example values
            "image_url": [
                f"data/aso/USCATM/images/plot{i}.png" for i in range(0, len(tm_aso_df))
            ],
        }
        points_source = ColumnDataSource(points_data)

        # Hover tooltip for ASO plot points
        aso_hover = HoverTool(
            renderers=[
                line.circle(
                    "time",
                    "value",
                    source=points_source,
                    size=10,
                    color="blue",
                    line_color="black",
                    line_width=1,
                    legend_label="ASO Flights",
                )
            ],
            tooltips=[
                ("Date", "@time{%F}"),  # Display the date in YYYY-MM-DD format
                ("Value (mm)", "@value"),  # Display the value
            ],
            formatters={
                "@time": "datetime",  # Use 'datetime' formatter for the time value
            },
            mode="mouse",  # Show tooltip for the closest data point to the mouse
        )
        # Add the hover tool to the line figure
        line.add_tools(aso_hover)

        # Display the updated plots
        st.bokeh_chart(column(scatter, line), use_container_width=False)

    st.write("Select a date to visualize a LiDAR derived map from a historic flight map.")

    col5, col6 = st.columns(2)
    # Add a selectbox to each column
    with col5:
        flight_dates = ["Dates"]
        if selected_1 == 'San Joaquin':
            sj_flights = [text.replace("-", "/") for text in sj_aso_df['time']]
            flight_dates.extend(sj_flights)
        else:
            tm_flights = [text.replace("-", "/") for text in tm_aso_df['time']]
            flight_dates.extend(tm_flights)
        aso_flight_date = st.selectbox(
        'Select ASO flight date:',
        flight_dates
        )

    # Show ASO flight scan image
    if aso_flight_date != "Dates":
        st.image(f'data/aso/{"USCASJ" if selected_1 == "San Joaquin" else "USCATM"}/images/plot{flight_dates.index(aso_flight_date)-1}.png')
    else:
        st.image('data/aso/blank.png')



## Section 3 - Training Data - Slide 6
with tab_train: 
# Write title 
    st.title("Training")
    st.write("""
            For ***Less Data*** we created multiple linear regression models using a cross-validation approach to reduce overfitting and optimize the selected pillows used in each model. The user can make the selections below based on **Basin**, **Elevation Interval**, **Imputation Strategy**, and **Season Segmentation** to interpret each model's training performance.
            """)

    # Path to images folder
    image_folder = "images"

    # Basin Options
    basin_options = ['San Joaquin', 'Toulumne']
    selected_basin = st.selectbox('Select Basin', basin_options,index=0)

    # Dropdown for selecting elevation bin
    elevation_options = ['<7k','7-8k','8-9k','9-10k','10-11k','11-12k','>12k','Total']
    selected_elevation = st.selectbox('Select Elevation', elevation_options,index=7)

    # Dropdown for selecting elevation bin
    impute_options = ['Drop NaNs','Predict NaNs']
    selected_imputation = st.selectbox('Select Imputation Strategy', impute_options,index=0)

    # Dropdown for selecting elevation bin
    meltSeason_options = ['Total Season','Accumulation','Melt']
    selected_season = st.selectbox('Select Season Segmentation', meltSeason_options,index=0)

    # convert selected values to inputs to model.
    aso_site_name,elev_band,isImpute,isSplit,isAccum,start_wy,end_wy = load_data.select_vals_to_inputs(selected_basin,selected_elevation,selected_imputation,selected_season)

    # load data.
    obs_gdf,obs_data,df_sum_total,slice_df,all_pils,baseline_pils,aso_tseries_1,dem_bin = load_data.load_mlr_basin_data(aso_site_name,root_dir)

    # get terrain plotting info.
    sorted_elev, elev_cunane_position = plotting.terrain_cdf_distribution(dem_bin[-1,:,:].values.flatten())

    # run mlr cross validation.
    predictions_bestfit,predictions_validation,aso_tseries_2,obs_data_6,table_dict,selected_pillow,max_swe_,title_str,stations2 = lm_model.run_mlr_cross_validation(aso_site_name,root_dir,aso_tseries_1,elev_band,isSplit,isImpute,
                             isAccum,df_sum_total,all_pils,obs_data,baseline_pils,start_wy,end_wy)

    # create plot.
    fig = plotting.combine_cross_validation_plots(aso_site_name,elev_band,isImpute,aso_tseries_2,predictions_bestfit,predictions_validation,
                                   max_swe_,stations2,dem_bin,obs_gdf,sorted_elev,elev_cunane_position,obs_data_6,
                                   start_wy,end_wy,table_dict,title_str)
    # plot figure.
    st.pyplot(fig)

with tab_inference:
    ## Section 4 -- Testing/Results

    st.title("Results and Evaluation")
    st.write("Here we are comparing model inference results in the San Joaquin. We compare the best ***MLR Drop NaNs*** and ***Predict NaNs*** models to our label (***ASO***) and benchmarks (***SNODAS*** and ***UASWE***). The flights occured on February 26th, 2025 and March 25th, 2025. Future work will extend a similar comparison to the Tuolumne basin.")
    st.write("- First, we compare the models for each elevation interval, ***separately***.")
    st.write("- Next, we compare the models across elevation intervals and generate statisctics relevant to our ***KPI*** metrics.")
    ### Elevation -------------------------------------------- \
    st.write("## Elevation Comparison")

    # Basin Options
    basin_options_2 = ['San Joaquin']
    selected_basin_2 = st.selectbox('Basin', basin_options_2,index=0)

    # Dropdown for selecting elevation bin
    elevation_options_2 = ['<7k','7-8k','8-9k','9-10k','10-11k','11-12k','>12k','Total']
    selected_elevation_2 = st.selectbox('Elevation', elevation_options_2,index=7)

    # convert selected values to inputs to model.
    aso_site_name,selected_dict = load_data.select_vals_to_outputs(selected_basin_2)

    # create plots.
    fig,ax = plt.subplots(1,2,figsize = (10,5),sharey = True,dpi = 200)
    ax[1],val1 = plotting.create_difference_plots_benchmark_mlr_mult_dates(root_dir,aso_site_name,ax[1],date_str = '20250325',
                                                     elevation_bin = selected_dict[selected_elevation_2]['elevation_bin'],
                                                     ymax_lim = selected_dict[selected_elevation_2]['ymax_lim'],
                                                     FirstPlot = False,text_adjust = selected_dict[selected_elevation_2]['text_adjust'])
    ax[1],val1 = plotting.create_difference_plots_benchmark_mlr_mult_dates(root_dir,aso_site_name,ax[0],date_str = '20250226',
                                                     elevation_bin = selected_dict[selected_elevation_2]['elevation_bin'],
                                                     ymax_lim = selected_dict[selected_elevation_2]['ymax_lim'],
                                                     FirstPlot = True,text_adjust = selected_dict[selected_elevation_2]['text_adjust'])
    plt.suptitle(f'{selected_basin_2} - {selected_elevation} Mean SWE Comparison',fontweight = 'bold',fontsize = 24)
    plt.tight_layout()
    st.pyplot(fig)
    ### Model -------------------------------------------- \
    st.write("## Model Comparison")


    # Dropdown for selecting elevation bin
    date_options = ['2025-02-26','2025-03-25']
    selected_test_date = st.selectbox('Test Dates', date_options,index=1)

    fig = plotting.create_model_comparison(root_dir,aso_site_name,date_str = selected_test_date.replace('-',''))
    st.pyplot(fig)

with tab_conclusions:
    st.write("Conclusions")

    col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

    with col1:
        st.markdown("## Hypothesis 1 - Less Data")
        st.write("""
            Based on the results of our models for the final flight (2025-03-25) in the ***San Joaquin***:
            - Drop NaNs
                - Fails Accuracy KPI (within 10% of ASO for Total Mean SWE).
                - Better than SNODAS.
                - Worse than UASWE.
            - Predict NaNs
                - Passes Accuracy KPI (within 10% of ASO for Total Mean SWE).
                - Better than SNODAS.
                - Worse than UASWE for Total Accuracy. ***However***, shows promise of performing better across elevational gradients.
            """)

    with col2:
        image_fpath = os.path.join(root_dir, "images", "Overview", "less_is_more.png")
        st.image(image_fpath, width=500)
with col4:
    flight_dates = []
    if selected_1 == "San Joaquin":
        sj_flights = [text.replace("-", "/") for text in sj_aso_df["time"]]
        flight_dates.extend(sj_flights)
    else:
        tm_flights = [text.replace("-", "/") for text in tm_aso_df["time"]]
        flight_dates.extend(tm_flights)
    aso_flight_date = st.selectbox("Select ASO flight date:", flight_dates)
    if aso_flight_date != "Dates":
        st.image(
            f'data/aso/{"USCASJ" if selected_1 == "San Joaquin" else "USCATM"}/images/plot{flight_dates.index(aso_flight_date)}.png'
        )
    else:
        st.image("data/aso/blank.png", use_container_width=False)

## Section 3 - Training Data - Slide 6

# Write title
st.title("Modeling & Training Validation")

# Path to images folder
image_folder = "images"

# Dropdown for selecting elevation bin
elevation_options = [
    "<7k",
    "7-8k",
    "8-9k",
    "9-10k",
    "10-11k",
    "11-12k",
    ">12k",
    "Total",
]
selected_elevation = st.selectbox("Select Elevation", elevation_options)

# Basin Options
basin_options = ["San Joaquin", "Toulumne"]
selected_basin = st.selectbox("Select Basin", basin_options)

# Form file path
png_path = os.path.join(
    "images",
    "USCASJ" if selected_basin == "San Joaquin" else "USCATM",
    "MLR",
    (
        "7k"
        if selected_elevation == "<7k"
        else "12k" if selected_elevation == ">12k" else selected_elevation
    ),
    "validation.png",
)


# Display selected png
st.image(png_path)

## Section 4

st.subheader("Results and Evaluation")
st.write("Compare results against the latest ASO flights")
col1, col2, cola, colb = st.columns(4)
# Add a selectbox to each column
with col1:
    option1 = st.selectbox("Select ASO flight date:", ("2025/03/25", "2025/02/26"))
with col2:
    option2 = st.selectbox(
        "Select elevation:",
        (
            "<7k",
            "7-8k",
            "8-9k",
            "9-10k",
            "10-11k",
            "11-12k",
            ">12k",
            "Total",
            "Combined",
        ),
    )
# Display image
image_path = os.path.join(
    "data",
    "MLR_Comparison",
    "7k" if option2 == "<7k" else "12k" if option2 == ">12k" else option2,
    f"{option1.replace('/', '_')}.png",
)

st.image(image_path)
