import streamlit as st
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CustomJS, DatetimeTickFormatter, HoverTool, Range1d, TapTool
from bokeh.layouts import column
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models import WMTSTileSource
from bokeh.events import Tap
from bokeh.embed import components
import pandas as pd
import math
import numpy as np
from bokeh.io import show
from bokeh.models import Toggle

## Section 1 - EDA Less Data Overview & Motivation
st.title("EDA - Less Data")
st.subheader("Overview")
st.write("Lorem Ipsum")
st.subheader("Motivation")
st.write("Lorem Ipsum")

st.subheader("Market Comparison")
st.write("Compare our results against the latest ASO flights")
col1, col2 = st.columns(2)
# Add a selectbox to each column
with col1:
    option1 = st.selectbox(
        'Select ASO flight date:',
        ('2025/03/25', '2025/02/26')
    )
with col2:
    option2 = st.selectbox(
        'Select elevation:',
        ('7k', '7-8k', '8-9k','9-10k','10-11k','11-12k','12k')
    )
# Display image
st.image(f'data/MLR_Comparison/{option2}/{option1.replace("/","_")}.png')

## Section 2 - Basin Data Exploration
st.title("Basin Data Exploration")
st.write("Lorem Ipsum Intro Text")

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
    for i in sj_pillow_readings_df.columns:
        if i != 'time':
            sj_pillow_readings_df[i] = sj_pillow_readings_df[i].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)

    # Covert coordinates to Mercarto Projection
    def coor_conv(df, lon="longitude", lat="latitude"):
        k = 6378137
        df["x"] = df[lon] * (k * np.pi/180.0)
        df["y"] = np.log(np.tan((90 + df[lat]) * np.pi /360)) * k
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
    tile_provider = get_provider(Vendors.CARTODBPOSITRON)
    scatter.add_tile(tile_provider)

    # Data
    source = ColumnDataSource(data=dict(x=sj_pillow_df["x"], y=sj_pillow_df["y"], snow_pillow=sj_pillow_df['id']))

    # Create pillow readings dictionary
    pillow_readings = {'x': pd.to_datetime(sj_pillow_readings_df["time"].tolist())}
    for i in range(1,sj_pillow_readings_df.shape[1]):
        pillow_readings[f'{sj_pillow_readings_df.columns[i]}'] = sj_pillow_readings_df[f'{sj_pillow_readings_df.columns[i]}'].tolist()

    spr = ColumnDataSource(data=pillow_readings)

    # Add random colors to dots to make the points more distinct
    colors = ['red', 'black', 'green', 'orange', 'purple','gray']
    source.data['color'] = [colors[i % len(colors)] for i in range(len(sj_pillow_df))]
    scatter.circle(x='x', y='y', size=10, source=source, color='color',line_color='black', line_width=1)
    color_mapping = dict(zip(source.data['snow_pillow'], source.data['color']))

    # Add text labels to points
    text_glyph = scatter.text(x='x', y='y', text='snow_pillow', source=source, text_color='black', text_font_size='8pt', x_offset=5, y_offset=-2)
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
    pillow_name_list = sj_pillow_df['id']
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
        },
        mode='mouse'  # Show tooltip for the closest data point to the mouse
    )
    # Add the hover tool to the line figure
    line.add_tools(aso_hover)

    # Display the updated plots
    st.bokeh_chart(column(scatter, line), use_container_width=False)
else:
    # Load snow pillow data from the CSV file
    tm_pillow_df = pd.read_csv('data/snow_pillows/locations/tm_pillow_locations.csv')
    tm_pillow_readings_df = pd.read_csv('data/snow_pillows/measurements/tm_pillow_qa_table.csv')
    tm_pillow_readings_df = tm_pillow_readings_df.fillna(0)
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
    scatter = figure(title="SJ Snow Pillows", tools="tap,pan,wheel_zoom,reset,lasso_select", 
                x_axis_type="mercator", y_axis_type="mercator",
            width=700, height=500,  x_range=(-13428333, -13090833), y_range=(4363056, 4671115))

    # Tile for map plot (Option 1)
    url = "http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png"
    #scatter.add_tile(WMTSTileSource(url=url))

    # Tile for map plot (Option 2)
    tile_provider = get_provider(Vendors.CARTODBPOSITRON)
    scatter.add_tile(tile_provider)

    # Data
    source = ColumnDataSource(data=dict(x=tm_pillow_df["x"], y=tm_pillow_df["y"], snow_pillow=tm_pillow_df['id']))

    # Create pillow readings dictionary
    pillow_readings = {'x': pd.to_datetime(tm_pillow_readings_df["time"].tolist())}
    for i in range(1,tm_pillow_readings_df.shape[1]):
        pillow_readings[f'{tm_pillow_readings_df.columns[i]}'] = tm_pillow_readings_df[f'{tm_pillow_readings_df.columns[i]}'].tolist()

    spr = ColumnDataSource(data=pillow_readings)

    # Add random colors to dots to make the points more distinct
    colors = ['red', 'black', 'green', 'orange', 'purple','gray']
    source.data['color'] = [colors[i % len(colors)] for i in range(len(tm_pillow_df))]
    scatter.circle(x='x', y='y', size=10, source=source, color='color',line_color='black', line_width=1)
    color_mapping = dict(zip(source.data['snow_pillow'], source.data['color']))

    # Add text labels to points
    text_glyph = scatter.text(x='x', y='y', text='snow_pillow', source=source, text_color='black', text_font_size='8pt', x_offset=5, y_offset=-2)
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
        },
        mode='mouse'  # Show tooltip for the closest data point to the mouse
    )
    # Add the hover tool to the line figure
    line.add_tools(aso_hover)

    # Display the updated plots
    st.bokeh_chart(column(scatter, line), use_container_width=False)

col5, col6 = st.columns(2)
# Add a selectbox to each column
with col5:
    flight_dates = ["Dates"]
    if selected_1 == 'San Joaquin':
        sj_flights = sj_aso_df['time'].tolist()
        flight_dates.extend(sj_flights)
    else:
        tm_flights = tm_aso_df['time'].tolist()
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

import streamlit as st
import os
import base64
from PIL import Image
from pdf2image import convert_from_path

# Write title 
st.title("Modeling/Training Validation")

# Function to display a local PDF file
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# Path to images folder
image_folder = "images"

# Dropdown for selecting elevation bin
category_options = ['Elevation 1', 'Elevation 2', 'Elevation 3', 'Elevation 4', 'Elevation 5', 'Elevation 6', 'Elevation 7']
selected_category = st.selectbox('Select Category', category_options)

# Extract the bin number from selection
bin_number = category_options.index(selected_category) + 1
pdf_filename = f"bin{bin_number}.pdf"
pdf_path = os.path.join(image_folder, pdf_filename)

# Display selected PDF
display_pdf(pdf_path)

# Load your images
img1 = Image.open(pdf_path)  # Main image (left)
img2 = Image.open(pdf_path)  # Top right image
img3 = Image.open(pdf_path)  # Bottom right image

# Layout with columns
col1, col2 = st.columns([2, 1])  # Wider left column

# Left column: one big image
with col1:
    st.image(img1, caption="Main Image", use_column_width=True)

# Right column: two stacked images
with col2:
    st.image(img2, caption="Top Image", use_column_width=True)
    st.image(img3, caption="Bottom Image", use_column_width=True)



## Section 4 - Slide 7

