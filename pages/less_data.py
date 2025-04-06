import streamlit as st
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.layouts import column
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models import WMTSTileSource
import pandas as pd
import math
import numpy as np

st.title("Mo Snow Less Data Hello World")

# Load snow pillow data from the CSV file
sj_pillow_df = pd.read_csv('data/snow_pillows/locations/sj_pillow_locations.csv')
sj_pillow_readings_df = pd.read_csv('data/snow_pillows/measurements/sj_pillow_qa_table.csv')

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
           active_drag="lasso_select", x_axis_type="mercator", y_axis_type="mercator",
           width=500, height=500,  x_range=(-13428333, -13090833), y_range=(4363056, 4671115))

# Tile for map plot (Option 1)
url = "http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png"
#scatter.add_tile(WMTSTileSource(url=url))

# Tile for map plot (Option 2)
tile_provider = get_provider(Vendors.CARTODBPOSITRON)
scatter.add_tile(tile_provider)

# Data
source = ColumnDataSource(data=dict(x=sj_pillow_df["x"], y=sj_pillow_df["y"], snow_pillow=sj_pillow_df['id']))

# Add random colors to dots to make the points more distinct
colors = np.random.choice(['red', 'blue', 'green', 'yellow', 'purple'], size=len(sj_pillow_df))
source.data['color'] = colors
scatter.circle(x='x', y='y', size=10, source=source, color='color',line_color='black', line_width=1)

# Add text labels to points
scatter.text(x='x', y='y', text='snow_pillow', source=source, text_color='black', text_font_size='8pt', x_offset=5, y_offset=-2)

# Remove the line chart and its data source
# line_source = ColumnDataSource(data=dict(x=[], y=[]))
# line = figure(title="Line Chart for Selected Snow Pillow", width=400, height=400)
# line.line('x', 'y', source=line_source)
# Ensure the p chart is defined and ready to be updated

sj_pillow_readings_df['time'] = pd.to_datetime(sj_pillow_readings_df['time'])
line = figure(title="Line Chart Example", x_axis_label='Time', y_axis_label='Values', x_axis_type='datetime')
colors = ['blue', 'green', 'red', 'orange', 'purple', 'pink', 'brown', 'gray']
for i, col in enumerate(sj_pillow_readings_df.columns[1:]):
    line.line(sj_pillow_readings_df['time'], sj_pillow_readings_df[col], legend_label=col, line_width=2, color=colors[i % len(colors)])

df_json = sj_pillow_readings_df.to_json(orient='columns')

# Modify the callback to update the p chart
callback_2 = CustomJS(args=dict(source=source, p=line, df=df_json), code="""
    const indices = cb_obj.indices;
    if (indices.length === 0) return;
    let selectedSnowPillows = [];
    let selectedIds = []; // New array to store the ids
    for (let i = 0; i < indices.length; i++) {
        const index = indices[i];
        const data = source.data;
        const selectedSnowPillow = data['snow_pillow'][index];
        selectedSnowPillows.push(selectedSnowPillow);
        selectedIds.push(data['snow_pillow'][index]); // Add the id to the new array
    }
                    
    console.log(selectedIds);
                      
    console.log(selectedIds);
    // Get the data for the line chart
    // const df = JSON.parse(df_json);
    const x_values = df['time'];
    // Create a new data source with the selected snow pillows
    const new_data = {};
    new_data['time'] = x_values;
    for (let i = 0; i < selectedIds.length; i++) {
        const id = selectedIds[i];
        new_data[id] = df[id];
    }
    // Update the line chart's data source
    p.data = new_data;
    p.change.emit();
""")
source.selected.js_on_change("indices", callback_2)
# Display the updated plots
st.bokeh_chart(column(scatter, line), use_container_width=True)


# New Code 

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
