import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.layouts import column
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models import WMTSTileSource
import pandas as pd
import math
import numpy as np

st.title("Mo Snow Less Data Hello World")

# Load snowpillow data from the CSV file
sj_pillow_df = pd.read_csv('sj_pillow_locations.csv')

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

# Create bar plot
bar_source = ColumnDataSource(data=dict(snow_pillow=[], counts=[]))
bar = figure(x_range=list(sj_pillow_df['id'].unique()),
             title="Snow Pillow Count for Selected Point",
             width=400, height=400)
bar.vbar(x='snow_pillow', top='counts', width=0.9, source=bar_source)

# Create line plot
line_source = ColumnDataSource(data=dict(x=[], y=[]))
line = figure(title="Line Chart for Selected Snow Pillow", width=400, height=400)
line.line('x', 'y', source=line_source)

# Callback function
callback = CustomJS(args=dict(source=source, bar_source=bar_source, line_source=line_source), code="""
    const indices = cb_obj.indices;
    if (indices.length === 0) return;
    let snow_pillows = [];
    let counts = [];
    let x_values = [];
    let y_values = [];
    for (let i = 0; i < indices.length; i++) {
        const index = indices[i];
        const data = source.data;
        const selectedSnowPillow = data['snow_pillow'][index];
        let count = 0;
        for (let j = 0; j < data['snow_pillow'].length; j++) {
            if (data['snow_pillow'][j] === selectedSnowPillow) {
                count++;
            }   
        }
        snow_pillows.push(selectedSnowPillow);
        counts.push(count);
        for (let j = 0; j < data['snow_pillow'].length; j++) {
            if (data['snow_pillow'][j] === selectedSnowPillow) {
                x_values.push(data['x'][j]);
                y_values.push(data['y'][j]);
            }
        }
    }
    bar_source.data = { snow_pillow: snow_pillows, counts: counts };
    bar_source.change.emit();
    line_source.data = { x: x_values, y: y_values };
    line_source.change.emit();
""")

source.selected.js_on_change("indices", callback)

# Display plots
st.bokeh_chart(column(scatter, bar, line), use_container_width=True)