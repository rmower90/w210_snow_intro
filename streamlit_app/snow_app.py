### Import the package we need

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import geopandas as gpd
from netCDF4 import Dataset
import pathlib

st.set_page_config(
    page_title="Mo Data Sno Problem",
    page_icon="❄️",
    layout="centered"
)

### HERE IS WHERE WE BUILD THE DATAFRAMES WE USE TO CREATE THE VISUALIZATIONS ###

# Dataframe for Snow Pillow Coordinates
aso_site_name = 'USCASJ'
insitu_dir = '/Users/vqu/VirtualEnv/myVirtualEnv/Capstone/w210_snow_intro/streamlit_app/qa/'

# Function to load snow pillow data and locations
def load_pillow_nc(insitu_dir):
    obs_data = []
    for file in sorted(os.listdir(insitu_dir)):
        if file.endswith('.nc'):
            # Open the NetCDF file
            file_path = os.path.join(insitu_dir, file)
            dataset = Dataset(file_path, mode='r')
            
            # Extract data from the NetCDF file
            # Assuming you want to extract a specific variable, replace 'your_variable_name' with the actual variable name
            variable_name = 'your_variable_name'  # Replace with the actual variable name
            if variable_name in dataset.variables:
                data = dataset.variables[variable_name][:]
                obs_data.append(data)
            
            # Close the dataset
            dataset.close()
    insitu_locations = None
    for file in sorted(os.listdir(insitu_dir)):
        if 'obs_summary.shp' in file:
            insitu_locations = gpd.read_file(os.path.join(insitu_dir, file))
            insitu_locations = insitu_locations.set_crs('EPSG:4326')
            insitu_locations.rename(columns={'elevation_': 'elevation_m'}, inplace=True)
    return obs_data, insitu_locations
# Load data
obs_data, insitu_locations = load_pillow_nc(insitu_dir)
# Extract longitude and latitude
if insitu_locations is not None:
    insitu_locations['longitude'] = insitu_locations.geometry.x
    insitu_locations['latitude'] = insitu_locations.geometry.y
# Display the DataFrame with longitude and latitude
insitu_locations_df = insitu_locations[['id', 'longitude', 'latitude', 'elevation_m']]

# Table for Snow Pillow Readings
summaryTable_dir = '/Users/vqu/VirtualEnv/myVirtualEnv/Capstone/w210_snow_intro/streamlit_app/USCASJ/'
obs_threshold = 0.5
# Load data
df_sum_total = pd.read_csv(f'{summaryTable_dir}total.csv')
# df_sum_total = pd.read_csv(f'qa/pillow_orig_table.csv')
# Create list of pillows
all_pils = df_sum_total.columns.to_list()
all_pils.remove('time')
all_pils.remove('aso_mean_bins_mm')
# Create a dataframe that removes pillows with less than half of flight dates missing
drop_bool = ((df_sum_total[all_pils].isnull().sum(axis=0) / len(df_sum_total)) < obs_threshold).values
pils_removed = np.array(all_pils)[drop_bool]
df_dropped_pils = df_sum_total[pils_removed]

pils_list = df_dropped_pils.columns.tolist()

sp_time = []
sp_name = []
sp_mm = []

for i in pils_list:
  sp_time.append([x for x in df_sum_total['time']])
  sp_mm.append([x for x in df_sum_total[i]])
  sp_name.append([i]*len(df_sum_total))

# Add a categorical variable to the 3D data
df_3d = pd.DataFrame({
    'Date': pd.to_datetime(sum(sp_time, [])),
    'Value': sum(sp_mm, []),
    'Category': sum(sp_name, [])
})

### HERE IS WHERE WE BUILD THE STREAMLIT DASHBOARD ###

# Title
st.title('Mo Data Sno Problem')

# Intro
st.write("Lorem Ipsum - This dashboard provides an overview of snow pillow data from various locations. The data is used to analyze and visualize the snow depth and other related metrics. The dashboard includes several interactive plots and maps that allow users to explore the data in different ways.")

st.header('San Joaquin')

st.write("The San Joaquin River originates in the high-elevation Eastern Sierra Nevada mountain range, flowing southwest to the San Joaquin Valley floor, before turning northwest to its confluence with the Sacramento River at the Sacramento-San Joaquin Delta (Delta). The San Joaquin River has three major tributaries: the Merced, Tuolumne, and Stanislaus rivers. The Cosumnes (a tributary to Mokelumne River), Mokelumne, and Calaveras rivers also flow into the San Joaquin River where the river joins the tidally influenced Delta.")
st.write("We'll use San Joaquin River Basin for our demo as they have the most upcoming ASO flights available for validation")

st.subheader('Map Plot of Snow Pillows')
st.write("Start by getting a better understanding of the snow pillows locations for the basin")
# Sample data with latitude, longitude, and an identifier
data = {
    'latitude': insitu_locations_df['latitude'],
    'longitude': insitu_locations_df['longitude'],
    'id': insitu_locations_df['id'],
    'elevation_m': insitu_locations_df['elevation_m'],
    'size_metric': [10] * len(insitu_locations_df)
}
df = pd.DataFrame(data)
# Create a Plotly Express scatter map
fig = px.scatter_mapbox(
    df,
    lat="latitude",
    lon="longitude",
    text="id",  # Use the 'id' column for hover text
    color="elevation_m",  # Use 'elevation_m' for color
    color_continuous_scale=px.colors.cyclical.IceFire,
    size='size_metric',
    size_max=10,
    zoom=7.2,
    mapbox_style="carto-positron"
)

# Update layout for centering and margins
center_lat = (max(insitu_locations_df['latitude']) + min(insitu_locations_df['latitude']))/2
center_long = (max(insitu_locations_df['longitude']) + min(insitu_locations_df['longitude']))/2
fig.update_layout(
    mapbox=dict(
        center=dict(lat=center_lat, lon=center_long)
    ),
    margin={"r":0,"t":0,"l":0,"b":0}
)

# Update traces to set text font color to black
fig.update_traces(textfont=dict(color='black'))

# Display the figure in Streamlit with scroll zoom enabled
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# 3D Line Chart with Categories
st.subheader('Pillow Snow Measurements on ASO Flight Dates')
fig_3d_line = px.line_3d(
    df_3d, 
    x='Category', 
    y='Date', 
    z='Value', 
    color='Category',
)
fig_3d_line.update_layout(
    scene_camera=dict(
        eye=dict(x=1.8, y=2.1, z=1.8),
        center=dict(x=0, y=0, z=-0.5)
          # Adjust these values to set the desired starting view
    ),
    width=1200,  # Set the width of the plot
    height=700,   # Set the height of the plot
    scene=dict(
        xaxis=dict(
            title='Snow Pillows',
            showticklabels=False
        )
    )
)
st.plotly_chart(fig_3d_line)


# 3D Scatter Plot with Categorical X-axis and Date Dropdown
st.subheader('3D Scatter Plot with Categories')
fig_3d = px.scatter_3d(
    df_3d, 
    x='Category', 
    y='Date', 
    z='Value', 
    color='Category',
    title='3D Scatter Plot with Categories'
)
# Update the layout to set the initial camera view
fig_3d.update_layout(
    scene_camera=dict(
        eye=dict(x=1.8, y=2.1, z=1.8),
        center=dict(x=0, y=0, z=-0.5)
          # Adjust these values to set the desired starting view
    ),
    width=1200,  # Set the width of the plot
    height=800   # Set the height of the plot
)
# Update marker size
fig_3d.update_traces(marker=dict(size=3))
st.plotly_chart(fig_3d)

# Sidebar for date filtering
st.sidebar.header('Filter by Date')
start_date = st.sidebar.date_input('Start date', df_3d['Date'].min())
end_date = st.sidebar.date_input('End date', df_3d['Date'].max())

# Add a button to reset the date filter in the sidebar
if st.sidebar.button('Reset Date Filter'):
    start_date = df_2d['Date'].min()
    end_date = df_2d['Date'].max()