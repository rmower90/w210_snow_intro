import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.layouts import column
import pandas as pd
import requests
from io import StringIO

st.title("Interactive Bokeh Layout in Streamlit")

# Make the request using requests
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
response = requests.get(url)

# Create a file-like object from the response content
csv_data = StringIO(response.content.decode('utf-8'))

# Read the CSV data into a pandas DataFrame
df = pd.read_csv(csv_data)

if df is not None:
    source = ColumnDataSource(data=df)

    # Create scatter plot
    scatter = figure(title="Iris Scatter Plot", tools="tap", width=400, height=400)
    scatter.circle("sepal_length", "sepal_width", size=10, source=source)

    # Create bar plot
    bar_source = ColumnDataSource(data=dict(species=[], counts=[]))
    bar = figure(x_range=list(df['species'].unique()),
                 title="Species Count for Selected Point",
                 width=400, height=400)
    bar.vbar(x='species', top='counts', width=0.9, source=bar_source)

    # JavaScript callback for interactivity
    callback = CustomJS(args=dict(source=source, bar_source=bar_source), code="""
        const indices = cb_obj.indices;
        if (indices.length === 0) return;
        const index = indices[0];
        const data = source.data;
        const selectedSpecies = data['species'][index];
        let count = 0;
        for (let i = 0; i < data['species'].length; i++) {
            if (data['species'][i] === selectedSpecies) {
                count++;
            }   
        }
        bar_source.data = { species: [selectedSpecies], counts: [count] };
        bar_source.change.emit();
    """)
    source.selected.js_on_change("indices", callback)

    # Display the plots using Streamlit's native Bokeh support
    st.bokeh_chart(column(scatter, bar), use_container_width=True)