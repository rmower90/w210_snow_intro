import os
import streamlit as st
# from bokeh.io import show
import matplotlib.pyplot as plt
import base64

# import helper scripts
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import src.helper.load_data as load_data
import src.helper.plotting as plotting

st.set_page_config(page_title="Less Data Dashboard", layout="centered")

tab_conclusions, tab_imputation = st.tabs(["Conclusions", "Imputation"])
base_dir = os.path.dirname(os.path.abspath(__file__))  # /.../pages
root_dir = os.path.abspath(os.path.join(base_dir, "..")) # /.../ (one level up)

def get_pdf_as_base64(file_path):
    with open(file_path, "rb") as file:
        pdf_bytes = file.read()
    # Encode the PDF file's bytes to a base64 string.
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    return base64_pdf

# Path to your PDF file
pdf_file_1 = "../w210_snow_intro/docs/futureresearchabstracts.pdf"
pdf_file_2 = "../w210_snow_intro/docs/cheatsheet.pdf"

with open(pdf_file_1, "rb") as f:
    pdf_data_1 = f.read()
    
    
base64_pdf_1 = get_pdf_as_base64(pdf_file_1)
base64_pdf_2 = get_pdf_as_base64(pdf_file_2)

# Create an HTML iframe to embed the PDF
pdf_display_1 = f'<iframe src="data:application/pdf;base64,{base64_pdf_1}" width="900" height="1000" type="application/pdf"></iframe>'
pdf_display_2 = f'<iframe src="data:application/pdf;base64,{base64_pdf_2}" width="900" height="1000" type="application/pdf"></iframe>'

with tab_conclusions:
    ## Section 4 -- Testing/Results

    st.title("Results and Evaluation")
    st.write("Here we are comparing model inference results in the San Joaquin. We compare the best ***MLR Drop NaNs*** and ***Predict NaNs*** models to our label (***ASO***) and benchmarks (***SNODAS*** and ***UASWE***). The flights occured on February 26th, 2025 and March 25th, 2025. Future work will extend a similar comparison to the Tuolumne basin.")
    st.write("- First, we compare the models for each elevation interval, ***separately***.")
    st.write("- Next, we compare the models across elevation intervals and generate statistics relevant to our ***KPI*** metrics.")
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
    plt.suptitle(f'{selected_basin_2} - {selected_elevation_2} Mean SWE Comparison',fontweight = 'bold',fontsize = 24)
    plt.tight_layout()
    st.pyplot(fig)
    ### Model -------------------------------------------- \
    st.write("## Model Comparison")


    # Dropdown for selecting elevation bin
    date_options = ['2025-02-26','2025-03-25']
    selected_test_date = st.selectbox('Test Dates', date_options,index=1)

    fig = plotting.create_model_comparison(root_dir,aso_site_name,date_str = selected_test_date.replace('-',''))
    st.pyplot(fig)
    
with tab_imputation:
    st.download_button(
        label="Download PDF",
        data=pdf_data_1,
        file_name="futureresearchabstracts.pdf",
        mime="application/pdf",
    )
    
    st.markdown(pdf_display_1, unsafe_allow_html=True)
    st.markdown(pdf_display_2, unsafe_allow_html=True)
# # Create line plot
#         line_source = ColumnDataSource(data=dict())
#         line = figure(
#             title="Line Chart for Selected Snow Pillow",
#             width=700,
#             height=500,
#             x_axis_type="datetime",
#             y_range=Range1d(start=0, end=3500),
#         )
#         line.xaxis.formatter = DatetimeTickFormatter(
#             days="%d %b %Y",  # Format for day-level ticks
#             months="%b %Y",  # Format for month-level ticks
#             years="%Y",  # Format for year-level ticks
#         )

#         # Modify the callback to update the p chart
#         select_SnowPillow = CustomJS(
#             args=dict(source=source, line_source=line_source, spr=pillow_readings),
#             code="""
#             const indices = cb_obj.indices;
#             if (indices.length === 0) return;
#             let selectedSnowPillows = [];
#             let x_values = [];
#             let y_values = [];
#             let selectedData = {x: spr['x']};
#             for (let i = 0; i < indices.length; i++) {
#                 const index = indices[i];
#                 const data = source.data;
#                 const selectedSnowPillow = data['snow_pillow'][index];
#                 selectedSnowPillows.push(selectedSnowPillow);
#                 console.log(selectedSnowPillows)
#             }
#             for (let i = 0;i < selectedSnowPillows.length; i++) {
#                 selectedData[`${selectedSnowPillows[i]}`] = spr[`${selectedSnowPillows[i]}`]
#             }
#             line_source.data = selectedData;
#             line_source.change.emit();
#         """,
#         )
#         source.selected.js_on_change("indices", select_SnowPillow)

#         # Create time series line plots
#         lines = {}
#         pillow_name_list = tm_pillow_df["id"]
#         for name in pillow_name_list:
#             if name != "time":
#                 lines[name] = line.line(
#                     "x",
#                     name,
#                     source=line_source,
#                     name=name,
#                     color=color_mapping[name],
#                     line_width=2,
#                 )

#         # Add Hover tooltip for each line item
#         for name, renderer in lines.items():
#             hover = HoverTool(
#                 renderers=[renderer],  # Apply this hover tool only to the specific line
#                 tooltips=[
#                     ("Pillow ID", name),
#                     ("Date", "@x{%F}"),  # Display the date in YYYY-MM-DD format
#                     (
#                         "Units: mm",
#                         f"@{name}",
#                     ),  # Display the value for the specific line
#                 ],
#                 formatters={
#                     "@x": "datetime",  # Use 'datetime' formatter for the x value
#                 },
#                 mode="mouse",  # Show tooltip for the closest data point to the mouse
#             )
#             line.add_tools(hover)

#         # ASO flight data
#         tm_aso_df = pd.read_csv("data/aso/USCATM/uscatm_aso_sum.csv")
#         tm_aso_df["aso_mean_bins_mm"] = tm_aso_df["aso_mean_bins_mm"].apply(
#             lambda x: f"{x:.2f}" if pd.notnull(x) else x
#         )

#         # Plot points for ASO flights
#         points_data = {
#             "time": pd.to_datetime(tm_aso_df["time"]),  # Example dates
#             "value": tm_aso_df["aso_mean_bins_mm"],  # Example values
#             "image_url": [
#                 f"data/aso/USCATM/images/plot{i}.png" for i in range(0, len(tm_aso_df))
#             ],
#         }
#         points_source = ColumnDataSource(points_data)

#         # Hover tooltip for ASO plot points
#         aso_hover = HoverTool(
#             renderers=[
#                 line.circle(
#                     "time",
#                     "value",
#                     source=points_source,
#                     size=10,
#                     color="blue",
#                     line_color="black",
#                     line_width=1,
#                     legend_label="ASO Flights",
#                 )
#             ],
#             tooltips=[
#                 ("Date", "@time{%F}"),  # Display the date in YYYY-MM-DD format
#                 ("Value (mm)", "@value"),  # Display the value
#             ],
#             formatters={
#                 "@time": "datetime",  # Use 'datetime' formatter for the time value
#             },
#             mode="mouse",  # Show tooltip for the closest data point to the mouse
#         )
#         # Add the hover tool to the line figure
#         line.add_tools(aso_hover)



# # Create line plot
#         line_source = ColumnDataSource(data=dict())
#         line = figure(
#             title="Line Chart for Selected Snow Pillow",
#             width=700,
#             height=500,
#             x_axis_type="datetime",
#             y_range=Range1d(start=0, end=3500),
#         )
#         line.xaxis.formatter = DatetimeTickFormatter(
#             days="%d %b %Y",  # Format for day-level ticks
#             months="%b %Y",  # Format for month-level ticks
#             years="%Y",  # Format for year-level ticks
#         )

#         # Modify the callback to update the p chart
#         select_SnowPillow = CustomJS(
#             args=dict(source=source, line_source=line_source, spr=pillow_readings),
#             code="""
#             const indices = cb_obj.indices;
#             if (indices.length === 0) return;
#             let selectedSnowPillows = [];
#             let x_values = [];
#             let y_values = [];
#             let selectedData = {x: spr['x']};
#             for (let i = 0; i < indices.length; i++) {
#                 const index = indices[i];
#                 const data = source.data;
#                 const selectedSnowPillow = data['snow_pillow'][index];
#                 selectedSnowPillows.push(selectedSnowPillow);
#                 console.log(selectedSnowPillows)
#             }
#             for (let i = 0;i < selectedSnowPillows.length; i++) {
#                 selectedData[`${selectedSnowPillows[i]}`] = spr[`${selectedSnowPillows[i]}`]
#             }
#             line_source.data = selectedData;
#             line_source.change.emit();
#         """,
#         )
#         source.selected.js_on_change("indices", select_SnowPillow)

#         # Create time series line plots
#         lines = {}
#         pillow_name_list = sj_pillow_df["id"]
#         for name in pillow_name_list:
#             if name != "time":
#                 lines[name] = line.line(
#                     "x",
#                     name,
#                     source=line_source,
#                     name=name,
#                     color=color_mapping[name],
#                     line_width=2,
#                 )

#         # Add Hover tooltip for each line item
#         for name, renderer in lines.items():
#             hover = HoverTool(
#                 renderers=[renderer],  # Apply this hover tool only to the specific line
#                 tooltips=[
#                     ("Pillow ID", name),
#                     ("Date", "@x{%F}"),  # Display the date in YYYY-MM-DD format
#                     (
#                         "Units: mm",
#                         f"@{name}",
#                     ),  # Display the value for the specific line
#                 ],
#                 formatters={
#                     "@x": "datetime",  # Use 'datetime' formatter for the x value
#                 },
#                 mode="mouse",  # Show tooltip for the closest data point to the mouse
#             )
#             line.add_tools(hover)

#         # ASO flight data
#         sj_aso_df = pd.read_csv("data/aso/USCASJ/uscasj_aso_sum.csv")
#         sj_aso_df["aso_mean_bins_mm"] = sj_aso_df["aso_mean_bins_mm"].apply(
#             lambda x: f"{x:.2f}" if pd.notnull(x) else x
#         )

#         # Plot points for ASO flights
#         points_data = {
#             "time": pd.to_datetime(sj_aso_df["time"]),  # Example dates
#             "value": sj_aso_df["aso_mean_bins_mm"],  # Example values
#             "image_url": [
#                 f"data/aso/USCASJ/images/plot{i}.png" for i in range(0, len(sj_aso_df))
#             ],
#         }
#         points_source = ColumnDataSource(points_data)

#         # Hover tooltip for ASO plot points
#         aso_hover = HoverTool(
#             renderers=[
#                 line.circle(
#                     "time",
#                     "value",
#                     source=points_source,
#                     size=10,
#                     color="blue",
#                     line_color="black",
#                     line_width=1,
#                     legend_label="ASO Flights",
#                 )
#             ],
#             tooltips=[
#                 ("Date", "@time{%F}"),  # Display the date in YYYY-MM-DD format
#                 ("Value (mm)", "@value"),  # Display the value
#             ],
#             formatters={
#                 "@time": "datetime",  # Use 'datetime' formatter for the time value
#             },
#             mode="mouse",  # Show tooltip for the closest data point to the mouse
#         )
#         # Add the hover tool to the line figure
#         line.add_tools(aso_hover)
