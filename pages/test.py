# Create line plot
        line_source = ColumnDataSource(data=dict())
        line = figure(
            title="Line Chart for Selected Snow Pillow",
            width=700,
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



# Create line plot
        line_source = ColumnDataSource(data=dict())
        line = figure(
            title="Line Chart for Selected Snow Pillow",
            width=700,
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
