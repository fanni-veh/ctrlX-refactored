import pandas as pd
import plotly.io as pio
from plotly import express as px
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix

pio.templates.default = "plotly_dark"


def ts_visualisation(dataframe_ts_df,
                     title_pie='Label Distribution',
                     title_unknown_signals='Unknown Signals',
                     title_bad_signals='Bad Quality Signals',
                     title_good_signals='Good Quality Signals',
                     title_cycles='Cycles',
                     title_scatter_x='ΔTime (s)',
                     title_scatter_y='value'):
    """
    Function to produce figures of timeseries data.
    If signals are labeled, it produces 2 plots (good/bad)

    :param dataframe_ts_df: the collection of time series data to plot
    :return: array of figures (1 or 2 figures)
    """

    # Assuming label_dict is defined elsewhere to map labels to human-readable names
    label_dict = {-1: title_unknown_signals, 0: title_bad_signals, 1: title_good_signals}

    # Pre-sort entire dataframe once instead of sorting each group
    if not dataframe_ts_df.empty:
        dataframe_ts_df = dataframe_ts_df.sort_values(['cycle_id', 'delta_time'])

    if 'label' in dataframe_ts_df.columns:

        # Split the data into DataFrames based on label
        grouped_data = dataframe_ts_df.groupby('label', sort=False)  # sort=False for performance

        fig_list_out = []
        # Pre-calculate global y-range for all figures at once
        global_y_min = dataframe_ts_df['value'].min()
        global_y_max = dataframe_ts_df['value'].max()
        y_range = [global_y_min, global_y_max]

        for label, df_label in grouped_data:
            # Create a new Plotly figure for each label
            fig_signal = go.Figure()

            # Group the data by 'id' to process each time series
            time_series_groups = df_label.groupby('cycle_id', sort=False)

            # Iterate over each time series group
            traces_data = []
            for cycle_id, group in time_series_groups:
                name = str(cycle_id)

                traces_data.append({
                    'x': group['delta_time'].values,
                    'y': group['value'].values,
                    'name': name
                })

            # Add all traces at once using list comprehension
            fig_signal.add_traces([
                go.Scatter(x=trace['x'], y=trace['y'], name=trace['name'])
                for trace in traces_data
            ])

            fig_signal.update_layout(
                title=label_dict[label],
                xaxis_title=title_scatter_x,
                yaxis_title=title_scatter_y,
                yaxis_range=y_range,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',  # make background transparent
                legend_title_text=title_cycles,
                meta={'label_id': label}
            )
            fig_list_out.append(fig_signal)

        # Optimized pie chart creation using pandas value_counts
        label_counts = dataframe_ts_df.groupby('label')['cycle_id'].nunique().reset_index()
        label_counts.columns = ['Label', 'Count']
        label_counts['Label'] = label_counts['Label'].map(label_dict)

        # colors
        colors = ['red' if label == label_dict[0] else 'green' if label == label_dict[1] else 'gray'
                  for label in label_counts['Label']]

        fig_balance_signal = go.Figure(data=[
            go.Pie(
                labels=label_counts['Label'],
                values=label_counts['Count'],
                hole=.6,
                hoverinfo='label+percent',
                textinfo='value',
                textfont_size=20,
                marker=dict(colors=colors)
            )
        ])
        fig_balance_signal.update_layout(
            title=title_pie,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
    else:
        fig = go.Figure()
        # id -> cycle_id
        cycle_groups = dataframe_ts_df.groupby('cycle_id', sort=False)
        traces = [
            go.Scatter(
                x=group['delta_time'].values,
                y=group['value'].values,
                name=str(cycle_id)
            )
            for cycle_id, group in cycle_groups
        ]
        fig.add_traces(traces)

        fig.update_layout(
            title_text='Time Series',
            xaxis_title=title_scatter_x,
            yaxis_title=title_scatter_y,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig_list_out = [fig]
        fig_balance_signal = None

    return fig_list_out, fig_balance_signal


# Create the confidence gauge
def create_confidence_gauge(confidence_percentage, prediction):
    """
    Function to produce a figure of a plotly gauge to demonstrate confidence of the prediction.

    :param confidence_percentage: the percentage of confidence of the prediction (e.g. 60.2%)
    :param prediction: the prediction (1 or 0)

    Note: This assumes good = 1, bad = 0.

    :return: the gauge figure
    """
    # Calculate the needle position on a scale of 0 to 100
    if prediction == 1:
        needle_position = 50 + (confidence_percentage / 2)
    else:
        needle_position = 50 - (confidence_percentage / 2)

    # Define the gradient from red to green, fading to transparent past the needle
    n_steps = 100  # Number of gradient steps
    gradient_steps = []
    for i in range(n_steps):
        if i <= needle_position:
            color = f"rgb({255 - int(2.55 * i)}, {int(2.55 * i)}, 0)"
        else:
            color = f"rgba({255 - int(2.55 * i)}, {int(2.55 * i)}, 0, 0.1)"  # Fade to transparent
        gradient_steps.append({'range': [i, i + 1], 'color': color})

    # Define the gauge chart
    fig = go.Figure()

    # Add gauge indicator
    fig.add_trace(go.Indicator(
        mode="gauge",
        value=needle_position,
        gauge={
            'axis': {'range': [0, 100], 'visible': False},  # Hide the axis
            'bar': {'color': "rgba(0,0,0,0)"},  # Hide the default bar
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'bordercolor': "rgba(0,0,0,0)",
            'steps': gradient_steps,
            'threshold': {
                'line': {'color': "gray", 'width': 20},
                'thickness': 0,
                'value': needle_position
            }
        },
        domain={'x': [0.3, 0.7], 'y': [0.5, 0.9]}  # Adjust this to reduce size
    ))

    # Add annotation for the word "Confidence"
    fig.add_annotation(
        text="Confidence",
        font={'size': 14, 'color': 'white'},  # Adjust font size as needed
        showarrow=False,
        align='center',
        x=0.5,  # Center the annotation horizontally
        y=0.7,  # Position the annotation above the number
        xref='paper',
        yref='paper'
    )

    # Add number indicator in the center
    fig.add_trace(go.Indicator(
        mode="number",
        value=confidence_percentage,
        number={'suffix': '%', 'font': {'size': 24, 'color': 'white'}},  # Adjust font size as needed
        domain={'x': [0.45, 0.55], 'y': [0.55, 0.7]}  # Adjust this to center and size the number
    ))

    # Update layout to remove tick colors and marks and set background color
    fig.update_layout(
        margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'}
    )

    return fig


def create_performance_plots(y_validation, y_predicted):
    """
    This function creates the plots used to visualise the performance of the models.
        - Creates a confusion matrix plot
        - Creates a bar chart
        - Creates a snake chart
    :param y_true: truth labels for test data set
    :param y_predicted: predicted labels for test data set
    :return: fig_matrix, fig_bar, fig_sankey
    """

    # Define custom labels for the x-axis and y-axis
    x_labels = ["Bad Signal", "Good Signal"]
    y_labels = ["Bad Signal", "Good Signal"]

    # True Positive (TP): The number of instances correctly predicted as positive (Good).
    # True Negative (TN): The number of instances correctly predicted as negative (Bad).
    # False Positive (FP): The number of instances incorrectly predicted as positive (Good) when they are actually negative (Bad).
    # False Negative (FN): The number of instances incorrectly predicted as negative (Bad) when they are actually positive (Good).
    conf_matrix = confusion_matrix(y_validation, y_predicted, labels=[0, 1])
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=x_labels, index=y_labels)

    # CM = [TN, FP]
    #      [FN, TP]
    tn, fp, fn, tp = conf_matrix.ravel()

    # Confusion Matrix
    fig_matrix = px.imshow(conf_matrix_df, text_auto=True, labels={'x': 'Predicted', 'y': 'True'}, color_continuous_scale=['black', 'red', 'red'])
    fig_matrix.update_coloraxes(showscale=False)
    fig_matrix.update_xaxes(side='top')
    fig_matrix.update_traces(textfont_size=14)
    fig_matrix.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    # Bar Chart
    fig_bar = go.Figure(
        data=[
            go.Bar(x=["Bad Signal", "Good Signal"], y=[tn, tp], name='Correctly Predicted', marker_color='teal'),
            go.Bar(x=["Bad Signal", "Good Signal"], y=[fp, fn], name='Incorrectly Predicted', marker_color='coral')
        ],
        layout=go.Layout(
            title='Matrix Breakdown',
            yaxis=dict(title='Count'),
            barmode='stack',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )

    # Sankey chart
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["<span style='color: black;text-shadow: none;font-size:16px;'>True Good Signal</span>",
                   "<span style='color: black;text-shadow: none;font-size:16px;'>True Bad Signal</span>",
                   "<span style='color: black;text-shadow: none;font-size:16px;'>Predicted Good Signal</span>",
                   "<span style='color: black;text-shadow: none;font-size:16px;'>Predicted Bad Signal</span>"],
            color=["teal", "midnightblue", "teal", "midnightblue"]
        ),
        link=dict(
            source=[0, 0, 1, 1],
            target=[2, 3, 2, 3],
            value=[tp, fn, fp, tn],
            color=["#CFC",  "#CFC",  "lightblue", "lightblue"]
        ))])

    fig_sankey.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig_matrix_html = fig_matrix.to_html(full_html=False, include_plotlyjs=False)
    fig_bar_html = fig_bar.to_html(full_html=False, include_plotlyjs=False)
    fig_sankey_html = fig_sankey.to_html(full_html=False, include_plotlyjs=False)

    return fig_matrix_html, fig_bar_html, fig_sankey_html


def create_health_history_plot(title: str, y_title: str, x_title: str, data: dict) -> go.Figure:
    """
    Function to create a health history plot from the provided data.

    :param data: Dict containing health history data with.
    :return: Plotly figure object representing the health history plot.
    """
    fig = go.Figure()

    color_palette = px.colors.qualitative.Set2
    color_map = {}
    sorted_categories = sorted(data.keys())

    for idx, trace_name in enumerate(sorted_categories):
        values = data[trace_name]
        if values:  # Ensure there is data to plot
            sorted_values = sorted(values, key=lambda x: x[0])
            timestamps, scores = zip(*sorted_values)  # Extract timestamps and scores
            if trace_name not in color_map:
                color_map[trace_name] = color_palette[idx % len(color_palette)]

            fig.add_trace(go.Scatter(
                x=list(timestamps),
                y=list(scores),
                mode="lines+markers",  # Connected line with markers
                marker=dict(size=6, color=color_map[trace_name]),
                line=dict(width=2, color=color_map[trace_name]),
                name=trace_name  # Use trace_name as label
            ))

    # Setting layout properties
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        margin={'t': 40},
        yaxis=dict(range=[0, 100]),  # Assuming scores are between 0 and 100
        legend=dict(
            orientation="h",  # Horizontale Ausrichtung
            x=0.5,  # center on X-axe
            y=1.2,  # above the plot
            xanchor="center",  # aligned to center
            yanchor="bottom"  # aligned to bottom
        )
    )

    return fig
