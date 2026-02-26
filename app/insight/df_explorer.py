import pandas as pd


def df_metadata(df):
    # Ensure 'timestamp' column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Group by 'cycle_id' and calculate the number of data points and signal durations
    signal_info = df.groupby('cycle_id').agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        num_datapoints=('point_id', 'size')
    )

    # Calculate signal duration in seconds
    signal_info['signal_duration'] = (signal_info['end_time'] - signal_info['start_time']).dt.total_seconds()

    # Dynamically bin the signal durations (time)
    duration_bins = pd.qcut(signal_info['signal_duration'], q=5, duplicates='drop')  # 5 quantile-based bins
    duration_bins_count = signal_info.groupby(duration_bins).size()

    # Dynamically bin the number of data points
    datapoint_bins = pd.qcut(signal_info['num_datapoints'], q=5, duplicates='drop')  # 5 quantile-based bins
    datapoint_bins_count = signal_info.groupby(datapoint_bins).size()

    # Summary output
    metadata = {
        'total_signals': len(signal_info),
        'duration_bins_count': duration_bins_count,
        'datapoint_bins_count': datapoint_bins_count
    }

    return metadata
