import gc
import logging
import time
import numpy as np
import pandas as pd
from app.scripts.tsk_param import TskParam
from app import models
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d


# Function to split data on train and test
def split_data_by_id(data_ts_df: pd.DataFrame, test_percent):
    """
    sim is option that artifically simulate signals if data set is too small
    ! Use only if data signal belongs to range from 5 to 10 per class !
    """

    # Detect if the input dataframe includes simulated data
    includes_simulation = True if 'simulation' in data_ts_df.columns and data_ts_df['simulation'].any() else False

    final_selected_df = data_ts_df

    if includes_simulation:
        # exclude sim signal
        sim_mask = pd.isna(data_ts_df['simulation'])
        data_ts_df = data_ts_df.loc[sim_mask]
        with_sim_df = data_ts_df.loc[~sim_mask]

    if 'label' in data_ts_df.columns:
        # Gruppiere nach 'cycle_id' und erhalte die Klassenverteilung
        cycle_id_classes = data_ts_df.groupby('cycle_id')['label'].unique().reset_index()

        # Erhalte alle Klassifikationen
        classifications = data_ts_df['label'].unique()

        # Initialisiere DataFrames für Training und Test
        train_ids = set()
        test_cycle_ids = set()

        # Versuche sicherzustellen, dass jede Klasse in beiden Sets vorkommt
        for classification in classifications:
            # Daten für die aktuelle Klassifikation
            class_cycle_ids = cycle_id_classes[cycle_id_classes['label'].apply(lambda x: classification in x)]

            # Zufällige Aufteilung der 'cycle_id's für die aktuelle Klassifikation
            train_class_cycle_ids, test_class_cycle_ids = train_test_split(
                class_cycle_ids['cycle_id'],
                test_size=test_percent,
                random_state=42
            )

            # Füge die 'cycle_id's zu den Trainings- und Test-IDs hinzu
            train_ids.update(train_class_cycle_ids)
            test_cycle_ids.update(test_class_cycle_ids)
    else:
        unique_cycle_ids = data_ts_df['cycle_id'].unique()

        # Directly split unique_cycle_ids into train and test using train_test_split
        train_ids, test_cycle_ids1 = train_test_split(unique_cycle_ids, test_size=test_percent, random_state=42)

    train_mask = data_ts_df['cycle_id'].isin(train_ids)
    train_data = data_ts_df.loc[train_mask]
    test_data = data_ts_df.loc[~train_mask]

    if includes_simulation:
        train_data = pd.concat([train_data, with_sim_df], ignore_index=True)

    return test_data, train_data, final_selected_df


def upsample_data(selected_ts_df, factor_of_upsample):
    """
    This function linearly interpolates the time series 'value' column to add intermediate data points between original points based on the 'factor_of_upsample'.
    The main purpose is to increase the size of the dataset without adding new artefacts to the data.
    The function has been made available for when there are not enough data points to adequately extract all of the necesssary features.
    This function does not add additional information to the time series, it only interpolates values between two original time stamps, and associates these
    interpolated values to intermediate timestamps.

    For example, if selected_ts_df has data acquired in 0.01 s intervals (freq=100Hz), and the factor_of_upsample is 2, then there will be an additional datapoint added
    that lie on the linear interpolation between each of the two points at intervals of 0.005 s (new freq=200Hz).

    Note: Function is not currently used in processing data.

    :param selected_ts_df: Time series data with columns including 'delta_time', 'value', 'cycle_id', 'label'.
    :param factor_of_upsample: integer indicating the increase in the frequency of data acquisition for each cycle.
                                The function uses this factor to determine the number of intermediate points to insert between original data points.

    :return all_cycles_upsampled_df: Final selected time series df containing interpolated 'value' with additional timestamps for all cycles.
                                Returns original input DataFrame if any exception/error occurs.
    """

    try:
        # Check if DataFrame is empty
        if selected_ts_df.empty:
            raise ValueError("upsample_data: Input DataFrame is empty")

        # Check if required columns exist
        required_columns = ['value', 'delta_time', 'cycle_id']
        missing_columns = set(required_columns) - set(selected_ts_df.columns)
        if missing_columns:
            raise KeyError(f"Missing columns: {', '.join(missing_columns)}")

        all_cycle_data_grouped = selected_ts_df.groupby('cycle_id')
        all_cycles_upsampled_df = pd.DataFrame()

        # For each cycle
        for _, this_cycle_df in all_cycle_data_grouped:

            # Check if the cycle has sufficient data points for interpolation
            if len(this_cycle_df) < 2:
                continue  # Skip cycles with less than 2 data points

            # Extract other columns excluding 'delta_time' and 'value'
            other_columns = this_cycle_df.drop(columns=['value', 'delta_time'])
            other_columns_value_array = other_columns.values[0]

            # Interpolate values using linear interpolation
            new_times = np.linspace(this_cycle_df['delta_time'].iloc[0],
                                    this_cycle_df['delta_time'].iloc[-1], (len(this_cycle_df) - 1) * factor_of_upsample + 1)
            interpolated_values = interp1d(this_cycle_df['delta_time'], this_cycle_df['value'], kind='linear')(new_times)

            # Create DataFrame for upsampled cycle
            value_time_cycle_df = pd.DataFrame({
                'value': interpolated_values,
                'delta_time': new_times
            })

            # Concatenate the value_time_cycle_df with an increased row length other_columns (but increase the number of rows in other columns to match those of the value_time_cycle_df)
            upsampled_cycle_df = pd.concat([value_time_cycle_df, pd.DataFrame([other_columns_value_array] *
                                           len(value_time_cycle_df), columns=other_columns.columns)], axis=1)

            # Add this cycles upsampled df to the tota
            all_cycles_upsampled_df = pd.concat([all_cycles_upsampled_df, upsampled_cycle_df])

        return all_cycles_upsampled_df

    except Exception as e:
        print("An error occurred:", e)
        return selected_ts_df   # Return original input DataFrame


def downsample_data(selected_ts_df, factor_of_downsample):
    """
    This function downsamples the input data, and it does not add information to the original time series.
    It simply removes data points by skipping a certain number of points determined by the factor_of_downsample.
    For example, if you wanted to half the size of each cycle's data, factor_of_downsample = 2. This would eliminate every second data point.
    Note: Function is not currently used in processing data.

    :param selected_ts_df: Time series data with columns including 'delta_time', 'value', 'cycle_id', 'label'.
    :param factor_of_downsample: integer indicating the total decrease in the length of the selected_ts_df, and the number of elements to skip when downsampling.
                                The function uses this factor to determine how many points to skip in the original selected_ts_df.

    :return downsampled_df: Final selected time series df downsampled
                            Returns original input DataFrame if any exception/error occurs.
    """
    try:
        # Group data by 'cycle_id'
        cycle_grouped_data = selected_ts_df.groupby('cycle_id')

        # Check if DataFrame is empty
        if cycle_grouped_data.ngroups == 0:
            raise ValueError("Input DataFrame is empty")

        # Downsample - skip a certain number of data points to reduce the dataset per cycle
        downsampled_list = [group.iloc[::factor_of_downsample] for _, group in cycle_grouped_data]

        # Concatenate downsampled data
        downsampled_df = pd.concat(downsampled_list)

        return downsampled_df

    except Exception as e:
        # Handle all errors
        print("An unexpected error occurred:", e)
        return selected_ts_df  # Return original input DataFrame


def label_train_data(ts_df: pd.DataFrame, cycles: set[models.CycleData]):
    """
    Function for labeling the data if no classification is present in the database.
    Set the 'label' column on the input DataFrame based on the conditions specified in the function.

    :param df: the input dataframe.
    :param cycles: set of all relevant cycledata to load metadata from.

    :return: pd.DataFrame with 1 (good) or 0 (bad) in 'label' column.
    :return type: DataFrame
    """
    if ts_df[ts_df['label'] != -1].empty:
        # No labeled data, label them by our selfe

        cycles_dict = {cycle.id: cycle for cycle in cycles}

        # Iterate through each CycleData object in the dictionary
        for cycle in cycles_dict.values():
            # Convert all keys in the cycleConfig dictionary to lowercase
            cycle.cycleConfig = {k.lower(): v for k, v in cycle.cycleConfig.items()}

        # Create a DataFrame from the cycles dictionary for efficient merging
        cycles_df = pd.DataFrame({
            'cycle_id': list(cycles_dict.keys()),
            'load_torque_mnm': [cycle.cycleConfig['loadtorque'] for cycle in cycles_dict.values()],
            'spike_torque_mnm': [cycle.cycleConfig['spiketorque'] for cycle in cycles_dict.values()],
            'spike_start_time_ms': [cycle.cycleConfig['spikestarttime'] for cycle in cycles_dict.values()]
        })

        # Merge ts_df with cycles_df on 'cycle_id'
        ts_df = ts_df.merge(cycles_df, on='cycle_id', how='left')

        # Calculate the absolute difference between load_torque_mnm and spike_torque_mnm and save it as a new column into the original ts_df
        ts_df['abs_difference_load_spike'] = abs(ts_df['load_torque_mnm'].astype(int) - ts_df['spike_torque_mnm'].astype(int))
        # ts_df['abs_difference_load_spike'] = (ts_df['load_torque_mnm'].astype(int) - ts_df['spike_torque_mnm'].astype(int)).abs()

        # 1.7 Condition of labels assigned
        dict_condition_for_good = {
            'spike_start_time_ms': ('==', 0),
            'spike_torque_mnm': ('<=', 148)
        }
        # dict_condition_for_good = {
        #     'load_torque_mnm': ('<', 150)
        # }
        # dict_condition_for_good = {
        #     'abs_difference_load_spike': ('<', 20)
        # }

        conditions = []
        for col_name, (col_sign, col_threshold) in dict_condition_for_good.items():
            conditions.append(f"(ts_df['{col_name}'].astype(float) {col_sign} {col_threshold})")

        condition_string = " & ".join(conditions)

        # Populates the label column with 1=good or 0=bad based on the conditions
        ts_df['label'] = np.where(eval(condition_string), 1, 0)
        ts_df.drop(columns=['load_torque_mnm', 'spike_torque_mnm', 'spike_start_time_ms', 'abs_difference_load_spike'], inplace=True)
    return ts_df


def process_data(ts_df):
    """
    Function for preprocessing data for either training or prediction.

    Takes the dataframe from the query response or csv file and shapes it for later stages:
        - Changes df['_time'] from datetime64 to float64 and converts the value to deltaTime (s)
        - Selects columns with names containing 'act_current' and 'delta_time' and 'label' (if training)
        - Fixes time lags in the data
        - If training, splits the data in 20% test data, 80% training data. Simulates data if 5 <= Nsamples <= 10 for negative or positive cases


    :param df: the input dataframe from the database query.
    :type df: pandas.core.frame.DataFrame
    :param training: If to process for model training/Creation then True. Model prediction/application = False.
    :type training: bool

    :return: If training, final_selected_df, test_data, train_data dataframes containing the selected data, test data and training sets.
             If prediction, final_selected_df.
    :return type: tuple or pandas.core.frame.DataFrame
    """
    # Create column delta_time copying column 'timestamp' already converted to datatime
    ts_df['delta_time'] = ts_df['timestamp']

    # Creates a vector of equal length than the entire dataset with the min for each cycle_id
    min_timestamps = ts_df.groupby('cycle_id')['delta_time'].transform('min')

    # Subtracts min_timestamps and converts to seconds
    ts_df['delta_time'] = (ts_df['delta_time'] - min_timestamps) / np.timedelta64(1, 's')

    desired_columns = ['delta_time', 'value', 'label', 'cycle_id']
    existing_columns = [col for col in desired_columns if col in ts_df.columns]
    final_selected_ts_df = ts_df[existing_columns]

    return final_selected_ts_df


def process_data_predict(ts_df, reference_ts_df=None,  tsk_param: TskParam = None):
    """
    Function for preprocessing data for either training or prediction.

    Takes the dataframe from the query response or csv file and shapes it for later stages:
        - Changes df['_time'] from datetime64 to float64 and converts the value to deltaTime (s)
        - Selects columns with names containing 'value' and 'delta_time' and 'label' (if training)
        - Fixes time lags in the data
        - If training, splits the data in 20% test data, 80% training data. Simulates data if 5 <= Nsamples <= 10 for negative or positive cases


    :param df: the input dataframe from the database query.
    :type df: pandas.core.frame.DataFrame
    :param training: If to process for model training/Creation then True. Model prediction/application = False.
    :type training: bool

    :return: If training, final_selected_df, test_data, train_data dataframes containing the selected data, test data and training sets.
             If prediction, final_selected_df.
    :return type: tuple or pandas.core.frame.DataFrame
    """
    # Create column delta_time copying column 'timestamp' converted to datatime
    ts_df['delta_time'] = pd.to_datetime(ts_df['timestamp'], format='mixed')

    # Creates a vector of equal length than the entire dataset with the min for each cycle_id
    min_timestamps = ts_df.groupby('cycle_id')['delta_time'].transform('min')

    # Subtracts min_timestamps and converts to seconds
    ts_df['delta_time'] = (ts_df['delta_time'] - min_timestamps) / np.timedelta64(1, 's')

    desired_columns = ['delta_time', 'value', 'label', 'cycle_id']
    existing_columns = [col for col in desired_columns if col in ts_df.columns]
    final_selected_ts_df = ts_df[existing_columns]

    return final_selected_ts_df
