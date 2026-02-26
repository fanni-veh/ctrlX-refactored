''' 
Data reduction pipeline

v0.1.1 - Initial version
v0.1.2 - Callbacks, HDBScan
v0.1.3 - Model plotting to log
v0.1.4 - __private to _protected. PCA params to metadata.
v0.1.5 - MinMax scaling option
v0.1.6 - Fisher score for input data quality
v0.1.7 - HDBScan input parameters
v0.1.8 - time_lags changed to preprocessing
v0.1.9 - Improved lag calculation and signal alignment. 
       - Added reduction_summary. 
       - Flexible this_model_cycle_ids_and_conf1_df wwith extra metrics. 
v0.1.10 - adjusted model geometry for axial and perpendicular spread factors.        
'''

# General
import logging
import time
import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
# import heapq
from scipy.ndimage import binary_closing
import yaml


# Internal
from app.pipeline.tools.features.feature_set1 import features_dict
from app.pipeline.tools.features.similarity_values import features_similarity
from app.pipeline.tools.features import feature_extraction_functions as fef
from app import RunState, RunTask, models, PredictionMetric
from app.pipeline.tools.hdbscan_clustering import HDBSCAN_Clustering
from app.pipeline.tools.onnx_converter import onnx_predict
from app.insight.ts_plotter import plot_simple_ts, plot_stacked_ts, plot_ts_and_xcorr
from app.insight.model_plotter import plot_model_2D, plot_model_3D
from app.insight.feature_plotter import plot_fisher_score, plot_feature_similarity_heatmap

# DR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from itertools import combinations
# from sklearn.decomposition import TruncatedSVD

# ML
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from catboost import CatBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from app.scripts.serialisation import make_json_serializable
from app.update_message import UpdateMessage


ver_major = 0
ver_minor = 1
ver_patch = 10

CLASSIFIERS = {cls.__name__: cls for cls in [
    HDBSCAN_Clustering
]}


class BaseClass:

    def __init__(self, data, pipeline_config: dict, logger: logging.Logger, callback: callable = None):

        # Observer
        self._callbacks = set()
        self.subscribe(callback)

        # logger
        self._logger = logger
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                self._log_folder = os.path.dirname(handler.baseFilename)
                break

        self._logger.info('Initialising base class.')

        self._config: dict = pipeline_config or yaml.safe_load(open('model_config.yaml', 'r'))

        # Data
        self._input_ts_df = data
        self._data_ts_df = pd.DataFrame()
        self._feature_dict = {}
        self._features_ft_df = pd.DataFrame()
        self._normalised_features_ft_df = pd.DataFrame()
        self._tag_df = pd.DataFrame()
        self._pipeline_mode = None
        self._cycle_ids = self._input_ts_df['cycle_id'].unique().tolist()
        self._reduction_summary = {}  # Dict to store data to be returned to the calling code

        # Exernal methods
        self._plot_simple_ts = plot_simple_ts
        self.__plot_stacked_ts = plot_stacked_ts
        self.__plot_ts_and_xcorr = plot_ts_and_xcorr
        # self.__df_metadata = df_metadata

        if self._log_folder:
            self.__write_config_to_log(pipeline_config, self._log_folder)

        if self._config['add_tag_to_df']:
            self.__create_tag_to_df()

        if 'cycle_id' in self._input_ts_df.columns:
            self.__initalise_reduction_summary()
        else:
            self._logger.error("Input DataFrame must contain a 'cycle_id' column.")

    def subscribe(self, callback):
        """
        Adds an callback-function for notifications.
        """
        if callable(callback):
            self._callbacks.add(callback)

    def unsubscribe(self, callback):
        """
        Removes an callback-function for notifications.
        """
        self._callbacks.discard(callback)

    def _notify(self, data):
        """
        Calls all registered callback functions with the given message.

        Args:
            data: The data to pass to each callback.
        """
        for callback in self._callbacks:
            try:
                callback(data)
            except Exception:
                self._logger.exception("Error in calling notify-callback")

    def __initalise_reduction_summary(self):
        """
        Initializes a pipeline_result dict from a DataFrame
        containing a 'cycle_id' column.
        """
        reduction_summary = {"cycles_summary": {}}

        for cycle_id in self._input_ts_df['cycle_id'].unique():

            # Try to get the label for this cycle_id, or None if not present
            label = 'None'
            if 'label' in self._input_ts_df.columns:
                label = int(self._input_ts_df[self._input_ts_df['cycle_id'] == cycle_id]['label'].iloc[0])

            reduction_summary["cycles_summary"][int(cycle_id)] = {
                "status": "Valid",      # default status
                "notes": "",              # optional, blank initially
                "label": label
            }

        self._reduction_summary = reduction_summary

    def __update_reduction_summary(self, cycle_ids, updates):
        """
        Update fields for multiple cycles in self._pipeline_result['cycles_summary'].

        Args:
            cycle_ids (iterable): List or set of cycle IDs to update.
            updates (dict): Fields to update for each cycle, e.g. {"status": "completed"}.
        """
        cs = self._reduction_summary.get("cycles_summary", {})

        for cid in cycle_ids:
            cid = int(cid)  # ensure plain int keys
            if cid not in cs:
                # Initialize if it doesn't exist yet
                cs[cid] = {"status": "pending", "notes": ""}

            cs[cid].update(updates)

        # store back (optional if dict is mutable)
        self._reduction_summary["cycles_summary"] = cs

    def __create_tag_to_df(self):
        if 'tag' in self._input_ts_df.columns:
            self._tag_df = self._input_ts_df[['cycle_id', 'tag']].drop_duplicates(subset=['cycle_id'])
        else:
            self._tag_df = self._input_ts_df[['cycle_id']].copy()
            self._tag_df['tag'] = ''

    def _save_fig_to_log(self, fig, filename='plot.png',  dpi=300):
        # Create the full path to save the figure
        file_path = os.path.join(self._log_folder, filename)

        # Save the figure as a PNG file
        fig.savefig(file_path, dpi=dpi)

        plt.close(fig)
        # Optionally, return the file path for confirmation
        return file_path

    def __write_config_to_log(self, config, dest_folder):
        try:
            # Create the destination path
            dest = os.path.join(dest_folder, 'model_config.yaml')

            # Check if destination directory exists, create it if not
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            with open(dest, 'w') as file:
                yaml.dump(config, file)
            self._logger.info("File successfully written to '%s'", dest)

        except FileNotFoundError as e:
            self._logger.warning("Error: %s", e)

        except PermissionError:
            self._logger.warning("Permission Error: You don't have permission to access the file or destination.")

        except Exception as e:
            self._logger.warning("An unexpected error occurred: %s", e)

    def _create_delta_times(self, df):
        """
        Creates a '_delta_time' column representing the time elapsed in seconds
        from the start of each cycle, based on the 'cycle_id'. This function subtracts
        the minimum timestamp for each cycle from each timestamp in the 'timestamp' column
        and returns only the relevant columns.

        The process involves:
        - Copying the input DataFrame.
        - Calculating the minimum 'timestamp' for each 'cycle_id'.
        - Subtracting the minimum timestamp from each timestamp to get the elapsed time in seconds.
        - Selecting the relevant columns ['delta_time', 'value', 'label', 'cycle_id'].

        :param self: Instance of the class. Expects the attribute `input_ts_df` to be a DataFrame
                    containing the 'timestamp' and 'cycle_id' columns.

        :return: None. Creates the class level dataframe `_input_delta_ts_df` in  with the ['delta_time', 'value', 'label', 'cycle_id'] columns.
        :rtype: None
        """

        self._logger.info('START create_delta_times()')
        start = time.perf_counter()

        df = df.copy()  # Detach from the original dataframe

        # If 'delta_time' does not exist, create it from 'timestamp'
        if 'delta_time' not in df.columns:
            df['delta_time'] = df['timestamp'].view('int64') / 1e9
            # df['delta_time'] = df['delta_time'].view('int64') / 1e9

        # Creates a vector of equal length than the entire dataset with the min for each cycle_id
        min_timestamps = df.groupby('cycle_id')['delta_time'].transform('min')

        # Subtracts min_timestamps
        df['delta_time'] = df['delta_time'] - min_timestamps

        desired_columns = ['delta_time', 'value', 'label', 'cycle_id']
        existing_columns = [col for col in desired_columns if col in df.columns]
        df = df[existing_columns]

        end = time.perf_counter()
        self._logger.info('FINISH create_delta_times() in %.2f s\n', end-start)

        return df

    def __compute_fft_lag(self, x, y, cycle_id=None):
        """
        Computes the lag between two signals using FFT-based cross-correlation with sub-sample accuracy.

        The process involves:
        - Computing the cross-correlation between the two input signals using FFT.
        - Identifying the lag corresponding to the maximum correlation.
        - Refining the lag estimate using parabolic interpolation for sub-sample precision.
        - Optionally plotting and saving the cross-correlation and time series for inspection.

        :param self: Instance of the class.
        :param x: First input signal (numpy array).
        :param y: Second input signal (numpy array).
        :param cycle_id: Optional identifier for the current cycle, used for naming plots. If None, defaults to the length of `y`.

        :return: Estimated lag (float, can be fractional) at which the cross-correlation between `x` and `y` is maximized.
        :rtype: float
        """

        # Output length for FFT rounded to the next power of 2
        n = 2 ** int(np.ceil(np.log2(len(x) + len(y) - 1)))

        # Compute the FFT of both signals
        f1 = np.fft.fft(x, n)
        f2 = np.fft.fft(y, n)

        # Compute the cross-correlation using the inverse FFT
        cc = np.real(np.fft.ifft(f1 * np.conj(f2)))

        # Shift the zero lag to the center of the cross-correlation result for better interpretation
        cc = np.fft.fftshift(cc)

        # Generate the lag values
        lags = np.arange(-n // 2, n // 2)

        # First approximation to lag
        peak_idx = np.argmax(cc)
        peak_val = cc[peak_idx]
        lag = lags[peak_idx]

        # Improve lag using parabolic interpolation
        vertex_x, vertex_y, x_fit, y_fit = None, None, None, None
        if 0 < peak_idx < len(cc) - 1:

            # Define a threshold (e.g., 70% of the peak value)
            threshold = 0.99 * peak_val

            # Find the contiguous region around the peak
            # Start from the peak and expand left/right while above threshold
            # Added 10px from peak to avoid edge effects
            left = peak_idx - 10
            while left > 0 and cc[left - 1] >= threshold:
                left -= 1
            right = peak_idx + 10
            while right < len(cc) - 1 and cc[right + 1] >= threshold:
                right += 1

            # Select the lags and cc values for fitting
            x_vals = lags[left:right+1]
            y_vals = cc[left:right+1]

            # Fit a parabola: y = ax^2 + bx + c
            coeffs = np.polyfit(x_vals, y_vals, 2)
            a, b, c = coeffs

            # Vertex of the parabola (sub-sample peak)
            vertex_x = -b / (2 * a)
            vertex_y = a * vertex_x**2 + b * vertex_x + c

            # Generate smooth parabola around the peak
            x_fit = np.linspace(x_vals[0], x_vals[-1], 100)
            y_fit = a * x_fit**2 + b * x_fit + c

            lag = vertex_x

        if self._config['plots']['enabled'] and self._config['plots']['plot_ts_and_xcorr']:
            window = abs(right-left)  # Distance from the peak to the left and right. About 2*window points
            fig = self.__plot_ts_and_xcorr(x, y, cc, lags, lag, vertex_x, vertex_y, x_fit, y_fit, window)
            if cycle_id is None:  # if cycle_id is not provided, use the length of y as cycle_id to aviod (most) repeated plots
                cycle_id = len(y)
            self._save_fig_to_log(fig, f'cross_correlation_{cycle_id}.png')

        return lag

    def __align_signals_df_predict(self, df, lags_dict):
        """
        Align signals using lag in samples and crop to common overlapping range.
        Rebase delta_time per cycle.

        Parameters:
        - df: pandas DataFrame with ['cycle_id', 'delta_time', 'value', 'label']
        - lags_dict: dict {cycle_id: lag} (in sample indices)

        Returns:
        - aligned_df: DataFrame with aligned and rebased signals
        """

        reference_signal_len = len(self._reference_signal)

        # Ensure consistent ordering
        df = df.sort_values(['cycle_id', 'delta_time']).copy()

        cropped_dfs = []
        lengths = df.groupby('cycle_id')['value'].apply(len)
        min_length = lengths.min()
        self._logger.info("Minimum signal length across cycles: %d", min_length)

        for cycle_id, group in df.groupby("cycle_id"):

            lag = lags_dict.get(cycle_id, 0)
            signal = group.reset_index(drop=True)
            values = signal['value'].values
            delta_time = signal['delta_time'].values

            # Compare with reference signal
            if reference_signal_len > len(values):
                self._logger.warning("Cycle %d is shorter than the reference signal. Cycle will be ignored.", cycle_id)
                self._reduction_summary['cycles_summary'][int(cycle_id)]['status'] = 'Ignored'
                self._reduction_summary['cycles_summary'][int(cycle_id)]['notes'] = 'Shorter than reference signal'
                continue

            # Check size after lag
            if reference_signal_len > len(values) - abs(lag):
                self._logger.warning("Cycle %d (AFTER LAG ADJUSTMENT), is shorter than the reference signal. Cycle will be ignored.", cycle_id)
                self._reduction_summary['cycles_summary'][int(cycle_id)]['status'] = 'Ignored'
                self._reduction_summary['cycles_summary'][int(cycle_id)]['notes'] = 'Shorter than reference signal'
                continue

            # Check direction of lag
            if lag > 0:
                self._logger.warning("Cycle %d had a positive lag (%d). Cycle will be ignored.", cycle_id, lag)
                self._reduction_summary['cycles_summary'][int(cycle_id)]['status'] = 'Ignored'
                self._reduction_summary['cycles_summary'][int(cycle_id)]['notes'] = 'Positive Lag'
                continue

            if lag == 0:
                self._logger.debug("Cycle %d: No lag applied. Keeping length: %d.", cycle_id, len(values))

            # Only condition that changes length
            if lag < 0:
                self._logger.debug("Cycle %d: Applying negative lag of %d. Original length: %d. Padding at the back.", cycle_id, lag, len(values))
                values = values[-lag: reference_signal_len - lag]  # Remove the front 'lag' and chop to refernce lenght (double negative)
                delta_time = signal['delta_time'].values[-lag: reference_signal_len - lag]
                delta_time = np.round(delta_time - delta_time[0], 3)  # Rebase. Assumes constant sampling rate.

            data = {
                'cycle_id': cycle_id,
                'delta_time': delta_time,
                'value': values
            }
            if 'label' in signal.columns:
                data['label'] = signal['label'].values[-lag: reference_signal_len - lag]

            cropped_df = pd.DataFrame(data)
            cropped_dfs.append(cropped_df)

        if not cropped_dfs:
            self._logger.error("No cycles remained after alignment. Returning empty DataFrame.")

            cols = ['cycle_id', 'delta_time', 'value']
            if 'label' in df.columns:
                cols.append('label')

            self._notify(UpdateMessage(status=RunState.RUNNING, task=RunTask.PREDICTION, metadata={'step': 1, 'message': 'No cycles remained after alignment.'}))

            return pd.DataFrame(columns=cols)

        aligned_df = pd.concat(cropped_dfs, ignore_index=True)
        return aligned_df

    def __align_signals_df_train(self, df, lags_dict):
        """
        Align signals using lag in samples and crop to common overlapping range.
        Rebase delta_time per cycle.

        Parameters:
        - df: pandas DataFrame with ['cycle_id', 'delta_time', 'value', 'label']
        - lags_dict: dict {cycle_id: lag} (in sample indices)

        Returns:
        - aligned_df: DataFrame with aligned and rebased signals
        """
        # Ensure consistent ordering
        df = df.sort_values(['cycle_id', 'delta_time']).copy()

        # Get max_lag and lengths from df
        lags_list = list(lags_dict.values())
        max_lag = max(lags_list)
        lengths = df.groupby('cycle_id')['value'].apply(len)

        # Offset to rightmost signal
        left_trim_indices = np.abs(np.array(lags_list) - max_lag)

        # Compute right index to match min common length
        remaining = lengths - left_trim_indices
        common_length = remaining.min()
        right_trim_indices = remaining - common_length

        #####################
        # Pack up back into df
        total_len = lengths.sum()
        # cumulative offsets = start index of each cycle in the global mask
        offsets = np.cumsum(np.r_[0, lengths[:-1]])
        # start and stop positions
        starts = offsets + left_trim_indices
        stops = offsets + lengths - right_trim_indices

        # generate indices for all True positions
        indices_true = np.concatenate([np.arange(s, e) for s, e in zip(starts, stops)])

        lag_mask = np.zeros(total_len, dtype=bool)
        lag_mask[indices_true] = True

        aligned_df = df[lag_mask].reset_index(drop=True)

        # Reset delta times
        aligned_df = self._create_delta_times(aligned_df)

        return aligned_df

    def _data_preprocessing(self, reference_signal=None):
        """
        Aligns and preprocesses time series data by correcting time shifts across cycles.

        This method performs the following steps:
        - Optionally plots stacked time series before alignment.
        - Sets the reference signal (either provided or defaults to the first cycle).
        - Calculates cross-correlation lags for all cycles relative to the reference.
        - Aligns each cycle's signal based on its lag, padding as needed.
        - Optionally plots stacked time series after alignment.
        - Crops all cycles to match the reference signal length.
        - Removes leading/trailing zeros across all cycles for consistent alignment.
        - If in 'Train' mode, computes the median signal for label==1 cycles as the new reference.

        :param reference_signal: Optional numpy array to use as the reference for alignment.
        :type reference_signal: numpy.ndarray, optional

        :return: None. Modifies `self._data_ts_df` in place with aligned and cropped signals.
        """
        self._logger.info('START _data_preprocessing()')
        start = time.perf_counter()

        # Plot
        if self._config['plots']['enabled'] and self._config['plots']['plot_stacked_ts_pre_lags']:
            fig = self.__plot_stacked_ts(self._data_ts_df, reference_signal=reference_signal)
            self._save_fig_to_log(fig, 'stacked_ts_pre_lag.png')

        data_quality = self._config['data_quality']

        # Collect df data
        df = self._data_ts_df

        # Outliler filtering
        if data_quality.get('filter_outliers', False):
            self._remove_outliers(df)

        # Data quality checks
        self._data_quality_checks(df)

        # Set the reference signal: use the provided one or default to the first signal
        if reference_signal is None:
            self._logger.info('No reference signal found in model, using to first signal')
            # Use the 'value' column from the first cycle in the DataFrame as the reference signal
            self._reference_signal = self._data_ts_df[self._data_ts_df['cycle_id'] == min(self._cycle_ids)]['value'].values

        else:
            self._logger.info('Found reference signal in model.')
            self._reference_signal = reference_signal

        # Calculate cross-correlation lags for all signals
        lag_dict = self.__calculate_lags_from_df()
        self._logger.info("Lags dict: %s", lag_dict)

        # Lag based checks
        self._data_lags_checks(lag_dict)

        if self._pipeline_mode == 'Train':
            self._data_ts_df = self.__align_signals_df_train(self._data_ts_df, lag_dict)

        elif self._pipeline_mode == 'Predict':
            # Align each cycle's signal based on its lag
            self._data_ts_df = self.__align_signals_df_predict(self._data_ts_df, lag_dict)

        # Plot
        if self._config['plots']['enabled'] and self._config['plots']['plot_stacked_ts_post_lags']:
            fig = self.__plot_stacked_ts(self._data_ts_df, reference_signal=reference_signal)
            self._save_fig_to_log(fig, 'stacked_ts_post_lag.png')

        end = time.perf_counter()
        self._logger.info("FINISH _data_preprocessing() in %.2f s\n", end-start)

    def __calculate_lags_from_df(self):
        """
        Calculate the lag for each signal relative to a reference signal using cross-correlation.

        This function takes a list of signals and calculates the lag (shift) required to align
        each signal with a reference signal. The reference signal is decided earlier and available in self._reference_signal.
        The lag is calculated using the cross-correlation of the reference signal and each
        subsequent signal.

        Parameters:
        signals (list of numpy arrays): A list of numpy arrays, where each array represents a signal.

        Returns:
        numpy.ndarray: An array of lag values.
        """
        # lag_array = []
        lag_dict = {}
        # Iterate over each signal in the list
        for cycle_id, group in self._data_ts_df.groupby('cycle_id'):
            signal = group['value'].values

            # Compute lag using cross-correlation with reference
            lag = self.__compute_fft_lag(self._reference_signal, signal, cycle_id=cycle_id)
            lag_dict[cycle_id] = round(lag)
            # lag_array.append(lag)

        return lag_dict

    def _feature_extraction(self, features_to_extract=None):
        """
        Extracts features from the time series data for each cycle. The function supports extracting all features
        or a subset based on the input dictionary `features_to_extract`. For each cycle, it computes features defined
        in `features_dict` or in the provided subset and stores the results in a dataframe.

        The function processes the data in cycles, extracting specified features and appending them to a result dataframe.

        :param features_to_extract: A dictionary specifying which features to extract, with feature names as keys
                                    and parameters for the functions as values. If None, all features from `features_dict`
                                    are extracted. Defaults to None.
        :type features_to_extract: dict, optional

        :return: None. The extracted features are stored in `self.features_ft_df` as a dataframe.
        :rtype: None
        """
        self._logger.info('START feature_extraction()')

        # Use all features if no specific subset is provided
        if features_to_extract is None:

            self._logger.info('Using the complete features list.')
            features = features_dict
            skip_metadata = False  # Retain metadata when extracting all features

        else:

            self._logger.info('Using the model specific features list.')

            # Initialize the feature dictionary for a subset
            features = {}
            skip_metadata = True  # Skip metadata for a specific subset of features
            for feature_column, feature_value in features_to_extract.items():
                for key, value in feature_value['function_call'].items():
                    if key in features:
                        # Append value if the key already exists
                        features[key].append(value)
                    else:
                        # Initialize the key with the value as a list
                        features[key] = [value] if value is not None else None

        self._logger.info("Features (%d): %s", len(features), list(features.keys()))

        # Initialize an empty dataframe to store extracted features
        self._features_ft_df = pd.DataFrame()
        logged_labels = {}  # used in the one_cyle_per_label option
        logged_labels_per_feature = {}

        # Process each cycle in the dataset
        for cycle_idx, (cycle_id, cycle_ts_df) in enumerate(self._data_ts_df.groupby('cycle_id')):

            # Initialize a dictionary to store the results for this cycle, starting with the cycle_id
            cycle_results = {'cycle_id': cycle_id}

            if self._config['debug_features']:
                self._logger.debug('Extracting features from cycle_id: %s', cycle_id)

            # Extract features for the current cycle based on the specified features
            for feature_name, param in features.items():

                experimental_features = self._config.get('experimental', {})
                # Skip feature if it's marked as experimental and disabled in config
                if (feature_name in experimental_features and not experimental_features.get(feature_name, True)) or \
                   (experimental_features.get('enabled') is False and feature_name in experimental_features):
                    self._logger.info("Skipping experimental feature '%s' as it is disabled in the config.", feature_name)
                    continue

                if feature_name not in logged_labels_per_feature:
                    logged_labels_per_feature[feature_name] = {0: False, 1: False}

                # Retrieve the feature extraction function dynamically
                feature_function = getattr(fef, feature_name, None)

                if feature_function is None:
                    self._logger.info("Warning: Feature function '%s' not found in 'fef'. Skipping.", feature_name)
                    continue  # Skip if the function does not exist

                # Call the feature function with the corresponding parameters
                try:

                    # Create a logger dict if feature debugging is intended.
                    _logging_dict = None
                    debug_features = self._config['debug_features']
                    if debug_features['enabled'] and feature_name in debug_features:

                        # Skip if feature_name is None
                        if not debug_features[feature_name]:
                            _logging_dict = None

                        else:

                            _logging_dict = {}
                            _logging_dict['logger'] = self._logger
                            _logging_dict['self_instance'] = self
                            _logging_dict['cycle_id'] = cycle_id
                            if 'label' in cycle_ts_df.columns:
                                label = _logging_dict['label'] = int(cycle_ts_df["label"].iloc[0])

                            # Logging is done unless prevented by any of the conditions below
                            if debug_features['one_cycle_per_label'] and self._pipeline_mode == 'Train':         # Logging condition: one_cycle_per_label (only for training)
                                if label == 0:
                                    if logged_labels_per_feature[feature_name].get(0, False):
                                        _logging_dict = None
                                    else:
                                        logged_labels_per_feature[feature_name][0] = True

                                if label == 1:
                                    if logged_labels_per_feature[feature_name].get(1, False):
                                        _logging_dict = None
                                    else:
                                        logged_labels_per_feature[feature_name][1] = True

                            elif debug_features['first_cycle_only'] and cycle_idx > 0:  # avoid logging for all cycles except the first
                                _logging_dict = None

                    # Call the feature extraction function
                    feature_return = feature_function(cycle_ts_df['value'].values, param, _logging_dict=_logging_dict)

                except Exception:
                    self._logger.exception("Error while extracting feature '%s' for cycle %d", feature_name, cycle_id)
                    continue  # Skip this feature extraction if an error occurs

                # Unpack and process the results if the return value is zipped or a list
                try:
                    if isinstance(feature_return, list) and (all(isinstance(f, np.float64) for f in feature_return) or all(isinstance(f, np.float32) for f in feature_return)):
                        feature_return = np.array(feature_return)
                    elif isinstance(feature_return, (zip, list)):
                        _, feature_return = zip(*feature_return)
                except Exception:
                    pass

                # Store the extracted feature results in the cycle's dictionary
                cycle_results = self.__unpack_and_write_results(
                    feature_name, param, feature_return, cycle_results, cycle_idx, skip_metadata=skip_metadata)

            # Append the cycle results to the main dataframe
            df_the_dict = pd.DataFrame(cycle_results, index=[0])
            self._features_ft_df = pd.concat([self._features_ft_df, df_the_dict], ignore_index=True)

        # Reorganize columns and convert numerical features to float32
        if features_to_extract is None:
            # If extracting all features, convert all features (except 'cycle_id') to float32
            self._features_ft_df[self._features_ft_df.columns.drop('cycle_id')] = \
                self._features_ft_df[self._features_ft_df.columns.drop('cycle_id')].astype('float32')
        else:
            # For a subset of features, keep only those specified and convert to float32
            self._features_ft_df = pd.concat([
                self._features_ft_df[['cycle_id']],
                self._features_ft_df[list(features_to_extract.keys())].astype('float32')
            ], axis=1)

        self._logger.info("%d features extracted from %d cycles", self._features_ft_df.shape[1] - 1, self._features_ft_df.shape[0])
        self._logger.info('FINISH feature_extraction()\n')

    def __unpack_and_write_results(self, feature_name, params, feature_return, cycle_results, cycle_idx, skip_metadata):
        """
        Unpacks the feature return values and stores them in the `cycle_results` dictionary under appropriately named columns.
        Handles the creation of feature column names based on the parameters and the type of feature return value.

        The method also updates the metadata (`self.feature_dict`) during the first cycle if `skip_metadata` is False.

        :param feature_name: The name of the feature to be extracted.
        :type feature_name: str

        :param params: Parameters used to extract the feature, can be a list of values or a single value.
        :type params: list or None

        :param feature_return: The value(s) returned by the feature extraction function.
        :type feature_return: any (can be a single value or a list/tuple of values)

        :param cycle_results: The dictionary where the feature extraction results for the current cycle are stored.
        :type cycle_results: dict

        :param cycle_idx: The index of the current cycle, used to manage metadata updates.
        :type cycle_idx: int

        :param skip_metadata: Flag to skip metadata updates (useful for subsets of features).
        :type skip_metadata: bool

        :return: Updated `cycle_results` with the newly added feature data.
        :rtype: dict
        """

        # Handle the case where params is None (no parameters to format for feature name)
        if params is None:

            col_name = f"{feature_name}"  # Generate a simple name
            cycle_results[col_name] = feature_return

            # Update metadata only for the first cycle if not skipping
            if cycle_idx == 0 and not skip_metadata:
                self._feature_dict[col_name] = {'function_call': {feature_name: None}, 'mean': None, 'std': None}

        # Handle the case where params is a list
        elif isinstance(params, list):
            for idx, param in enumerate(params):
                if isinstance(param, dict):

                    # If param is a dictionary, generate a descriptive column name
                    col_name = feature_name + ''.join([f"__{key}_{value}" for key, value in sorted(param.items())])

                else:
                    # If param is a single value, append it to the feature name
                    col_name = f"{feature_name}__{param}"

                # Final cleanup
                col_name = re.sub(r'[()\[\]]', '', col_name)  # Remove brackets
                col_name = col_name.replace(', ', '_')     # Replace ', ' with '_'

                if isinstance(feature_return, (dict)):                    # Pair the column name with the corresponding feature_return value
                    cycle_results[col_name] = feature_return[param]

                else:
                    # Pair the column name with the corresponding feature_return value
                    cycle_results[col_name] = feature_return[idx]

                # only for the first cycle
                if cycle_idx == 0 and not skip_metadata:
                    self._feature_dict[col_name] = {'function_call': {feature_name: param}, 'mean': None, 'std': None}

        else:
            # If params is neither None nor a list, handle as an unexpected case
            self._logger.warning("Warning: Unexpected type for params in feature '%s': %s", feature_name, type(params))

        return cycle_results

    def _normalise_features(self):
        """
        Normalizes the features in the DataFrame using Standard Scaling (z-score normalization).
        This method standardizes the features by removing the mean and scaling to unit variance,
        which can improve the performance of machine learning models.

        The method performs the following steps:
        1. Selects the training data based on cycle IDs.
        2. Fits the scaler on the training data.
        3. Transforms all features (including the validation set) using the fitted scaler.
        4. Updates the feature dictionary with the mean and standard deviation of each feature.

        :return: None. Updates the `normalised_features_ft_df` DataFrame with normalized feature values.
        """

        self._logger.info('START normalise_features()')

        # Create a copy of the original DataFrame for normalized features
        self._normalised_features_ft_df = self._features_ft_df.copy()

        # Prepare scaling variable
        scaling_type = 'Standard'
        if 'scaling_type' in self._config:
            scaling_type = self._config['scaling_type']

        # Initialize the StandardScaler
        if scaling_type == 'Standard':
            self._logger.info('Using StandardScaler()')
            scaler = StandardScaler()
        else:
            self._logger.info('Using MinMaxScaler()')
            scaler = MinMaxScaler()

        # Mask to select the rows corresponding to training cycle IDs
        mask = self._features_ft_df['cycle_id'].isin(self._training_cycle_ids)

        # Check if there is at least one training cycle ID
        if not mask.any():
            self._logger.info("Warning: No training cycle IDs found in the features DataFrame.")
            return

        # Fit the scaler only on the training data
        try:
            scaler.fit(self._features_ft_df[mask].drop(columns='cycle_id'))
        except Exception:
            self._logger.exception("Error fitting scaler")
            return

        # Extract features used for scaling - use float64 for json-compatibility
        if scaling_type == 'Standard':
            features_mean = scaler.mean_.astype(np.float64)        # Mean of all features for scaling
            features_std = scaler.scale_.astype(np.float64)        # Standard deviation of all features for scaling
        else:
            features_min = scaler.min_.astype(np.float64)
            features_scale = scaler.scale_.astype(np.float64)
            features_data_min = scaler.data_min_.astype(np.float64)
            features_data_max = scaler.data_max_.astype(np.float64)
            features_data_range = scaler.data_range_.astype(np.float64)

        # Transform all features (including validation) based on the fitted scaler
        try:
            normalised_features = scaler.transform(self._features_ft_df.drop(columns='cycle_id').values)
        except Exception:
            self._logger.exception("Error transforming features")
            return

        self._normalised_features_ft_df.iloc[:, 1:] = normalised_features  # Update the feature columns with normalized values

        # Write values to the feature dictionary for each feature
        if scaling_type == 'Standard':
            for col, mean, std in zip(self._normalised_features_ft_df.columns[1:], features_mean, features_std):
                if col in self._feature_dict:
                    self._feature_dict[col]['mean'] = mean
                    self._feature_dict[col]['std'] = std
                else:
                    self._logger.warning("Warning: %s not found in feature_dict. Skipping update.", col)
        else:
            for col, col_min, col_scale, col_data_min, col_data_max, col_data_range in zip(self._normalised_features_ft_df.columns[1:], features_min, features_scale, features_data_min, features_data_max, features_data_range):
                if col in self._feature_dict:
                    self._feature_dict[col]['min'] = col_min
                    self._feature_dict[col]['scale'] = col_scale
                    self._feature_dict[col]['data_min'] = col_data_min
                    self._feature_dict[col]['data_max'] = col_data_max
                    self._feature_dict[col]['data_range'] = col_data_range

                else:
                    self._logger.warning("Warning: %s not found in feature_dict. Skipping update.", col)

        self._logger.info('FINISH normalise_features()\n')

    def _find_stationary_sections(
        self,
        df,
        window_mode="length",              # "fixed", "length", "stability", or "hybrid"
        fixed_window=10,
        window_fraction=0.01,
        threshold_quantile=0.8,
        threshold_multiplier=1.5,
        std_tolerance=0.001,
        vote_fraction=0.5,                 # ignored in median mode
        min_section_length=20,
        apply_smoothing=True,
        smoothing_kernel_size=5
    ):
        """
        Identifies stationary and non-stationary sections based on the median signal across all cycles.
        """

        def find_stable_window(values, min_window=5, max_window=100, std_tolerance=0.001):
            prev_std = None
            for w in range(min_window, max_window):
                rolling_std = pd.Series(values).rolling(window=w, center=True).std()
                mean_std = rolling_std.mean()
                if prev_std is not None and abs(mean_std - prev_std) < std_tolerance:
                    return w
                prev_std = mean_std
            return max_window

        # Get aligned matrix of all cycles
        grouped = df.groupby('cycle_id')['value']
        values_matrix = np.stack(grouped.apply(np.array).values)  # shape: (n_cycles, signal_len)
        median_signal = np.median(values_matrix, axis=0)

        signal_len = median_signal.shape[0]

        # Dynamically choose window
        if window_mode == "fixed":
            window = fixed_window
        elif window_mode == "length":
            window = max(5, int(signal_len * window_fraction))
        elif window_mode == "stability":
            window = find_stable_window(median_signal, std_tolerance=std_tolerance)
        elif window_mode == "hybrid":
            base_window = max(5, int(signal_len * window_fraction))
            window = find_stable_window(median_signal, min_window=base_window,
                                        max_window=base_window * 2,
                                        std_tolerance=std_tolerance)
        else:
            raise ValueError(f"Unsupported window_mode: {window_mode}")

        # Compute rolling std of median signal
        rolling_std = pd.Series(median_signal).rolling(window=window, center=True).std().values

        # Threshold based on quantile and scaling
        std_values = rolling_std[~np.isnan(rolling_std)]
        std_threshold = np.quantile(std_values, threshold_quantile) * threshold_multiplier

        # Determine where signal is stationary
        stationary_mask = (rolling_std < std_threshold)

        # Optional smoothing (merge close gaps)
        if apply_smoothing:
            stationary_mask = binary_closing(stationary_mask, structure=np.ones(smoothing_kernel_size))

        # Helper to extract contiguous regions
        def get_sections(mask, min_len):
            sections = []
            start = None
            for i, val in enumerate(mask):
                if val and start is None:
                    start = i
                elif not val and start is not None:
                    if i - start >= min_len:
                        sections.append((start, i - 1))
                    start = None
            if start is not None and len(mask) - start >= min_len:
                sections.append((start, len(mask) - 1))
            return sections

        stationary_sections = get_sections(stationary_mask, min_section_length)
        non_stationary_sections = get_sections(~stationary_mask, min_section_length)

        result = {}
        for idx, (start, end) in enumerate(stationary_sections):
            result[f's{idx}'] = (start, end)
        for idx, (start, end) in enumerate(non_stationary_sections):
            result[f'ns{idx}'] = (start, end)

        # Forcing the result for tests
        # result = {'s0': (7, 34), 's1': (264, 1429), 'ns0': (35,  249)}

        return result, median_signal

    def _data_quality_checks(self, df):

        data_quality = self._config['data_quality']

        lengths = df.groupby('cycle_id')['value'].apply(len)
        min_length = lengths.min()
        max_length = lengths.max()
        median_length = lengths.median()
        num_cycles = len(lengths)
        self._logger.info("Number of cycles: %d", num_cycles)
        self._logger.info("Min length: %d, Median length: %d, Max length: %d", min_length, median_length, max_length)

        min_length_ratio = data_quality['min_length_ratio']

        if min_length / median_length < min_length_ratio:
            self._logger.warning(
                f"Minimum signal length ({min_length}) is less than {min_length_ratio*100:.0f}% of the median signal length ({median_length})."
            )
            if data_quality['min_length_cancel']:
                pass  # TODO Cancel reduction code here

            if data_quality['min_length_remove']:
                # Remove cycles with too short signals
                removed_cycles = [cid for cid in lengths[lengths < min_length_ratio * median_length].index]
                self.__update_reduction_summary(removed_cycles, {"status": "Removed", "notes": f"Removed due to short length (<{min_length_ratio*100:.0f}% of median)."})
                self._logger.info("Removing cycles with too short signals: %s", removed_cycles)
                self._data_ts_df = df[df['cycle_id'].isin(lengths[lengths >= min_length_ratio * median_length].index)]

    def _data_lags_checks(self, lag_dict):

        data_quality = self._config['data_quality']

        reference_signal_len = len(self._reference_signal)
        self._logger.info("Reference signal length: %d.", reference_signal_len)

        # Compute the median and MAD of the lags
        lags = list(lag_dict.values())
        median_lag = np.median(lags)
        mad = np.median([abs(lag - median_lag) for lag in lags])

        # Avoid division by zero
        if mad == 0:
            mad = 1e-8

        # Coefficient to scale the outlier threshold
        mad_z_thresh = data_quality['max_lag_mad_z']

        # Identify outliers using modified z-score
        cycles_with_long_lags = [
            cycle_id for cycle_id, lag in lag_dict.items()
            if abs(0.6745 * (lag - median_lag) / mad) > mad_z_thresh
        ]

        self._logger.info("Cycles with modified z-score > %d: %s", mad_z_thresh, cycles_with_long_lags)

        # Remove if requested
        if data_quality['max_lag_remove'] and cycles_with_long_lags:
            self._logger.warning("Removing cycles with long lags (MAD method).")
            self.__update_reduction_summary(cycles_with_long_lags, {"status": "Removed", "notes": "Removed due to long lag."})

            self._data_ts_df = self._data_ts_df[~self._data_ts_df['cycle_id'].isin(cycles_with_long_lags)]
            lag_dict = {cycle_id: lag for cycle_id, lag in lag_dict.items() if cycle_id not in cycles_with_long_lags}

            # Log the lags
            self._logger.info("Lags to apply after max lag removal: %s.", lag_dict)

    def _remove_outliers(self, df: pd.DataFrame):

        tol = 1e-2  # tolerance to consider a value as zero
        zeros_coeff = 0.05  # coefficient to determine the number of allowed preciding zeros

        # Calc auto-outliers based on values mean
        if df.size > 0:
            df['outlier'] = False  # Default to False
            if 'label' in df.columns:
                groups = df.groupby('label')
            else:
                groups = [('all', df)]

            for label, df_sub in groups:
                grouped = df_sub.groupby('cycle_id').agg(
                    mean_value=('value', 'mean'),
                    size=('value', 'size'),
                    duration_sec=('delta_time', 'max'),  # Use max delta_time for duration
                    # prec_zeros=('value', lambda x: (x.abs() <= tol).cumprod().sum())
                    prec_zeros=('value', lambda x: (x < 0).cumprod().sum())
                )
                # grouped['duration_sec'] = grouped['duration'] / pd.Timedelta(seconds=1)

                # Calc auto-outliers based on mean of values
                P5_mean, P95_mean = np.percentile(grouped['mean_value'], [5, 95])
                # Calc auto-outliers based on size of values
                P1_size, P99_size = np.percentile(grouped['size'], [1, 99])
                # Calc auto-outliers based on cycle duration
                P5_time, P95_time = np.percentile(grouped['duration_sec'], [1, 99])

                def is_outlier(x):
                    mean_outlier = x['mean_value'] < P5_mean or x['mean_value'] > P95_mean
                    size_outlier = x['size'] < P1_size or x['size'] > P99_size
                    time_outlier = x['duration_sec'] < P5_time or x['duration_sec'] > P95_time
                    zero_prefix_outlier = x['prec_zeros'] >= zeros_coeff * x['size']

                    return mean_outlier or size_outlier or time_outlier or zero_prefix_outlier

                outlier_series = grouped.apply(is_outlier, axis=1)
                outlier_cycle_ids = outlier_series[outlier_series].index

                self.__update_reduction_summary(outlier_cycle_ids, {"status": "Removed", "notes": "Removed as outlier."})
                self._logger.info("Removing cycles with too short signals: %s", outlier_cycle_ids)
                self._data_ts_df = df[~df['cycle_id'].isin(outlier_cycle_ids)]

    @staticmethod
    def mask_outliers(df: pd.DataFrame):
        zeros_coeff = 0.05  # coefficient to determine the number of allowed preciding zeros

        # Calc auto-outliers based on values mean
        if df.size > 0:
            df['outlier'] = False  # Default to False
            if 'label' in df.columns:
                groups = df.groupby('label')
            else:
                groups = [('all', df)]

            for label, df_sub in groups:
                grouped = df_sub.groupby('cycle_id').agg(
                    mean_value=('value', 'mean'),
                    size=('value', 'size'),
                    duration=('timestamp', lambda x: x.max() - x.min()),  # Use max delta_time for duration
                    # prec_zeros=('value', lambda x: (x.abs() <= tol).cumprod().sum())
                    prec_zeros=('value', lambda x: (x < 0).cumprod().sum())
                )
                grouped['duration_sec'] = grouped['duration'] / pd.Timedelta(seconds=1)

                # Calc auto-outliers based on mean of values
                P5_mean, P95_mean = np.percentile(grouped['mean_value'], [5, 95])
                # Calc auto-outliers based on size of values
                P1_size, P99_size = np.percentile(grouped['size'], [1, 99])
                # Calc auto-outliers based on cycle duration
                P5_time, P95_time = np.percentile(grouped['duration_sec'], [1, 99])

                def is_outlier(x):
                    mean_outlier = x['mean_value'] < P5_mean or x['mean_value'] > P95_mean
                    size_outlier = x['size'] < P1_size or x['size'] > P99_size
                    time_outlier = x['duration_sec'] < P5_time or x['duration_sec'] > P95_time
                    zero_prefix_outlier = x['prec_zeros'] >= zeros_coeff * x['size']

                    return mean_outlier or size_outlier or time_outlier or zero_prefix_outlier

                outlier_series = grouped.apply(is_outlier, axis=1)
                outlier_cycle_ids = outlier_series[outlier_series].index
                mask = (
                    (df['label'] == label) if 'label' in df.columns
                    else pd.Series([True] * len(df), index=df.index)
                )
                df.loc[mask, 'outlier'] = df_sub['cycle_id'].isin(outlier_cycle_ids)


class Predictor(BaseClass):

    def __init__(self, data, logger: logging.Logger, models: list[models.Model], pipeline_config: dict = None, callback: callable = None):

        super().__init__(data, pipeline_config, logger, callback)

        self._logger.info("Ini the beninging. Predictor class v%d.%d.%d initialising.", ver_major, ver_minor, ver_patch)

        self._db_models = models
        # self._cycle_ids = self._input_ts_df['cycle_id'].unique().tolist()

        self._pipeline_mode = 'Predict'
        self._logger.info('Predictor class initialised.')

    def predict(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predicts the output using the models stored in the class.
        This method checks if there are any models stored in the class. If no models are found,
        it logs an error message and raises a ValueError. If models are found, it runs the models
        and logs the resulting DataFrame containing cycle IDs and confidence values.

        Returns:
            pd.DataFrame: A DataFrame with cycle- and model-id, and the corresponding metrics per cycle.
            pd.DataFrame: A DataFrame containing the normalized features.

        Raises:
            ValueError: If no models are found in the class.
        """

        if len(self._db_models) == 0:
            self._logger.error('No models found in class.')
            raise ValueError('No models found in class.')
        else:
            cycle_ids_and_conf1_df, X_valid = self.__run_models()

            pd.set_option('display.max_columns', None)
            self._logger.info("Conf_1 values:\n%s", cycle_ids_and_conf1_df.to_string())
            # Log the average confidence value for the first model
            if cycle_ids_and_conf1_df.shape[1] > 1:
                avg_conf1 = cycle_ids_and_conf1_df.iloc[:, 1].mean()
                self._logger.info("Average conf_1: %.4f", avg_conf1)
            return cycle_ids_and_conf1_df, X_valid

    def __normalise_features_from_metadata(self, scaling_type='Standard'):
        """
        Normalizes the features in the feature dataframe using the specified scaling type.

        Args:
            scaling_type (str): The type of scaler to use for normalization.
                        Options are 'Standard' for StandardScaler and 'MinMax' for MinMaxScaler.
                        Default is 'Standard'.

        Returns:
            None

        This method retrieves the mean and standard deviation (for StandardScaler) or the min and max values
        (for MinMaxScaler) from the feature dictionary and applies the corresponding scaler to normalize
        the features in the feature dataframe. The normalized features are stored in the
        __normalised_features_ft_df attribute.

        """

        self._logger.info('START normalise_features_from_metadata()')

        # Retrieve relevant arrays
        if scaling_type == 'Standard':
            self._logger.info('Using StandardScaler() specified in model')

            # Collect the mean and std into numpy arrays in the right order
            scaler = StandardScaler()
            scaler.mean_ = np.array([self._feature_dict[col]['mean'] for col in self._features_ft_df.columns[1:]])
            scaler.scale_ = np.array([self._feature_dict[col]['std'] for col in self._features_ft_df.columns[1:]])

        else:
            self._logger.info('Using MinMaxScaler() specified in model')

            # Collect the min and max into numpy arrays in the right order
            scaler = MinMaxScaler()
            scaler.min_ = np.array([self._feature_dict[col]['min'] for col in self._features_ft_df.columns[1:]])
            scaler.scale_ = np.array([self._feature_dict[col]['scale'] for col in self._features_ft_df.columns[1:]])
            scaler.data_min_ = np.array([self._feature_dict[col]['data_min'] for col in self._features_ft_df.columns[1:]])
            scaler.data_max_ = np.array([self._feature_dict[col]['data_max'] for col in self._features_ft_df.columns[1:]])
            scaler.data_range_ = np.array([self._feature_dict[col]['data_range'] for col in self._features_ft_df.columns[1:]])

        # Apply the scaler to normalize the features
        self.__normalised_features_ft_df = self._features_ft_df.copy()
        self.__normalised_features_ft_df.iloc[:, 1:] = scaler.transform(self._features_ft_df.iloc[:, 1:])

        self._logger.info('FINISH normalise_features_from_metadata()\n')

    def __run_models(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the prediction models on the provided data and compiles the results into a DataFrame.

        This method performs the following tasks:
        1. Initializes required variables and creates delta times.
        2. Iterates through each model in the database, performs preprocessing, and executes predictions.
        3. Optionally plots time series data before and after time lag adjustments.
        4. Extracts features, normalizes them, and applies the appropriate scaling method.
        5. Makes predictions using an ONNX model and stores the results in a DataFrame.
        6. Optionally visualizes model predictions and saves them to log.
        7. Merges the individual model results into a global DataFrame.
        8. Optionally tags the resulting DataFrame with additional cycle information.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame with cycle- and model-id, and the corresponding metrics per cycle.
            pd.DataFrame: A DataFrame containing the normalized features.

        Raises:
            LoggingWarning: If 'scaling_type' is missing from model metadata, a warning is logged and default scaling is applied.
            ValueError: If any unexpected errors occur during the model prediction or data processing.

        Side Effects:
            - Logs the progress of the model running process.
            - Saves plotted figures of time series data to log if configured.
            - Sends update notifications during the process.
        """

        self._logger.info('START run_models()')

        cycle_ids_and_conf1_df = pd.DataFrame()
        self._input_delta_ts_df = self._create_delta_times(self._input_ts_df)

        self._notify(UpdateMessage(status=RunState.RUNNING, task=RunTask.PREDICTION, metadata={'step': 1}))
        for db_model in self._db_models:
            self._logger.info(db_model.model_name)

            # Reset to input signal
            self._data_ts_df = self._input_delta_ts_df.copy()

            reference_signal = None
            if 'reference_signal' in db_model.model_metadata:
                reference_signal = np.array(db_model.model_metadata['reference_signal'])

            if self._config['plots']['enabled'] and self._config['plots']['plot_ts_simple_pre_lags']:
                fig = self._plot_simple_ts(self._data_ts_df)
                self._save_fig_to_log(fig, f'{db_model.model_name}_ts_pre_lag.png')

            # removed to test new version
            self._data_preprocessing(reference_signal)

            if self._data_ts_df.empty:
                self._logger.warning('No data left after preprocessing for model %s. Skipping to next model.', db_model.model_name)
                continue

            if self._config['plots']['enabled'] and self._config['plots']['plot_ts_simple_post_lags']:
                fig = self._plot_simple_ts(self._data_ts_df)
                self._save_fig_to_log(fig, f'{db_model.model_name}_ts_post_lag.png')

            self._feature_dict = db_model.model_metadata['features']
            self._feature_extraction(features_to_extract=self._feature_dict)

            if 'scaling_type' in db_model.model_metadata:
                self.__normalise_features_from_metadata(scaling_type=db_model.model_metadata['scaling_type'])
            else:
                self._logger.warning('scaling_type not found in model metadata. Using scaling_type from config file on test data. This will affect results.')
                self._normalise_features()

            X_valid = self.__normalised_features_ft_df.iloc[:, 1:]
            X_valid = X_valid[sorted(X_valid.columns)]

            if 'HDBSCAN_Clustering' in db_model.model_name:
                # Validate the model contains all required metadata
                required_metadata_keys = ['params', 'medoids', 'std_devs', 'spread_factor_ax', 'spread_factor_perp']
                for key in required_metadata_keys:
                    if key not in db_model.model_metadata:
                        self._logger.error("Missing required metadata '%s' for HDBSCAN_Clustering model '%s'.", key, db_model.model_name)
                        raise ValueError(f"Selected model is no longer compatible. The required metadata '{key}' is not available. Please retrain a new model.")

                clustering = HDBSCAN_Clustering(**db_model.model_metadata['params'])
                clustering.medoids = {int(k): np.array(v) for k, v in db_model.model_metadata['medoids'].items()}
                clustering.std_devs = np.array(db_model.model_metadata['std_devs'], dtype='float')
                clustering.spread_factor_ax = db_model.model_metadata['spread_factor_ax']
                clustering.spread_factor_perp = db_model.model_metadata['spread_factor_perp']
                labels, confidences = clustering.compute_labels_conf1(X_valid)
                cycle_id_and_conf1 = np.column_stack((self.__normalised_features_ft_df['cycle_id'], confidences[1]))
                this_model_cycle_ids_and_conf1_df = pd.DataFrame(cycle_id_and_conf1, columns=['cycle_id', PredictionMetric.CONFIDENCE_1.value])
            else:
                this_model_cycle_ids_and_conf1_df = onnx_predict(db_model.model_onnx, X_valid, self._cycle_ids)

            if self._config['plots']['enabled'] and self._config['plots']['plot_model_2D_predict']:
                y_validation = np.round(this_model_cycle_ids_and_conf1_df[PredictionMetric.CONFIDENCE_1.value])
                plot_model_2D(db_model=db_model,
                              targetFolder=self._log_folder,
                              X_validation=X_valid,
                              y_validation=y_validation,
                              quality=self._config['plots']['plot_model_2D_predict_quality'])

            # If enabled and 3d
            if self._config['plots']['enabled'] and self._config['plots']['plot_model_3D_predict']:
                y_validation = np.round(this_model_cycle_ids_and_conf1_df[PredictionMetric.CONFIDENCE_1.value])
                plot_model_3D(db_model=db_model,
                              targetFolder=self._log_folder,
                              X_validation=X_valid,
                              y_validation=y_validation,
                              quality=self._config['plots']['plot_model_3D_predict_quality'])

            # Add model_id to the dataframe
            this_model_cycle_ids_and_conf1_df['model_id'] = db_model.id
            # Add the overall label/classification to each cycle
            this_model_cycle_ids_and_conf1_df[PredictionMetric.LABEL.value] = (this_model_cycle_ids_and_conf1_df[PredictionMetric.CONFIDENCE_1.value] >= 0.5).astype(int)

            if cycle_ids_and_conf1_df.empty:
                cycle_ids_and_conf1_df = this_model_cycle_ids_and_conf1_df
            else:
                # append this models data to the global results dataframe
                cycle_ids_and_conf1_df = pd.concat([cycle_ids_and_conf1_df, this_model_cycle_ids_and_conf1_df], ignore_index=True)

        self._notify(UpdateMessage(status=RunState.RUNNING, task=RunTask.PREDICTION, metadata={'step': 2}))

        if cycle_ids_and_conf1_df.empty:
            self._logger.error('No results generated from models.')
            return cycle_ids_and_conf1_df, pd.DataFrame()

        else:
            # Add tag if using it
            if self._config['add_tag_to_df']:
                cycle_ids_and_conf1_df = cycle_ids_and_conf1_df.merge(
                    self._tag_df[['cycle_id', 'tag']], on='cycle_id', how='left')

            cycle_ids_and_conf1_df['cycle_id'] = cycle_ids_and_conf1_df['cycle_id'].astype(int)

            self._logger.info('FINISH run_models()\n')
            return cycle_ids_and_conf1_df, X_valid


class Trainer(BaseClass):

    def __init__(self, data: pd.DataFrame, logger: logging.Logger, pipeline_config: dict = None, callback: callable = None):

        super().__init__(data, pipeline_config, logger, callback)

        self._logger.info("Ini the beninging. Trainer class v%d.%d.%d initialising.", ver_major, ver_minor, ver_patch)

        # Data
        self._training_cycle_ids = []   # IDs for training set
        self._validation_cycle_ids = []  # IDs for validation set
        self._y_training = []         # Labels for training set
        self._y_validation = []      # Labels for validation set
        # self._class_weights = {}       # Class (labels) weights for training set
        self._dr_model_matrix_dict = {}  # Dict containing models and other info for each model. Each item has the same structure: Rows are DR methods, colmumns are models.
        self._top_models = {}           # Dict with top n models in order of the requested feature. Keys are 'Model #', items are tuples with the row, col of corresponding model in _dr_model_matrix_dict

        self._avg_fisher_score = None

        # DR Modes
        self._dr_modes = self._config['dr_modes']
        self.step_to_method = {
            'run_fisher_score': self.__run_fisher_score,
        }

        # Models:
        self._models = self._config['models']

        # Settings
        self._split_random_state = self._config['split_random_state']
        self._validation_size = self._config['validation_size']

        self._pipeline_mode = 'Train'
        self._logger.info('Trainer class initialised.')

    def train(self):

        self._logger.info("Received data shape is %s", self._input_ts_df.shape)
        self._input_delta_ts_df = self._create_delta_times(self._input_ts_df)

        self._data_ts_df = self._input_delta_ts_df.copy()

        if self._config['plots']['enabled'] and self._config['plots']['plot_ts_simple_pre_lags']:
            fig = self._plot_simple_ts(self._data_ts_df)
            self._save_fig_to_log(fig, 'ts_pre_lag.png')

        self._reference_signal = self.__create_median_reference_signal()

        self._data_preprocessing()

        signal_sections_dict, median_signal = self._find_stationary_sections(self._data_ts_df)

        self._logger.info("Signal_sections_dict is %s", signal_sections_dict)

        if self._config['plots']['enabled'] and self._config['plots']['plot_ts_simple_post_lags']:
            fig = self._plot_simple_ts(self._data_ts_df, signal_sections_dict=signal_sections_dict)
            self._save_fig_to_log(fig, 'ts_post_lag.png')
            fig2, ax = plt.subplots()
            ax.plot(median_signal)
            self._save_fig_to_log(fig2, 'median_signal.png')

        use_section = self._config['data_quality']['use_section']
        if use_section and use_section in signal_sections_dict.keys():
            # Convert indices to delta_time values for filtering
            start_idx, end_idx = signal_sections_dict[use_section]
            cycle_df = self._data_ts_df[self._data_ts_df['cycle_id'] == self._cycle_ids[0]]
            delta_times = cycle_df['delta_time'].values
            start = delta_times[start_idx]
            end = delta_times[end_idx]
            self._data_ts_df = self._data_ts_df[(self._data_ts_df['delta_time'] >= start) & (self._data_ts_df['delta_time'] <= end)]
            self._logger.info("Using section %s from %ds to %ds", use_section, start, end)

            if self._config['plots']['enabled'] and self._config['plots']['plot_ts_simple_post_lags']:
                fig = self._plot_simple_ts(self._data_ts_df)
                self._save_fig_to_log(fig, f'ts_post_lag_section_{use_section}.png')

        self._reference_signal = self.__create_median_reference_signal()
        self._logger.info('New reference signal from median values. length: %d.', len(self._reference_signal))

        self.__split_data_by_id()

        self._notify(UpdateMessage(status=RunState.RUNNING, task=RunTask.TRAIN, metadata={'step': 1, 'message': 'Extracting features...'}))
        self._feature_extraction()

        self.__clean_and_clip(self._features_ft_df)

        self._notify(UpdateMessage(status=RunState.RUNNING, task=RunTask.TRAIN, metadata={'step': 1, 'message': 'Preparing features...'}))
        self._normalise_features()

        self.__dimensionality_reduction()

        if self._config['data_quality']['enabled']:
            self.__evaluate_input_data_quality()

        self._notify(UpdateMessage(status=RunState.RUNNING, task=RunTask.TRAIN, metadata={'step': 1, 'message': 'Training and optimising ML models...'}))
        self.__ML_training()

        self.__find_top_models()

    def get_top_models(self):
        """
        Returns an array with the top n db_models for database export
        """

        top_models = []
        # Display the top models for logging purposes
        for key, (dr_model, column_name) in self._top_models.items():

            # Extract the object from the dataframe using the row (dr_model) and column name
            model_object = self._dr_model_matrix_dict['models'].loc[dr_model, column_name]

            top_models.append(model_object)

        return top_models

    def __create_median_reference_signal(self):

        # Use all data if 'label' column is not present, otherwise use label==1
        if 'label' in self._data_ts_df.columns:
            df_label1 = self._data_ts_df[self._data_ts_df["label"] == 1]
        else:
            df_label1 = self._data_ts_df

        # Pivot: rows = time index within cycle, columns = cycle_id
        # This assumes index is 0...N-1 within each cycle (or reset)
        pivoted = df_label1.groupby("cycle_id")["value"].apply(list).apply(pd.Series)
        # Transpose so rows = time step, columns = cycles
        pivoted = pivoted.T
        # Compute median at each time step
        median_signal = pivoted.median(axis=1)

        return median_signal.values

    def __dimensionality_reduction(self):
        """
        Applies dimensionality reduction (DR) techniques to the dataset for various modes and steps.

        This method processes different DR modes defined in `self.dr_modes`, where each mode contains
        several steps (functions) that will be executed sequentially.

        It retrieves the DR function associated with each step and applies it to the dataset.
        The results are stored back in `self.dr_modes` to keep track of transformations.
        """
        self._logger.info('START dimensionality_reduction()')

        # Iterate over each DR mode (like PCA, TSNE, etc.)
        for mode, steps in self._dr_modes.items():
            self._logger.info("Processing DR Mode: %s", mode)  # Log the mode being processed

            # Iterate over each step within the current DR mode
            for step, kwargs in steps.items():

                if step != 'column_mask':
                    self._logger.info("Processing Step: %s, with arguments: %s", step, kwargs)  # Log the step and its arguments

                    # Try to retrieve the corresponding DR function (step) from the current class
                    dr_function = self.step_to_method.get(step)
                    if dr_function is None:
                        self._logger.error('Error: Function "%s" failed to be retrieved for DR mode: %s.', step, mode)
                        continue  # Skip this step and proceed with the next one

                    # Call the retrieved function with the arguments and row/column masks
                    try:
                        steps['column_mask'] = dr_function(**kwargs, row_mask=self._training_cycle_ids, column_mask=steps.get('column_mask'))
                        # Log the updated column mask
                        self._logger.info("Updated column_mask from %s (%d): %s", step, len(steps['column_mask']), steps['column_mask'])
                    except Exception as e:
                        self._logger.info("Error during the execution of function '%s' for DR mode '%s': %s", step, mode, e)

            self._logger.info("Final column_mask for mode %s (%d): %s", mode, len(steps['column_mask']), steps['column_mask'])

        self._logger.info('FINISH dimensionality_reduction()\n')

    def __evaluate_input_data_quality(self):

        # Input data quality evaluation
        data_quality = self._config['data_quality']
        if data_quality['method'] == 'fisher_score':
            if self._avg_fisher_score is not None:
                if self._avg_fisher_score >= data_quality['threshold']:
                    self._logger.info('Average fisher score in selected features is above threshold.')
                else:
                    self._logger.warning('Average fisher score in selected features is BELOW threshold.')
                    if data_quality['fisher_score_cancel']:
                        pass  # TODO Cancel reduction code here
            else:
                self._logger.warning('Fisher score method requested but score not computed. Check if the fisher score DR method is active. Computation skipped.')

    def __split_data_by_id(self):
        """
        Splits the dataset into training and validation sets based on unique cycle IDs, and
        creates the corresponding labels (`y_training` and `y_validation`) for each set.

        The process involves:
        - Grouping the dataset by 'cycle_id' and retrieving the first occurrence of each label.
        - Using `train_test_split` to split cycle IDs into training and validation sets.
        - Storing the cycle IDs for training and validation in separate lists.
        - Extracting the labels for training and validation based on the cycle IDs.
        - Optionally, loading a predefined list of training cycle IDs (commented out).

        :return: None. Modifies the class attributes `training_cycle_ids`, `validation_cycle_ids`,
                `y_training`, and `y_validation` in place.
        :rtype: None
        """
        self._logger.info('START split_data_by_id()')
        start = time.perf_counter()

        # Create a DataFrame with cycle IDs and their corresponding labels
        cycle_label_df = self._data_ts_df.groupby('cycle_id').first().reset_index()[['cycle_id', 'label']]

        # Split the data into training and validation sets based on cycle IDs
        train_ts_df, validation_ts_df = train_test_split(
            cycle_label_df, test_size=self._validation_size, random_state=self._split_random_state)

        # Store the cycle IDs for training and validation
        self._training_cycle_ids = train_ts_df['cycle_id'].tolist()
        self._validation_cycle_ids = validation_ts_df['cycle_id'].tolist()

        # Print the training and validation cycle IDs for debugging
        self._logger.info("Training cycles (%d): %s", len(self._training_cycle_ids), self._training_cycle_ids)
        self._logger.info("Validation cycles (%d): %s", len(self._validation_cycle_ids), self._validation_cycle_ids)

        # Extract labels for the training set based on the cycle IDs
        self._y_training = self._data_ts_df[self._data_ts_df['cycle_id'].isin(self._training_cycle_ids)].groupby('cycle_id').first()['label']

        # Extract labels for the validation set based on the cycle IDs
        self._y_validation = self._data_ts_df[self._data_ts_df['cycle_id'].isin(
            self._validation_cycle_ids)].groupby('cycle_id').first()['label']

        # Print the size and content of the y_training and y_validation sets for debugging
        self._logger.debug("y_training (%d): %s", len(self._y_training), self._y_training.to_list())
        self._logger.debug("y_validation (%d): %s", len(self._y_validation), self._y_validation.to_list())

        end = time.perf_counter()
        self._logger.info("FINISH split_data_by_id() in %.2f s\n", end-start)

    def __ML_training(self):
        """
        Trains and evaluates machine learning models for each dimensionality reduction (DR) mode,
        using the training and validation data. Models are evaluated and stored in a DataFrame.

        The method supports various models and DR techniques, and applies both row and column filters
        for training and validation datasets based on the selected DR mode and features.

        After training, the models are stored, and the accuracy is extracted for evaluation.
        """
        self._logger.info('START ML_training()')
        start = time.perf_counter()

        # DataFrame to hold models
        dr_model_matrix_df = pd.DataFrame()

        # Iterate over each DR mode (e.g., PCA, TSNE, etc.)
        for mode, steps in self._dr_modes.items():
            self._logger.info("Creating models for DR mode %s:", mode)  # Log the mode being processed

            # Create masks for training and validation sets based on cycle_ids
            training_row_mask = self._normalised_features_ft_df['cycle_id'].isin(self._training_cycle_ids)
            validation_row_mask = self._normalised_features_ft_df['cycle_id'].isin(self._validation_cycle_ids)

            # Apply row and column filters to select features.
            # Sort alphabetically to maintain column order during PCA
            X_train = self._normalised_features_ft_df.loc[training_row_mask, steps['column_mask']]
            X_train = X_train[sorted(X_train.columns)]
            X_validation = self._normalised_features_ft_df.loc[validation_row_mask, steps['column_mask']]
            X_validation = X_validation[sorted(X_validation.columns)]

            row_data = {'dr_mode': mode}

            # Iterate over the models to train
            # for model_idx, (model, kwargs) in enumerate(self.models.items(), 1):
            # Skip model training if no models are defined
            if not self._models:
                self._logger.warning("No models defined in configuration. Skipping model training for this DR mode.")
            else:
                for model_idx, (model_group, model_details) in enumerate(self._models.items(), 1):
                    model, kwargs = next(iter(model_details.items()))
                    self._logger.info("Processing model %s with arguments: %s", model, kwargs)  # Log model details

                    # Handle base estimator (if it's a string, convert to the actual estimator)
                    if 'estimator' in kwargs and isinstance(kwargs['estimator'], str):
                        estimator_class = CLASSIFIERS.get(kwargs['estimator'])
                        if estimator_class:
                            kwargs['estimator'] = estimator_class()
                        else:
                            self._logger.info("Error: Estimator '%s' not found.", kwargs['estimator'])
                            continue  # Skip this model if the estimator is invalid

                    # # Adjust class weights if necessary
                    # if kwargs.get('class_weight') == 'self.class_weights':
                    #     kwargs['class_weight'] = self._class_weights

                    # Retrieve the model function from globals
                    model_function = CLASSIFIERS.get(model)
                    if model_function is None:
                        self._logger.warning("Model function '%s' not found.", model)
                        continue  # Skip this model if the function is missing

                    # Train and evaluate the model
                    try:
                        db_model = self.__fit_and_evaluate_model(model_function(**kwargs), X_train, X_validation)
                        db_model.model_name = f'{mode}_{model}_{model_idx}'
                        accuracy = db_model.model_metadata['scores']['accuracy']
                        self._logger.info('%s accuracy: %f.', db_model.model_name, accuracy)
                        self._logger.debug('%s metadata: %s.', db_model.model_name, db_model.model_metadata)

                        # Store the model in the row_data dictionary
                        col_name = db_model.model_name.split('_')[-1]  # Use the model index for the column name
                        row_data[col_name] = db_model

                    except Exception:
                        self._logger.exception("Error during training/evaluation of %s", model)
                        db_model = None  # Set db_model to None if an error occurs
                        pass  # Continue with the next model after plotting if an error occurs

                    if self._config['plots']['enabled'] and self._config['plots']['plot_model_2D_train'] and db_model is not None:
                        plot_model_2D(db_model, self._log_folder, X_train, self._y_training, X_validation, self._y_validation, self._config['plots']['plot_model_2D_train_quality'])

                    if self._config['plots']['enabled'] and self._config['plots']['plot_model_3D_train'] and X_validation.shape[1] == 3:
                        plot_model_3D(db_model=db_model,
                                      targetFolder=self._log_folder,
                                      X_train=X_train,
                                      y_train=self._y_training,
                                      X_validation=X_validation,
                                      y_validation=self._y_validation,
                                      quality=self._config['plots']['plot_model_3D_train_quality'])

                    if self._config['plots']['enabled'] and self._config['plots']['plot_model_3D_with_vectors'] and X_validation.shape[1] == 3:
                        plot_model_3D(db_model=db_model,
                                      targetFolder=self._log_folder,
                                      X_train=X_train,
                                      y_train=self._y_training,
                                      X_validation=X_validation,
                                      y_validation=self._y_validation,
                                      quality=self._config['plots']['plot_model_3D_train_quality'],
                                      show_vectors=True,
                                      show_model=False)

            # Convert the row data into a DataFrame and append to the existing model matrix DataFrame
            row_df = pd.DataFrame([row_data])
            dr_model_matrix_df = pd.concat([dr_model_matrix_df, row_df], ignore_index=True)

        # Store the final model matrix dictionary
        self._dr_model_matrix_dict = {'models': dr_model_matrix_df}

        # If models = 0:
        #   notify(state=ERROR, msg='No models could be created')
        # raise

        # Extract accuracy scores for further evaluation
        self.__extract_model_scores('accuracy')
        end = time.perf_counter()
        self._logger.info("FINISH ML_training() in %.2f s\n", end-start)

    def __fit_and_evaluate_model(self, model_function, X_train,  X_validation):
        """
        This function fits and evaluates all models in _models in parallel.

        :param X_train: training set cycles, selected features, no labels or cycle_id
        :param X_validation: validation set cycles, selected features, no labels or cycle_id
        :param y_train: training set cycles labels
        :param y_validation: validation set cycles labels
        # :param class_weights: dict containing ratio of classes zero and one.

        """
        self._logger.info('START fit_and_evaluate_model()')

        # Fit the model with/without the class weights
        if "AdaBoost" in str(model_function):  # AdaBoost - apply class_weights only
            sample_weights = [self._class_weights[y_val] for y_val in self._y_training]
            model_function.fit(X_train, self._y_training, sample_weight=sample_weights)

        else:
            model_function.fit(X_train, self._y_training)

        # Predict the classification for the validation set X_validation, output to cluster_labels
        y_predicted = model_function.predict(X_validation)

        # Calculate the metrics based on true test set answers (y_valid) and predicted test set answers (cluster_labels)
        accuracy = accuracy_score(self._y_validation, y_predicted)
        precision = precision_score(self._y_validation, y_predicted)
        recall = recall_score(self._y_validation, y_predicted)
        conf_matrix = confusion_matrix(self._y_validation, y_predicted)
        f1 = f1_score(self._y_validation, y_predicted)
        matthews_cor = matthews_corrcoef(self._y_validation, y_predicted)
        roc_auc = roc_auc_score(self._y_validation, y_predicted)
        balanced_accuracy = balanced_accuracy_score(self._y_validation, y_predicted)

        model_parameters = make_json_serializable(model_function.get_params())

        pca = PCA(n_components=2)
        x_train_pca = pca.fit_transform(X_train)
        pca_params = {
            'mean': [float(val) for val in pca.mean_],  # Mean of each feature before scaling
            'components': [[float(val) for val in component] for component in pca.components_],  # Principal axes in feature space
            'explained_variance': [float(val) for val in pca.explained_variance_],  # Amount of variance explained by each of the selected components
            'explained_variance_ratio': [float(val) for val in pca.explained_variance_ratio_],  # Percentage of variance explained by each of the selected components
            'singular_values': [float(val) for val in pca.singular_values_],  # The singular values corresponding to each of the selected components
            'x_min': float(x_train_pca[:, 0].min()),
            'x_max': float(x_train_pca[:, 0].max()),
            'y_min': float(x_train_pca[:, 1].min()),
            'y_max': float(x_train_pca[:, 1].max()),
        }

        # Prepare scaling type variable
        scaling_type = 'Standard'
        if 'scaling_type' in self._config:
            scaling_type = self._config['scaling_type']

        # Instantiate a new model object containing the relevant information and results of the model.
        new_db_model = models.Model(
            # model_onnx=sk_to_onnx(X_train.shape[1], model_function),
            model_onnx=model_function,
            model_metadata={
                # 'train_time_ns': epoch_train_time,
                # 'evaluate_time_ns': epoch_evaluate_time,
                'model_parameters': model_parameters,
                'input_dim': X_train.shape[1],
                'validation': {
                    'y_validation': self._y_validation.tolist(),
                    'y_predicted': y_predicted.tolist()
                },
                'scores': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'conf_matrix': conf_matrix.tolist(),
                    'f1': f1,
                    'matthews_cor': matthews_cor,
                    'roc_auc': roc_auc,
                    'balanced_accuracy': balanced_accuracy
                },
                'features': {col: self._feature_dict[col] for col in X_train.columns},
                'reference_signal': self._reference_signal.tolist(),
                'pca_params': pca_params,
                'scaling_type': scaling_type
            })

        if "HDBSCAN_Clustering" in str(model_function):
            data_converted = {str(k): v.tolist() for k, v in model_function.medoids.items()}
            new_db_model.model_metadata['medoids'] = data_converted

            data_converted = make_json_serializable(model_function.std_devs)
            new_db_model.model_metadata['std_devs'] = data_converted

            #  Add spread factors to db
            data_converted = model_function.spread_factor_ax.tolist()
            new_db_model.model_metadata['spread_factor_ax'] = data_converted
            data_converted = model_function.spread_factor_perp.tolist()
            new_db_model.model_metadata['spread_factor_perp'] = data_converted

            # Include init params in metadata
            new_db_model.model_metadata["params"] = model_function.params

        # Serialise the estimator metadata
        if 'estimator' in new_db_model.model_metadata['model_parameters']:
            new_db_model.model_metadata['model_parameters']['estimator'] = make_json_serializable(
                new_db_model.model_metadata['model_parameters']['estimator'])

        self._logger.info('FINISH fit_and_evaluate_model()\n')

        return new_db_model

    def __find_top_models(self, n_models=3):
        """
        Identifies the top 'n' models based on the highest accuracy scores.

        This function assumes that 'dr_model_matrix_dict' contains a DataFrame with accuracy values,
        and it extracts the top 'n' models based on those accuracy scores.

        :param n_models: Number of top models to return based on accuracy (default is 3).
        :return: None. The top models are stored in the `self.top_models` dictionary.
        """
        self._logger.info('START find_top_models()')
        start = time.perf_counter()

        # Check if the accuracy data exists in the model matrix
        try:
            accuracy_df = self._dr_model_matrix_dict['accuracy']
        except KeyError:
            self._logger.exception("Error: 'accuracy' data not found in model matrix.")
            return

        # Ensure the dataframe has valid accuracy data
        if accuracy_df.empty:
            self._logger.error("Error: The 'accuracy' DataFrame is empty.")
            return

        # Get the top 'n_models' accuracy values
        n_models = self._config['n_models']  # Max number of standard deviations to accept as valid lag
        top_values = accuracy_df.drop(columns='dr_mode').stack().nlargest(n_models)

        # Store the indices of the top models in a dictionary
        self._top_models = {f'Model {i + 1}': index for i, index in enumerate(top_values.index)}

        # Display the top models for logging purposes
        self._logger.info("Top %d models:", n_models)
        for key, (dr_model, column_name) in self._top_models.items():

            # Extract the object from the dataframe using the row (dr_model) and column name
            model_object = self._dr_model_matrix_dict['models'].loc[dr_model, column_name]

            # Get the model_name property from the object
            accuracy = model_object.model_metadata['scores']['accuracy']

            self._logger.info("%s: %f", model_object.model_name, accuracy)

        end = time.perf_counter()
        self._logger.info("FINISH find_top_models() in %f s\n", end-start)

    def __clean_and_clip(self, df):
        """
        Cleans the DataFrame by removing columns with NaN or infinite values, and clips feature values to
        ensure they fall within a valid range. The function performs the following steps:
        1. Analyzes the DataFrame to report NaN or infinite values.
        2. Replaces infinite values with NaN and drops columns containing NaN values.
        3. Clips the values to fall within the range [-1e15, 1e15].

        :param df: The DataFrame to clean and clip.
        :type df: pandas.DataFrame

        :return: A cleaned and clipped DataFrame.
        :rtype: pandas.DataFrame
        """
        # TODO report which columns were removed.
        self._logger.info('START clean_and_clip()')

        # Analyze the dataframe for NaN and infinite values
        self.__analyze_df(df)

        # Clean feature values:
        # - Replace infinite values (both positive and negative) with NaN.
        # - Drop columns that contain any NaN values.
        # - Clip values to ensure they are within the specified range of -1e15 to 1e15.
        try:
            # Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Drop columns with NaN values
            df.dropna(axis=1, how='any', inplace=True)

            # Clip the values within the range [-1e15, 1e15]
            df.clip(lower=-1e15, upper=1e15, inplace=True)

        except Exception as e:
            self._logger.info("An error occurred while cleaning and clipping the DataFrame: %s", e)
            return None

        self._logger.info('FINISH clean_and_clip()\n')

        return df  # Return the cleaned and clipped DataFrame

    def __analyze_df(self, df):
        """
        Analyzes a DataFrame to identify NaN and infinite values, as well as the minimum and maximum values
        while ignoring NaN and infinite values. This function provides a detailed report on where NaN or
        infinite values occur and also outputs the minimum and maximum values in the DataFrame.

        :param df: The DataFrame to analyze.
        :type df: pandas.DataFrame

        :return: A list of dictionaries containing details of NaN and infinite values.
        :rtype: list
        """
        nan_inf_report = []

        # Loop through each row and column to identify NaN or infinite values
        for idx, row in df.iterrows():
            for col in df.columns:
                value = row[col]
                if pd.isna(value):  # Check for NaN
                    nan_inf_report.append({
                        'Row_Index': idx,
                        'Cycle_ID': row['cycle_id'],
                        'Column': col,
                        'Value': 'NaN'
                    })
                elif np.isinf(value):  # Check for infinity (after replacement)
                    nan_inf_report.append({
                        'Row_Index': idx,
                        'Cycle_ID': row['cycle_id'],
                        'Column': col,
                        'Value': 'inf' if value > 0 else '-inf'
                    })

        # Replace infinities with NaN for easier handling and only loop over rows with valid data
        df_cleaned = df.replace([np.inf, -np.inf], np.nan)

        # Calculate min and max values, ignoring NaNs and infinities
        min_value = df_cleaned.min().min()
        max_value = df_cleaned.max().max()

        # Display results
        if nan_inf_report:
            self._logger.info("NaN or inf values found:")
            for item in nan_inf_report:
                self._logger.warning("Cycle ID: %d, Column: %s, Value: %s", int(item['Cycle_ID']), item['Column'], item['Value'])
        else:
            self._logger.info("No NaN or inf values found.")

        # Output min and max values
        self._logger.info("Minimum value in the DataFrame (ignoring NaN and inf): %d", min_value)
        self._logger.info("Maximum value in the DataFrame (ignoring NaN and inf): %d", max_value)

        return nan_inf_report

    def _compute_fisher_score(self, df, feature, class_col='classification'):
        """
        Compute the Robust Fisher Score for a feature in a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe containing the data. One row corresponds to a cycle, each column is a feature. An additional column is the classe of the cycle.
        feature : str
            The name of the feature column.
        class_col : str
            The name of the classification column.

        Returns
        -------
        float
            The Fisher Score for the feature.
        """
        # Extract the class labels
        labels = df[class_col].values
        classes = np.unique(labels)

        # Ensure there are only two classes
        if len(classes) != 2:
            return 0
            # raise ValueError(f"Fisher's linear discriminant is only defined for two classes. There are {len(classes)} classes.")

        # Extract feature data
        feature_data = df[feature].values
        means = []
        deviations = []

        global_mean = np.mean(feature_data)

        # Compute means and deviations for each class
        for c in classes:
            class_data = feature_data[labels == c]
            means.append(np.mean(class_data))
            deviations.append(np.mean(np.abs(class_data - np.mean(class_data))))

        # Calculate Fisher Score for the feature
        score = (np.abs(means[0] - global_mean) + np.abs(means[1] - global_mean)) / np.max(((deviations[0] + deviations[1]), 1e-2))

        # Adjust Fisher Score based on the number of samples
        sample_count = len(df)
        score = score * np.log(sample_count)

        return score

    def __get_valid_feature_combinations(self, fisher_scores, similarity_df, threshold, top_n):
        """
        Get all possible feature combinations of length top_n with pairwise similarity below threshold.
        """

        # for k in fisher_scores.keys():
        #     print(k)
        # Only keep best parameters for each feature in fisher_scores
        features = [key.split('__')[0] for key in fisher_scores.keys()]

        # drop duplicates from features
        features = list(dict.fromkeys(features))

        # Keep only the best parameter for each feature
        for feature in features:
            # if feature != 'CWT': #CB testing
            keys = [key for key in fisher_scores.keys() if key.split('__')[0] == feature]
            max_key = max(keys, key=lambda x: fisher_scores[x])
            for key in keys:
                if key != max_key:
                    fisher_scores.pop(key)

        features = list(fisher_scores.keys())

        valid_combinations = []

        # Function to check if a combination of features is valid based on pairwise similarity
        def is_valid_combination(combination):
            # Check pairwise similarity in the combination
            for i, feat1 in enumerate(combination):
                for feat2 in combination[i+1:]:
                    feat1_name = feat1.split('__')[0]
                    feat2_name = feat2.split('__')[0]
                    if feat1_name not in similarity_df.index or feat2_name not in similarity_df.columns:  # if not in similarity_df assume no issue.
                        continue
                    if similarity_df.loc[feat1_name, feat2_name] >= threshold:
                        return False
            return True

        # Generate all combinations of features with length top_n
        feature_combinations = list(combinations(features, top_n))
        for combination in tqdm(feature_combinations, total=len(feature_combinations), desc='Possible combinations', disable=True):
            if is_valid_combination(combination):
                valid_combinations.append(combination)

        # Compute sum of fisher scores for each combination and sort them
        valid_combinations = {combination: sum([fisher_scores[feature] for feature in combination]) for combination in valid_combinations}
        valid_combinations = dict(sorted(valid_combinations.items(), key=lambda x: x[1], reverse=True))

        return valid_combinations

    def __run_fisher_score(self, similarity_threshold=.7, top_n=3, row_mask='', column_mask=''):

        # If empty mask, use all columns
        if column_mask == '':
            column_mask = self._normalised_features_ft_df.columns.drop('cycle_id').to_list()

        # Separate bad and good signals
        training_cycle_ids_bad = self._y_training[self._y_training == 0].index.tolist()
        training_cycle_ids_good = self._y_training[self._y_training == 1].index.tolist()
        df = self._normalised_features_ft_df

        training_bad_df = df[df['cycle_id'].isin(training_cycle_ids_bad)].assign(classification=0)
        training_good_df = df[df['cycle_id'].isin(training_cycle_ids_good)].assign(classification=1)
        df = pd.concat([training_good_df, training_bad_df], axis=0)
        df = df.drop(columns='cycle_id')

        # Read similarity scores
        features_similarity_df = pd.DataFrame.from_dict(features_similarity)

        # Compute Fisher scores for all features
        fisher_scores = {feature: self._compute_fisher_score(df, feature) for feature in df.drop(columns='classification').columns}

        # Plot?
        debug_features = self._config['debug_features']
        if debug_features['enabled'] and 'fisher_score' in debug_features:
            if debug_features['fisher_score']:
                # Plot Fisher scores in pages of n_plots_per_page features
                n_plots_per_page = 10
                feature_list = list(fisher_scores.keys())
                for i in range(0, len(feature_list), n_plots_per_page):
                    feature_range = (i, min(i + n_plots_per_page, len(feature_list)))
                    fig = plot_fisher_score(df, fisher_scores, feature_range)
                    self._save_fig_to_log(fig, f'fisher_score_{i//n_plots_per_page + 1}.png')

        valid_feature_sets = self.__get_valid_feature_combinations(fisher_scores, features_similarity_df, similarity_threshold, top_n)
        # print(f'There are {len(valid_feature_sets)} valid feature sets of {top_n} features with a similarity threshold below {similarity_threshold}.')

        best_features_set = list(valid_feature_sets.keys())[0]
        # print(f'The best feature set is {best_features_set} with a sum of Fisher scores of {valid_feature_sets[best_features_set]}.')

        # # Compute average score for top 10 feature sets
        # top_n_average_score = 10
        # average_score = np.mean([valid_feature_sets[feature_set] for feature_set in list(valid_feature_sets.keys())[:top_n_average_score]])
        # print(f'The average score for the top {top_n_average_score} feature sets is {average_score}.')

        self._avg_fisher_score = valid_feature_sets[best_features_set]/top_n

        selected_features = [feature for feature in best_features_set]

        if debug_features['enabled'] and 'similarity_matrix' in debug_features:
            if debug_features['similarity_matrix']:
                fig = plot_feature_similarity_heatmap(features_similarity, highlight_features=selected_features, fisher_scores=fisher_scores)
                self._save_fig_to_log(fig, f'Feature_redundancy_heatmap.png')

        return selected_features

    def __extract_model_scores(self, score_name):

        df = self._dr_model_matrix_dict['models']

        try:
            # these lines extract {score_name} to a new dict entry, unless there's no db_model object in the cell
            # Use DataFrame.apply with np.vectorize to avoid deprecated applymap
            def extract_score(d):
                if hasattr(d, 'model_metadata') and 'scores' in d.model_metadata:
                    return d.model_metadata['scores'][score_name]
                else:
                    return None

            self._dr_model_matrix_dict[score_name] = pd.concat(
                [df[['dr_mode']],
                 df.drop(columns='dr_mode').apply(lambda col: col.map(extract_score))], axis=1)
        except Exception:
            self._logger.exception("Error attempting to extract model score(%s).", score_name)
