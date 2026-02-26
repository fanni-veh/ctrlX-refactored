"""
This function was extracted from tsfresh library :
https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
You'll find only meaningful functions here, not all from the original `tsfresh` feature_extraction function.
We selected these functions based on computational complexity, logical relevance and dimensionality reductions approches.
Any functions with linear O(N) or worse complexity were discarded, as were features deemed useless, like mean, median, and standard deviation.
"""
# Libraries
from builtins import range
from collections import defaultdict
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import find_peaks_cwt, welch
from scipy.stats import linregress
from app.insight.feature_plotter import plot_energy_ratio_by_chunks, plot_c3, plot_cid_ce
from app.insight.feature_plotter import plot_absolute_sum_of_changes
from app.insight.feature_plotter import plot_autocorrelation, plot_agg_autocorrelation
from app.insight.feature_plotter import plot_change_quantiles, plot_fft_aggregated
from app.insight.feature_plotter import plot_agg_linear_trend, plot_binned_entropy
from app.insight.feature_plotter import plot_mean_second_derivative_central, plot_fft_coefficient
from app.insight.feature_plotter import plot_cwt_coefficients
from scipy.stats import entropy as scipy_entropy
from scipy.signal import find_peaks


import re
import pywt


def _dict_to_filename(data: dict, sep: str = "_") -> str:
    parts = [f"{key}={value}" for key, value in data.items()]
    raw_name = sep.join(parts)

    # Replace invalid filename characters with underscores
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', raw_name)

    # Optional: truncate if filename is too long
    return safe_name


def _into_subchunks(x, subchunk_length, every_n=1):
    """
    Split the time series x into subwindows of length "subchunk_length", starting every "every_n".

    For example, the input data if [0, 1, 2, 3, 4, 5, 6] will be turned into a matrix

        0  2  4
        1  3  5
        2  4  6

    with the settings subchunk_length = 3 and every_n = 2
    """
    len_x = len(x)

    assert subchunk_length > 1
    assert every_n > 0

    # how often can we shift a window of size subchunk_length over the input?
    num_shifts = (len_x - subchunk_length) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(subchunk_length)

    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
    return np.asarray(x)[indexer]


def _aggregate_on_chunks(x, f_agg, chunk_len):
    """
    Takes the time series x and constructs a lower sampled version of it by applying the aggregation function f_agg on
    consecutive chunks of length chunk_len

    :param x: the time series to calculate the aggregation of
    :type x: numpy.ndarray
    :param f_agg: The name of the aggregation function that should be an attribute of the pandas.Series
    :type f_agg: str
    :param chunk_len: The size of the chunks where to aggregate the time series
    :type chunk_len: int
    :return: A list of the aggregation function over the chunks
    :return type: list
    """
    return [
        getattr(x[i * chunk_len: (i + 1) * chunk_len], f_agg)()
        for i in range(int(np.ceil(len(x) / chunk_len)))
    ]


def _roll(a, shift):
    """
    Roll 1D array elements. Improves the performance of numpy.roll() by reducing the overhead introduced from the
    flexibility of the numpy.roll() method such as the support for rolling over multiple dimensions.

    Elements that roll beyond the last position are re-introduced at the beginning. Similarly, elements that roll
    back beyond the first position are re-introduced at the end (with negative shift).

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=2)
    >>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=-2)
    >>> array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=12)
    >>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    Benchmark
    ---------
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> %timeit _roll(x, shift=2)
    >>> 1.89 µs ± 341 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> %timeit np.roll(x, shift=2)
    >>> 11.4 µs ± 776 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    :param a: the input array
    :type a: array_like
    :param shift: the number of places by which elements are shifted
    :type shift: int

    :return: shifted array with the same shape as a
    :return type: ndarray
    """
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])


def absolute_sum_of_changes(x, param=None, _logging_dict=None):
    """
    Returns the sum over the absolute value of consecutive changes in the series x

    .. math::

        \\sum_{i=1, \\ldots, n-1} \\mid x_{i+1}- x_i \\mid

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """

    asc = np.sum(np.abs(np.diff(x)))

    # Debug data
    if _logging_dict is not None:
        if 'logger' in _logging_dict:
            logger = _logging_dict['logger']
            logger.debug('Feature: absolute_sum_of_changes is %f.', asc)

        if 'self_instance' in _logging_dict:
            self_instance = _logging_dict['self_instance']
            if 'cycle_id' in _logging_dict:
                cycle_id = _logging_dict['cycle_id']
            else:
                cycle_id = len(x)

            label = _logging_dict.get('label', None)
            fig = plot_absolute_sum_of_changes(x, label=label)
            self_instance._save_fig_to_log(fig, f'{cycle_id}_absolute_sum_of_changes_{label}.png')

    return asc


def agg_autocorrelation(x, param, _logging_dict=None):
    """
    Descriptive statistics on the autocorrelation of the time series.

    Calculates the value of an aggregation function :math:`f_{agg}` (e.g. the variance or the mean) over the
    autocorrelation :math:`R(l)` for different lags. The autocorrelation :math:`R(l)` for lag :math:`l` is defined as

    .. math::

        R(l) = \\frac{1}{(n-l)\\sigma^2} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

    where :math:`X_i` are the values of the time series, :math:`n` its length. Finally, :math:`\\sigma^2` and
    :math:`\\mu` are estimators for its variance and mean
    (See `Estimation of the Autocorrelation function <http://en.wikipedia.org/wiki/Autocorrelation#Estimation>`_).

    The :math:`R(l)` for different lags :math:`l` form a vector. This feature calculator applies the aggregation
    function :math:`f_{agg}` to this vector and returns

    .. math::

        f_{agg} \\left( R(1), \\ldots, R(m)\\right) \\quad \\text{for} \\quad m = max(n, maxlag).

    Here :math:`maxlag` is the second parameter passed to this function.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"f_agg": x, "maxlag", n} with x str, the name of a numpy function
                  (e.g. "mean", "var", "std", "median"), its the name of the aggregator function that is applied to the
                  autocorrelations. Further, n is an int and the maximal number of lags to consider.
    :type param: list
    :return: the value of this feature
    :return type: float
    """

    # if the time series is longer than the following threshold, we use fft to calculate the acf
    THRESHOLD_TO_USE_FFT = 1250
    var = np.var(x)
    n = len(x)
    max_maxlag = max([config["maxlag"] for config in param])

    if np.abs(var) < 10**-10 or n == 1:
        a = [0] * len(x)
    else:
        a = acf(x, adjusted=True, fft=n > THRESHOLD_TO_USE_FFT, nlags=max_maxlag)[1:]

    # Debug data
    if _logging_dict is not None:
        # if 'logger' in _logging_dict:
        #     logger = _logging_dict['logger']
        # log result here
        # logger.debug('Feature: agg_autocorrelation is %f %s.', x, param)

        if 'self_instance' in _logging_dict:
            self_instance = _logging_dict['self_instance']
            if 'cycle_id' in _logging_dict:
                cycle_id = _logging_dict['cycle_id']
            else:
                cycle_id = len(x)

            label = _logging_dict.get('label', None)
            fig = plot_agg_autocorrelation(x, param, label=label)
            self_instance._save_fig_to_log(fig, f'{cycle_id}_agg_autocorrelation_{label}.png')

    return [
        (
            'f_agg_"{}"__maxlag_{}'.format(config["f_agg"], config["maxlag"]),
            getattr(np, config["f_agg"])(a[: int(config["maxlag"])]),
        )
        for config in param
    ]


def agg_linear_trend(x, param, _logging_dict=None):
    """
    Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
    the sequence from 0 up to the number of chunks minus one.

    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

    The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
    "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

    The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

    Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

    Notes:
    ------
    len(x) has to be greater than chunk_len or np.nan is returned.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    # todo: we could use the index of the DataFrame here

    calculated_agg = defaultdict(dict)
    res_data = []
    res_index = []

    for parameter_combination in param:

        chunk_len = parameter_combination["chunk_len"]
        f_agg = parameter_combination["f_agg"]

        if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
            if chunk_len >= len(x):
                calculated_agg[f_agg][chunk_len] = np.nan
            else:
                aggregate_result = _aggregate_on_chunks(x, f_agg, chunk_len)
                lin_reg_result = linregress(
                    range(len(aggregate_result)), aggregate_result
                )
                calculated_agg[f_agg][chunk_len] = lin_reg_result

        attr = parameter_combination["attr"]

        if chunk_len >= len(x):
            res_data.append(np.nan)
        else:
            res_data.append(getattr(calculated_agg[f_agg][chunk_len], attr))

        res_index.append(
            'attr_"{}"__chunk_len_{}__f_agg_"{}"'.format(attr, chunk_len, f_agg)
        )

        # Debug data
        if _logging_dict is not None:
            # if 'logger' in _logging_dict:
            #     logger = _logging_dict['logger']
            # log result here
            # logger.debug('Feature: agg_autocorrelation is %f %s.', x, parameter_combination)

            if 'self_instance' in _logging_dict:
                self_instance = _logging_dict['self_instance']
                if 'cycle_id' in _logging_dict:
                    cycle_id = _logging_dict['cycle_id']
                else:
                    cycle_id = len(x)

                label = _logging_dict.get('label', None)
                fig = plot_agg_linear_trend(x, parameter_combination, label=label)
                self_instance._save_fig_to_log(fig, f'{cycle_id}_agg_linear_trend_{_dict_to_filename(parameter_combination)}_{label}.png')

    return zip(res_index, res_data)


def autocorrelation(x, lags, _logging_dict=None):
    """
    Calculates the autocorrelation for the specified lags.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lags: list of lag values
    :type lags: list
    :return: array of autocorrelation values for each lag
    :return type: numpy.ndarray
    """
    if isinstance(x, pd.Series):
        x = x.values

    n = len(x)
    x_mean = np.mean(x)
    v = np.var(x)
    if np.isclose(v, 0):
        return np.full(len(lags), np.nan)

    autocorr_values = []
    for lag in lags:
        if lag >= n:
            autocorr_values.append(np.nan)
        else:
            y1 = x[: (n - lag)]
            y2 = x[lag:]
            sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
            autocorr_value = sum_product / ((n - lag) * v)
            autocorr_values.append(autocorr_value)

            # Debug data
            if _logging_dict is not None:
                if 'logger' in _logging_dict:
                    logger = _logging_dict['logger']
                    logger.debug('Feature: autocorrelation for lag %d is %f.', lag, autocorr_value)

                if 'self_instance' in _logging_dict:
                    self_instance = _logging_dict['self_instance']
                    if 'cycle_id' in _logging_dict:
                        cycle_id = _logging_dict['cycle_id']
                    else:
                        cycle_id = len(x)

                    label = _logging_dict.get('label', None)
                    fig = plot_autocorrelation(x, lag, label=label)
                    self_instance._save_fig_to_log(fig, f'{cycle_id}_autocorrelation_lag_{lag}_{label}.png')

    return np.array(autocorr_values)


def binned_entropy(x, max_bins_list, _logging_dict=None):
    """
    Calculates binned entropy for the specified max_bins values.


    (See `Entropy (information theory) <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param max_bins_list: list of maximal number of bins
    :type max_bins_list: list
    :return: array of binned entropy values for each max_bins value
    :return type: numpy.ndarray
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    # nan makes no sense here
    if np.isnan(x).any():
        return np.full(len(max_bins_list), np.nan)

    binned_entropy_values = []
    for max_bins in max_bins_list:
        hist, bin_edges = np.histogram(x, bins=max_bins)
        probs = hist / x.size
        probs[probs == 0] = 1.0  # This avoids log(0), result is not affected as log(1) = 0
        entropy = -np.sum(probs * np.log(probs))
        binned_entropy_values.append(entropy)

        # Debug data
        if _logging_dict is not None:
            if 'logger' in _logging_dict:
                logger = _logging_dict['logger']
                logger.debug('Feature: binned_entropy for bin %d is %f.', max_bins, entropy)

            if 'self_instance' in _logging_dict:
                self_instance = _logging_dict['self_instance']
                if 'cycle_id' in _logging_dict:
                    cycle_id = _logging_dict['cycle_id']
                else:
                    cycle_id = len(x)

                label = _logging_dict.get('label', None)
                fig = plot_binned_entropy(x, max_bins, label=label)
                self_instance._save_fig_to_log(fig, f'{cycle_id}_binned_entropy_bins_{max_bins}_{label}.png')

    return np.array(binned_entropy_values)


def c3(x, lag_list, _logging_dict=None):
    """
    Calculates c3 statistics for the specified lag values.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag_list: list of lag values
    :type lag_list: list
    :return: array of c3 values for each lag
    :return type: numpy.ndarray
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    n = x.size

    c3_values = []
    for lag in lag_list:
        if 2 * lag >= n:
            c3_values.append(0)
        else:
            c3_value = np.mean((_roll(x, 2 * -lag) * _roll(x, -lag) * x)[0: (n - 2 * lag)])
            c3_values.append(c3_value)

            # Debug data
            if _logging_dict is not None:
                if 'logger' in _logging_dict:
                    logger = _logging_dict['logger']
                    logger.debug('Feature: c3 for lag %d is %f.', lag, c3_value)

                if 'self_instance' in _logging_dict:
                    self_instance = _logging_dict['self_instance']
                    if 'cycle_id' in _logging_dict:
                        cycle_id = _logging_dict['cycle_id']
                    else:
                        cycle_id = len(x)

                    label = _logging_dict.get('label', None)
                    fig = plot_c3(x, lag, smoothing_window=40, label=label)
                    self_instance._save_fig_to_log(fig, f'{cycle_id}_c3__lag_{lag}_{label}.png')

    return np.array(c3_values)


def change_quantiles(x, param_list, _logging_dict=None):
    """
    Calculates the feature for the specified parameter combinations.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param_list: list of dictionaries containing parameter values
    :type param_list: list of dicts
    :return: array of feature values for each parameter combination
    :return type: numpy.ndarray
    """
    feature_values = []
    for params in param_list:
        ql = params.get('ql', 0)
        qh = params.get('qh', 1)
        isabs = params.get('isabs', 3)
        f_agg = params.get('f_agg', 4)

        if ql >= qh:
            feature_values.append(0)
            continue

        div = np.diff(x)
        if isabs:
            div = np.abs(div)

        try:
            bin_cat = pd.qcut(x, [ql, qh], labels=False)
            bin_cat_0 = bin_cat == 0
        except ValueError:
            feature_values.append(0)
            continue

        ind = (bin_cat_0 & _roll(bin_cat_0, 1))[1:]

        if np.sum(ind) == 0:
            feature_values.append(0)
        else:
            ind_inside_corridor = np.where(ind == 1)
            aggregator = getattr(np, f_agg)
            feature_value = aggregator(div[ind_inside_corridor])
            feature_values.append(feature_value)

            # Debug data
            if _logging_dict is not None:
                if 'logger' in _logging_dict:
                    logger = _logging_dict['logger']
                    logger.debug(
                        f'Feature: change_quantiles with params {params}, feature_value={feature_value}.'
                    )

                if 'self_instance' in _logging_dict:
                    self_instance = _logging_dict['self_instance']
                    if 'cycle_id' in _logging_dict:
                        cycle_id = _logging_dict['cycle_id']
                    else:
                        cycle_id = len(x)

                    label = _logging_dict.get('label', None)
                    fig = plot_change_quantiles(x, params, bin_cat_0, div, ind_inside_corridor, label=label)
                    self_instance._save_fig_to_log(fig, f'{cycle_id}_change_quantiles_{_dict_to_filename(params)}_{label}.png')

    return np.array(feature_values)


def cid_ce(x, normalize_options, _logging_dict=None):
    """
    Calculate an estimate for a time series complexity for various normalization options.

    :param x: the time series to calculate the feature of
    :type x: list or numpy.ndarray
    :param normalize_options: a list of boolean values indicating normalization options
    :type normalize_options: list

    :return: a dictionary containing normalization options as keys and the corresponding result as values
    :rtype: dict
    """

    results = []

    for normalize in normalize_options:
        x_copy = x.copy()  # Make a copy of the input to avoid modifying the original data

        if normalize is True:
            s = np.std(x_copy)
            if s != 0:
                x_copy = (x_copy - np.mean(x_copy)) / s
            else:
                results[normalize] = 0.0
                continue

        x_diff = np.diff(x_copy)
        result = np.sqrt(np.dot(x_diff, x_diff))

        # Debug data
        if _logging_dict is not None:
            suffix = "normalised" if normalize else "not_normalised"
            if 'logger' in _logging_dict:
                logger = _logging_dict['logger']
                logger.debug('Feature: cid_ce %s is %f.', suffix, result)

            if 'self_instance' in _logging_dict:
                self_instance = _logging_dict['self_instance']
                if 'cycle_id' in _logging_dict:
                    cycle_id = _logging_dict['cycle_id']
                else:
                    cycle_id = len(x)

                label = _logging_dict.get('label', None)
                fig = plot_cid_ce(x, normalize=normalize, label=label)
                self_instance._save_fig_to_log(fig, f'{cycle_id}_cid_ce_{suffix}_{label}.png')

        results.append(result)

    return results


def CWT(x, param=None, _logging_dict=None):

    def extract_cwt_features(cwt_matrix, scale_range=None, time_range=None):
        """
        Extract features from a CWT matrix (power version recommended).

        Parameters:
        - cwt_matrix: 2D numpy array [scales x time], preferably np.abs(coeffs) ** 2
        - scale_range: Optional tuple (low, high) indices for band power (e.g., (0, 5))
        - time_range: Optional tuple (start, end) for time cropping

        Returns:
        - Dictionary of features
        """
        power = np.abs(cwt_matrix) ** 2  # ensure power
        scales, time_len = power.shape

        # Crop time window if requested
        if time_range:
            power = power[:, time_range[0]:time_range[1]]

        # Basic power stats
        mean_power = power.mean()
        std_power = power.std()
        max_power = power.max()

        # Time and scale indices of max power
        argmax_scale = np.argmax(power.mean(axis=1))  # mean over time
        argmax_time = np.argmax(power.mean(axis=0))   # mean over scale

        # Optional band power
        low_band = power[:scales//3].sum()
        mid_band = power[scales//3:2*scales//3].sum()
        high_band = power[2*scales//3:].sum()

        total_power = power.sum()
        band_power_ratio_low = low_band / total_power
        band_power_ratio_high = high_band / total_power

        # Flattened power distribution entropy (normalized)
        flat_power = power.flatten()
        power_probs = flat_power / flat_power.sum()
        spectral_entropy = scipy_entropy(power_probs)

        # Peak counting along max-power scale
        dominant_scale_idx = argmax_scale
        peaks, _ = find_peaks(power[dominant_scale_idx], height=np.percentile(power[dominant_scale_idx], 75))
        num_peaks = len(peaks)

        return {
            "mean_power": mean_power,
            "std_power": std_power,
            "max_power": max_power,
            "argmax_scale": argmax_scale,
            "argmax_time": argmax_time,
            "band_power_low": low_band,
            "band_power_high": high_band,
            "band_ratio_low": band_power_ratio_low,
            "band_ratio_high": band_power_ratio_high,
            "spectral_entropy": spectral_entropy,
            "num_peaks": num_peaks,
        }

    scales_requested = np.arange(1, 20)
    wavelet = 'cmor1.5-1.0'

    cwt_matrix, frequencies = pywt.cwt(x, scales=scales_requested, wavelet=wavelet, sampling_period=0.005)

    feat = extract_cwt_features(cwt_matrix)

    return feat


def cwt_coefficients(x, param, delta_t=0.005, _logging_dict=None):
    """
    Calculates a Continuous wavelet transform using PyWavelets.

    This feature calculator takes three different parameters: wavelet, level, and coeff. 
    For each dict in param, one feature is returned.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"wavelet": str, "level": int, "coeff": int}
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    """
    wavelet = wavelet
    width = scale
    widths = scales_requested
    w = output_scale
    coeff = time_idx

    
    """

    res = []
    indices = []

    # Collect wavelet shape
    wavelet = param[0].get("wavelet", "mexh")

    # collect all widths from param in a single list
    scales_requested = sorted({widths for parameter_combination in param for widths in parameter_combination.get("widths", (1,))})

    # [scales (frequencies), time]
    cwtmatr, frequencies = pywt.cwt(x, scales=scales_requested, wavelet=wavelet, sampling_period=delta_t)

    # Collect (freq, time) pairs from param
    for p in param:
        freq_idx = scales_requested.index(p.get("w", 0))
        time_idx = p.get("coeff", 0)
        time = time_idx * delta_t
        indices.append(f'time_{time}__freq_{frequencies[freq_idx]}__wavelet_{wavelet}')
        res.append(cwtmatr[freq_idx, time_idx])

        # Debug data
    if _logging_dict is not None:
        # if 'logger' in _logging_dict:
        # logger = _logging_dict['logger']
        # logger.debug('Feature: cid_ce %s is %f.', suffix, result)

        if 'self_instance' in _logging_dict:
            self_instance = _logging_dict['self_instance']
            if 'cycle_id' in _logging_dict:
                cycle_id = _logging_dict['cycle_id']
            else:
                cycle_id = len(x)

            label = _logging_dict.get('label', None)
            fig = plot_cwt_coefficients(x, param, label=label)
            self_instance._save_fig_to_log(fig, f'{cycle_id}_cwt_coefficients_{label}.png')

    return zip(indices, res)


def energy_ratio_by_chunks(x, param, _logging_dict=None):
    """
    Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole
    series.

    Takes as input parameters the number num_segments of segments to divide the series into and segment_focus
    which is the segment number (starting at zero) to return a feature on.

    If the length of the time series is not a multiple of the number of segments, the remaining data points are
    distributed on the bins starting from the first. For example, if your time series consists of 8 entries, the
    first two bins will contain 3 and the last two values, e.g. `[ 0.,  1.,  2.], [ 3.,  4.,  5.]` and `[ 6.,  7.]`.

    Note that the answer for `num_segments = 1` is a trivial "1" but we handle this scenario
    in case somebody calls it. Sum of the ratios should be 1.0.

    Update:
        Function updated to produce plots and log output. 

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"num_segments": N, "segment_focus": i} with N, i both ints
    :return: the feature values
    :return type: list of tuples (index, data)
    """
    res_data = []
    res_index = []
    full_series_energy = np.sum(x**2)

    for parameter_combination in param:
        num_segments = parameter_combination["num_segments"]
        segment_focus = parameter_combination["segment_focus"]
        assert segment_focus < num_segments
        assert num_segments > 0

        if full_series_energy == 0:
            energy_ratio = res_data.append(np.nan)
        else:
            energy_ratio = np.sum(np.array_split(x, num_segments)[segment_focus] ** 2.0) / full_series_energy

        res_data.append(energy_ratio)

        # Debug data
        if _logging_dict is not None:
            if 'logger' in _logging_dict:
                logger = _logging_dict['logger']
                logger.debug(
                    f'Feature: energy_ratio_by_chunks with params {parameter_combination}, full_series_energy={full_series_energy}, segment values =  {np.array_split(x, num_segments)[segment_focus]}, segment sum squared: {np.sum(np.array_split(x, num_segments)[segment_focus] ** 2.0)}'
                )

            if 'self_instance' in _logging_dict:
                self_instance = _logging_dict['self_instance']
                if 'cycle_id' in _logging_dict:
                    cycle_id = _logging_dict['cycle_id']
                else:
                    cycle_id = len(x)

                label = _logging_dict.get('label', None)
                fig = plot_energy_ratio_by_chunks(x, num_segments, segment_focus, energy_ratio, label=label)
                self_instance._save_fig_to_log(fig, f'{cycle_id}_energy_ratio_by_chunks_{_dict_to_filename(parameter_combination)}_{label}.png')

        res_index.append(
            "num_segments_{}__segment_focus_{}".format(num_segments, segment_focus)
        )

    # Materialize as list for Python 3 compatibility with name handling
    return zip(res_index, res_data)


def fft_aggregated(x, params, _logging_dict=None):
    """
    Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"aggtype": s} where s str and in ["centroid", "variance",
        "skew", "kurtosis"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    assert {config["aggtype"] for config in params} <= {
        "centroid",
        "variance",
        "skew",
        "kurtosis",
    }, 'Attribute must be "centroid", "variance", "skew", "kurtosis"'

    def get_moment(y, moment):
        """
        Returns the (non centered) moment of the distribution y:
        E[y**moment] = \\sum_i[index(y_i)^moment * y_i] / \\sum_i[y_i]

        :param y: the discrete distribution from which one wants to calculate the moment
        :type y: pandas.Series or np.array
        :param moment: the moment one wants to calcalate (choose 1,2,3, ... )
        :type moment: int
        :return: the moment requested
        :return type: float
        """
        return y.dot(np.arange(len(y), dtype=float) ** moment) / y.sum()

    def get_centroid(y):
        """
        :param y: the discrete distribution from which one wants to calculate the centroid
        :type y: pandas.Series or np.array
        :return: the centroid of distribution y (aka distribution mean, first moment)
        :return type: float
        """
        return get_moment(y, 1)

    def get_variance(y):
        """
        :param y: the discrete distribution from which one wants to calculate the variance
        :type y: pandas.Series or np.array
        :return: the variance of distribution y
        :return type: float
        """
        return get_moment(y, 2) - get_centroid(y) ** 2

    def get_skew(y):
        """
        Calculates the skew as the third standardized moment.
        Ref: https://en.wikipedia.org/wiki/Skewness#Definition

        :param y: the discrete distribution from which one wants to calculate the skew
        :type y: pandas.Series or np.array
        :return: the skew of distribution y
        :return type: float
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, skew should be 0 and variance 0.  However, in the discrete limit,
        # the skew blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                get_moment(y, 3) - 3 * get_centroid(y) * variance - get_centroid(y) ** 3
            ) / get_variance(y) ** (1.5)

    def get_kurtosis(y):
        """
        Calculates the kurtosis as the fourth standardized moment.
        Ref: https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments

        :param y: the discrete distribution from which one wants to calculate the kurtosis
        :type y: pandas.Series or np.array
        :return: the kurtosis of distribution y
        :return type: float
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, kurtosis should be 3 and variance 0.  However, in the discrete limit,
        # the kurtosis blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                get_moment(y, 4)
                - 4 * get_centroid(y) * get_moment(y, 3)
                + 6 * get_moment(y, 2) * get_centroid(y) ** 2
                - 3 * get_centroid(y)
            ) / get_variance(y) ** 2

    calculation = dict(
        centroid=get_centroid,
        variance=get_variance,
        skew=get_skew,
        kurtosis=get_kurtosis,
    )

    fft_abs = np.abs(np.fft.rfft(x))

    res = [calculation[config["aggtype"]](fft_abs) for config in params]
    index = ['aggtype_"{}"'.format(config["aggtype"]) for config in params]

    feature_value = zip(index, res)

    # Debug data
    if _logging_dict is not None:
        for param in params:

            agg = param['aggtype']
            if 'logger' in _logging_dict:
                logger = _logging_dict['logger']
                # using the index from index, get the feature value from res
                feature_value_for_agg = [fv for idx, fv in feature_value if idx == f'aggtype_"{agg}"']
                logger.debug(
                    f'Feature: fft_aggregated with params {param}, feature_value={feature_value_for_agg}.'
                )

            if 'self_instance' in _logging_dict:
                self_instance = _logging_dict['self_instance']
                if 'cycle_id' in _logging_dict:
                    cycle_id = _logging_dict['cycle_id']
                else:
                    cycle_id = len(x)

            label = _logging_dict.get('label', None)
            fig = plot_fft_aggregated(x, agg, label=label)
            self_instance._save_fig_to_log(fig, f'{cycle_id}_fft_aggregated__{agg}_{label}.png')

    return zip(index, res)


def fft_coefficient(x, param, _logging_dict=None):
    """
    Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast
    fourier transformation algorithm

    .. math::
        A_k =  \\sum_{m=0}^{n-1} a_m \\exp \\left \\{ -2 \\pi i \\frac{m k}{n} \\right \\}, \\qquad k = 0,
        \\ldots , n-1.

    The resulting coefficients will be complex, this feature calculator can return the real part (attr=="real"),
    the imaginary part (attr=="imag), the absolute value (attr=""abs) and the angle in degrees (attr=="angle).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"coeff": x, "attr": s} with x int and x >= 0, s str and in ["real", "imag",
        "abs", "angle"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    assert (
        min([config["coeff"] for config in param]) >= 0
    ), "Coefficients must be positive or zero."
    assert {config["attr"] for config in param} <= {
        "imag",
        "real",
        "abs",
        "angle",
    }, 'Attribute must be "real", "imag", "angle" or "abs"'

    fft = np.fft.rfft(x)

    def complex_agg(x, agg):
        if agg == "real":
            return x.real
        elif agg == "imag":
            return x.imag
        elif agg == "abs":
            return np.abs(x)
        elif agg == "angle":
            return np.angle(x, deg=True)

    res = [
        complex_agg(fft[config["coeff"]], config["attr"])
        if config["coeff"] < len(fft)
        else np.nan
        for config in param
    ]
    index = [
        'attr_"{}"__coeff_{}'.format(config["attr"], config["coeff"])
        for config in param
    ]

    # Debug data
    if _logging_dict is not None:
        if 'logger' in _logging_dict:
            logger = _logging_dict['logger']
            # logger.debug(
            #     f'Feature: _mean_second_derivative_central: {MSDC} .'
            # )

        if 'self_instance' in _logging_dict:
            self_instance = _logging_dict['self_instance']
            if 'cycle_id' in _logging_dict:
                cycle_id = _logging_dict['cycle_id']
            else:
                cycle_id = len(x)

            label = _logging_dict.get('label', None)
            fig = plot_fft_coefficient(x, param, delta_t=0.005, label=label)
            self_instance._save_fig_to_log(fig, f'{cycle_id}_fft_coefficient_{label}.png')

    return zip(index, res)


def mean_second_derivative_central(x, param=None, _logging_dict=None):
    """
    Returns the mean value of a central approximation of the second derivative

    .. math::

        \\frac{1}{2(n-2)} \\sum_{i=1,\\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)

    MSDC = (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.nan

    # Debug data
    if _logging_dict is not None:
        if 'logger' in _logging_dict:
            logger = _logging_dict['logger']
            logger.debug(
                f'Feature: _mean_second_derivative_central: {MSDC} .'
            )

        if 'self_instance' in _logging_dict:
            self_instance = _logging_dict['self_instance']
            if 'cycle_id' in _logging_dict:
                cycle_id = _logging_dict['cycle_id']
            else:
                cycle_id = len(x)

            label = _logging_dict.get('label', None)
            fig = plot_mean_second_derivative_central(x, label=label)
            self_instance._save_fig_to_log(fig, f'{cycle_id}_mean_second_derivative_central_{label}.png')

    return MSDC


def number_cwt_peaks(x, n_values, _logging_dict=None):
    """
    Calculates the number of CWT peaks for the specified n values.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param n_values: list of maximum width values to consider
    :type n_values: list
    :return: array of number of CWT peaks for each n value
    :return type: numpy.ndarray
    """
    cwt_peaks_counts = []
    for n in n_values:
        # Use PyWavelets' cwt with the 'mexh' (Mexican hat) wavelet
        cwtmatr, _ = pywt.cwt(x, scales=np.arange(1, n + 1), wavelet='mexh')
        # Find peaks in the absolute value of the cwt coefficients at the last scale
        peaks = find_peaks_cwt(np.abs(cwtmatr[-1]), widths=np.array([1]))
        cwt_peaks_counts.append(len(peaks))

    return np.array(cwt_peaks_counts)


def number_peaks(x, n_values, _logging_dict=None):
    """
    Calculates the number of peaks for the specified support values.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param n_values: list of support values for peaks
    :type n_values: list
    :return: array of number of peaks for each support value
    :return type: numpy.ndarray
    """
    x_reduced = x[max(n_values): -max(n_values)]

    peaks_counts = []
    for n in n_values:
        res = None
        for i in range(1, n + 1):
            result_first = x_reduced > _roll(x, i)[max(n_values): -max(n_values)]

            if res is None:
                res = result_first
            else:
                res &= result_first

            res &= x_reduced > _roll(x, -i)[max(n_values): -max(n_values)]
        peaks_counts.append(np.sum(res))

    return np.array(peaks_counts)


def partial_autocorrelation(x, param, _logging_dict=None):
    """
    Calculates the value of the partial autocorrelation function at the given lag.

    The lag `k` partial autocorrelation of a time series :math:`\\lbrace x_t, t = 1 \\ldots T \\rbrace` equals the
    partial correlation of :math:`x_t` and :math:`x_{t-k}`, adjusted for the intermediate variables
    :math:`\\lbrace x_{t-1}, \\ldots, x_{t-k+1} \\rbrace` ([1]).

    Following [2], it can be defined as

    .. math::

        \\alpha_k = \\frac{ Cov(x_t, x_{t-k} | x_{t-1}, \\ldots, x_{t-k+1})}
        {\\sqrt{ Var(x_t | x_{t-1}, \\ldots, x_{t-k+1}) Var(x_{t-k} | x_{t-1}, \\ldots, x_{t-k+1} )}}

    with (a) :math:`x_t = f(x_{t-1}, \\ldots, x_{t-k+1})` and (b) :math:`x_{t-k} = f(x_{t-1}, \\ldots, x_{t-k+1})`
    being AR(k-1) models that can be fitted by OLS. Be aware that in (a), the regression is done on past values to
    predict :math:`x_t` whereas in (b), future values are used to calculate the past value :math:`x_{t-k}`.
    It is said in [1] that "for an AR(p), the partial autocorrelations [ :math:`\\alpha_k` ] will be nonzero for `k<=p`
    and zero for `k>p`."
    With this property, it is used to determine the lag of an AR-Process.

    .. rubric:: References

    |  [1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
    |  Time series analysis: forecasting and control. John Wiley & Sons.
    |  [2] https://onlinecourses.science.psu.edu/stat510/node/62

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"lag": val} with int val indicating the lag to be returned
    :type param: list
    :return: the value of this feature
    :return type: float
    """
    # Check the difference between demanded lags by param and possible lags to calculate (depends on len(x))
    max_demanded_lag = max([lag["lag"] for lag in param])
    n = len(x)

    # Check if list is too short to make calculations
    if n <= 1:
        pacf_coeffs = [np.nan] * (max_demanded_lag + 1)
    else:
        # https://github.com/statsmodels/statsmodels/pull/6846
        # PACF limits lag length to 50% of sample size.
        if max_demanded_lag >= n // 2:
            max_lag = n // 2 - 1
        else:
            max_lag = max_demanded_lag
        if max_lag > 0:
            pacf_coeffs = list(pacf(x, method="ld", nlags=max_lag))
            pacf_coeffs = pacf_coeffs + [np.nan] * max(0, (max_demanded_lag - max_lag))
        else:
            pacf_coeffs = [np.nan] * (max_demanded_lag + 1)

    return [("lag_{}".format(lag["lag"]), pacf_coeffs[lag["lag"]]) for lag in param]


def percentage_of_reoccurring_values_to_all_values(x, param=None, _logging_dict=None):
    """
    Returns the percentage of values that are present in the time series
    more than once.

        len(different values occurring more than once) / len(different values)

    This means the percentage is normalized to the number of unique values,
    in contrast to the percentage_of_reoccurring_datapoints_to_all_datapoints.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if len(x) == 0:
        return np.nan

    unique, counts = np.unique(x, return_counts=True)

    if counts.shape[0] == 0:
        return 0

    return np.sum(counts > 1) / float(counts.shape[0])


def quantile(x, q_list, _logging_dict=None):
    """
    Calculates the quantiles for the specified quantile values.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param q_list: list of quantile values to calculate
    :type q_list: list
    :return: array of quantile values for each quantile value in q_list
    :return type: numpy.ndarray
    """
    quantile_values = []
    for q in q_list:
        if len(x) == 0:
            quantile_values.append(np.nan)
        else:
            quantile_value = np.quantile(x, q)
            quantile_values.append(quantile_value)

    return np.array(quantile_values)


def ratio_value_number_to_time_series_length(x, param=None, _logging_dict=None):
    """
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case.
    In principle, it just returns

        # unique values / # values

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if x.size == 0:
        return np.nan

    return np.unique(x).size / x.size


def sample_entropy(x, param=None, _logging_dict=None):
    """
    Calculate and return sample entropy of x.

    .. rubric:: References

    |  [1] http://en.wikipedia.org/wiki/Sample_Entropy
    |  [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray

    :return: the value of this feature
    :return type: float
    """
    x = np.array(x)

    # if one of the values is NaN, we can not compute anything meaningful
    if np.isnan(x).any():
        return np.nan

    m = 2  # common value for m, according to wikipedia...
    tolerance = 0.2 * np.std(
        x
    )  # 0.2 is a common value for r, according to wikipedia...

    # Split time series and save all templates of length m
    # Basically we turn [1, 2, 3, 4] into [1, 2], [2, 3], [3, 4]
    xm = _into_subchunks(x, m)

    # Now calculate the maximum distance between each of those pairs
    #   np.abs(xmi - xm).max(axis=1)
    # and check how many are below the tolerance.
    # For speed reasons, we are not doing this in a nested for loop,
    # but with numpy magic.
    # Example:
    # if x = [1, 2, 3]
    # then xm = [[1, 2], [2, 3]]
    # so we will substract xm from [1, 2] => [[0, 0], [-1, -1]]
    # and from [2, 3] => [[1, 1], [0, 0]]
    # taking the abs and max gives us:
    # [0, 1] and [1, 0]
    # as the diagonal elements are always 0, we substract 1.
    B = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= tolerance) - 1 for xmi in xm])

    # Similar for computing A
    xmp1 = _into_subchunks(x, m + 1)

    A = np.sum(
        [np.sum(np.abs(xmi - xmp1).max(axis=1) <= tolerance) - 1 for xmi in xmp1]
    )

    # Return SampEn
    return -np.log(A / B)


def spkt_welch_density(x, param, _logging_dict=None):
    """
    This feature calculator estimates the cross power spectral density of the time series x at different frequencies.
    To do so, the time series is first shifted from the time domain to the frequency domain.

    The feature calculators returns the power spectrum of the different frequencies.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"coeff": x} with x int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    freq, pxx = welch(x, nperseg=min(len(x), 256))
    coeff = [config["coeff"] for config in param]
    indices = ["coeff_{}".format(i) for i in coeff]

    if len(pxx) <= np.max(
        coeff
    ):  # There are fewer data points in the time series than requested coefficients

        # filter coefficients that are not contained in pxx
        reduced_coeff = [coefficient for coefficient in coeff if len(pxx) > coefficient]
        not_calculated_coefficients = [
            coefficient for coefficient in coeff if coefficient not in reduced_coeff
        ]

        # Fill up the rest of the requested coefficients with np.nan
        return zip(
            indices,
            list(pxx[reduced_coeff]) + [np.nan] * len(not_calculated_coefficients),
        )
    else:
        return zip(indices, pxx[coeff])
