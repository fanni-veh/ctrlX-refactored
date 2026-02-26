import pywt
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

matplotlib.use('Agg')  # or 'Qt5Agg', 'WebAgg'


def plot_fisher_score(ft_df, fisher_scores, feature_range):
    """
    Plots the Fisher score distributions for a specified range of features.

    Args:
        ft_df: DataFrame of features (must include 'classification' column).
        fisher_scores: Dict of {feature_name: fisher_score}.
        feature_range: Tuple (start_idx, end_idx) for feature indices to plot.
    """
    start_idx, end_idx = feature_range
    sorted_features = sorted(fisher_scores.items(), key=lambda x: -x[1])
    features_to_plot = sorted_features[start_idx:end_idx]
    num_features = len(features_to_plot)

    ncols = 2 if num_features > 1 else 1
    nrows = int(math.ceil(num_features / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows))
    axs = np.array(axs).flatten()

    fig.suptitle(f'Fisher Score Distributions (Features {start_idx + 1} to {end_idx})', fontsize=16)

    for i, (feature, score) in enumerate(features_to_plot):
        data_0 = ft_df[feature][ft_df['classification'] == 0]
        data_1 = ft_df[feature][ft_df['classification'] == 1]
        axs[i].hist([data_0, data_1], bins=20, color=['red', 'blue'], label=['Bad Signals', 'Good Signals'], alpha=0.8, stacked=False)
        axs[i].set_title(f'{feature} (Fisher: {score:.2f})')
        axs[i].legend()

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_feature_similarity_heatmap(features_similarity, highlight_features=None, fisher_scores=None):
    """
    Plots a heatmap of the 2D similarity array from the features_similarity dictionary,
    showing data only in the strict lower triangle of the matrix, with values annotated.
    Optionally highlights specified features on the axes and highlights the values between all pairs combinations.
    If fisher_scores is provided, appends the score in brackets to each feature name.

    Parameters
    ----------
    features_similarity : dict
        Nested dictionary of feature similarities (features_similarity[feature1][feature2] = similarity).
    highlight_features : list or None
        List of feature names to highlight on the axes. If None, no highlighting.
    fisher_scores : dict or None
        Dictionary mapping feature names to Fisher scores. If provided, feature labels will include scores.
    """
    features = sorted(features_similarity.keys())
    if fisher_scores is not None:
        # Collapse fisher_scores to base feature names (before '__'), keeping the highest score for each
        fisher_scores_collapsed = {}
        for k, v in fisher_scores.items():
            base = k.split("__")[0]
            if base not in fisher_scores_collapsed or v > fisher_scores_collapsed[base]:
                fisher_scores_collapsed[base] = v
        # Sort features by fisher_score (descending), fallback to original order if not present
        features_sorted = sorted(
            features,
            key=lambda f: fisher_scores_collapsed.get(f, float('-inf')),
            reverse=True
        )
        feature_labels = [
            f"{f} ({fisher_scores_collapsed[f]:.2f})" if f in fisher_scores_collapsed else f
            for f in features_sorted
        ]
        features = features_sorted
    else:
        feature_labels = features

    sim_matrix = np.array([[features_similarity[f1][f2] for f2 in features] for f1 in features])

    # Mask the upper triangle and diagonal
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=0)
    sim_matrix_masked = np.ma.array(sim_matrix, mask=mask)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_matrix_masked, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
    fig.colorbar(im, label='Similarity', ax=ax)
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(feature_labels, rotation=90)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(feature_labels)
    ax.set_title('Feature Similarity Heatmap (max score for each feature group)')
    fig.tight_layout()

    highlight_set = set(f.split("__")[0] for f in highlight_features) if highlight_features is not None else set()

    # Annotate values in the strict lower triangle (i > j)
    for i in range(len(features)):
        for j in range(len(features)):
            if i > j:
                value = sim_matrix[i, j]
                if features[i] in highlight_set and features[j] in highlight_set:
                    color = "black"
                    fontweight = "bold"
                    alpha = 1.0
                else:
                    color = "black"
                    fontweight = "normal"
                    alpha = 0.3
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    color=color, fontsize=8, fontweight=fontweight, alpha=alpha
                )

    # Set axis label colors and weights
    for idx, feat in enumerate(features):
        if feat in highlight_set:
            ax.get_xticklabels()[idx].set_color('black')
            ax.get_yticklabels()[idx].set_color('black')
            ax.get_xticklabels()[idx].set_fontweight('bold')
            ax.get_yticklabels()[idx].set_fontweight('bold')
            ax.get_xticklabels()[idx].set_alpha(1.0)
            ax.get_yticklabels()[idx].set_alpha(1.0)
        else:
            ax.get_xticklabels()[idx].set_color('black')
            ax.get_yticklabels()[idx].set_color('black')
            ax.get_xticklabels()[idx].set_fontweight('normal')
            ax.get_yticklabels()[idx].set_fontweight('normal')
            ax.get_xticklabels()[idx].set_alpha(0.3)
            ax.get_yticklabels()[idx].set_alpha(0.3)

    return fig


def plot_absolute_sum_of_changes(signal, label=None):
    signal = np.asarray(signal)
    diffs = np.diff(signal, prepend=signal[0])
    abs_changes = np.abs(diffs)
    cumsum_abs_changes = np.cumsum(abs_changes)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    # First axis: signal and change
    ax1.plot(signal, label='Signal', color='blue', linewidth=1.5)
    ax1.plot(abs_changes, label='Abs Change', color='orange', linestyle='--', linewidth=1)
    ax1.set_ylabel("Signal / Change")
    ax1.set_xlabel("Index")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Second axis: cumulative sum of abs changes
    ax2 = ax1.twinx()
    ax2.plot(cumsum_abs_changes, label='Cumulative Abs Change', color='green', linewidth=2)
    ax2.set_ylabel("Cumulative Abs Change")
    ax2.legend(loc="upper right")

    plt.title(f"Absolute Sum of Changes: {cumsum_abs_changes[-1]:.2f}")
    plt.tight_layout()
    return fig


def plot_agg_autocorrelation(signal, param, label=None):
    """
    Plot multiple autocorrelation vs lag plots with different agg_autocorrelation parameters.

    Args:
        signal (array-like): 1D time series input.
        param (list of dict): List of {'f_agg': <agg_func>, 'maxlag': <int>} dictionaries.

    Returns:
        matplotlib.figure.Figure: Figure with subplots for each agg_autocorrelation config.
    """
    signal = np.asarray(signal)
    num_plots = len(param)
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots), sharex=False)

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    if num_plots == 1:
        axes = [axes]  # Make iterable if only one plot

    for ax, config in zip(axes, param):
        f_agg = config.get('f_agg', 'mean')
        maxlag = config.get('maxlag', 40)

        acf_vals = acf(signal, nlags=maxlag, fft=True)

        # Aggregate computation (excluding lag 0)
        acf_lags = acf_vals[1:]
        if f_agg == 'mean':
            agg_value = np.mean(acf_lags)
        elif f_agg == 'median':
            agg_value = np.median(acf_lags)
        elif f_agg == 'var':
            agg_value = np.var(acf_lags)
        else:
            raise ValueError(f"Unsupported aggregation: {f_agg}")

        ax.stem(range(len(acf_vals)), acf_vals)
        ax.axhline(agg_value, color='red', linestyle='--', label=f'{f_agg} = {agg_value:.3f}')
        ax.set_title(f'Autocorrelation with f_agg="{f_agg}", maxlag={maxlag}')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the suptitle

    fig.suptitle('Agg_autocorrelation', fontsize=16)

    return fig


def plot_agg_linear_trend(signal, param, label=None):
    """
    Plot agg_linear_trend regression fits and per-chunk attribute bars with aggregation line.

    Args:
        signal (array-like): 1D time series.
        param (dict): {'attr': ..., 'chunk_len': ..., 'f_agg': ...}

    Returns:
        matplotlib.figure.Figure
    """
    signal = np.asarray(signal)
    chunk_len = param['chunk_len']
    attr = param['attr']
    f_agg = param['f_agg']

    x_full = np.arange(len(signal))
    fig, ax_signal = plt.subplots(figsize=(12, 6))

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    # Plot signal
    ax_signal.plot(x_full, signal, label='Signal', color='lightgray', zorder=1)

    values = []
    chunk_centers = []

    for i in range(0, len(signal) - chunk_len + 1, chunk_len):
        x = np.arange(chunk_len)
        y = signal[i:i + chunk_len]
        reg = linregress(x, y)

        # Extract attribute
        val = getattr(reg, attr)
        values.append(val)
        chunk_centers.append(i + chunk_len // 2)

        # Plot regression line
        y_pred = reg.slope * x + reg.intercept
        ax_signal.plot(x + i, y_pred, color='blue', linewidth=2, alpha=0.6)

    # Compute aggregation
    agg = {
        'mean': np.mean,
        'var': np.var,
        'min': np.min,
        'max': np.max
    }.get(f_agg)

    if agg is None:
        raise ValueError(f"Unsupported f_agg: {f_agg}")

    agg_value = agg(values)

    # Add twin axis for attribute values
    ax_attr = ax_signal.twinx()
    bar_container = ax_attr.bar(chunk_centers, values, width=chunk_len * 0.8, alpha=0.3,
                                label=f'Per-chunk {attr}', color='orange', zorder=0)
    ax_attr.axhline(agg_value, color='red', linestyle='--',
                    label=f'{f_agg}({attr}) = {agg_value:.3f}')

    # Labels and legends
    ax_signal.set_title(f'agg_linear_trend: {attr}, chunk_len={chunk_len}, agg={f_agg}: {agg_value:.3f}')
    ax_signal.set_xlabel('Time Index')
    ax_signal.set_ylabel('Signal')
    ax_attr.set_ylabel(attr)

    ax_signal.grid(True)
    ax_signal.legend(loc='upper left')
    ax_attr.legend(loc='upper right')

    fig.tight_layout()
    return fig


def plot_autocorrelation(signal, lag, label=None):
    signal = np.asarray(signal)
    n = len(signal)
    x_mean = np.mean(signal)
    v = np.var(signal)

    # Compute centered values and product
    y1 = signal[: n - lag]
    y2 = signal[lag:]
    centered_y1 = y1 - x_mean
    centered_y2 = y2 - x_mean
    product = centered_y1 * centered_y2

    # Autocorrelation value
    autocorr_values = product / v
    autocorr_value = np.sum(autocorr_values) / (n - lag)

    # Normalize product for colormap (divergent from 0)
    norm_autocorr_values = mcolors.TwoSlopeNorm(vmin=np.min(product), vcenter=0, vmax=np.max(autocorr_values))
    cmap = plt.get_cmap('coolwarm')

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    # Top plot: centered_y1 and centered_y2
    axs[0].plot(centered_y1, label='y1 = x - mean', color='blue', linewidth=0.5)
    axs[0].plot(centered_y2, label='y2 = x_lag - mean', color='orange',  linewidth=0.5)
    axs[0].set_ylabel('Mean-Centered Signals')
    axs[0].set_title(f'Mean-Centered Signals (lag={lag})')
    axs[0].legend()
    axs[0].grid(True)

    # Bottom plot: product with colored background
    ax1 = axs[1]
    for i, val in enumerate(product):
        ax1.axvspan(i - 0.5, i + 0.5, color=cmap(norm_autocorr_values(val)), alpha=0.5)

    sm = ScalarMappable(cmap=cmap, norm=norm_autocorr_values)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, pad=0.15, aspect=30, fraction=0.03, alpha=0.8)
    cbar.set_label("Autocorrelation")
    cbar.set_ticks([norm_autocorr_values.vmin, 0, norm_autocorr_values.vmax])
    cbar.set_ticklabels(["-1", "0", "1"])

    ax1.plot(product, label='y1 * y2', color='purple', linewidth=1.5)
    ax1.set_ylabel('Elementwise Product')
    ax1.set_title(f'Autocorrelation')

    # Second Y-axis for autocorr value
    ax2 = ax1.twinx()
    ax2.axhline(autocorr_value, color='cyan', linestyle='--', linewidth=1, label=f'Autocorr = {autocorr_value:.3f}')
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel('Autocorrelation')
    ax2.legend(loc='upper right')

    ax1.set_xlabel('Index')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the suptitle

    fig.suptitle(f'Autocorrelation: {autocorr_value:.3f}', fontsize=16)

    return fig


def plot_binned_entropy(signal, num_bins=10, label=None):
    """
    Plot histogram of the signal and show per-bin entropy contributions.

    Args:
        signal (array-like): 1D time series.
        num_bins (int): Number of bins for histogram.

    Returns:
        matplotlib.figure.Figure
    """
    signal = np.asarray(signal)

    # Compute histogram (counts and bin edges)
    counts, bin_edges = np.histogram(signal, bins=num_bins)
    probs = counts / signal.size

    # Compute entropy contributions safely (ignore log(0) issues)
    entropies = np.zeros_like(probs, dtype=float)
    mask = probs > 0
    entropies[mask] = -probs[mask] * np.log(probs[mask])
    total_entropy = np.sum(entropies)

    # Compute bin centers for plotting entropies
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    # Histogram
    ax1.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]),
            color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')
    ax1.set_xlabel('Signal Value')
    ax1.set_ylabel('Count', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # Twin axis for entropy
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, entropies, 'o-', color='darkred', linewidth=2, label='Per-bin Entropy')
    ax2.set_ylabel('Entropy Contribution', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')

    # Title
    ax1.set_title(f'Binned Entropy (bins={num_bins}) = {total_entropy:.4f}')
    ax1.grid(True)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    return fig


def plot_c3(signal, lag, smoothing_window=10, label=None):
    signal = np.asarray(signal)
    n = len(signal)

    if n <= 2 * lag:
        raise ValueError("Time series too short for the specified lag.")

    base = signal[:-2*lag]
    lag1 = signal[lag:-lag]
    lag2 = signal[2*lag:]

    product = base * lag1 * lag2
    abs_product = np.abs(product - np.mean(product))

    smoothed = np.convolve(abs_product, np.ones(smoothing_window) / smoothing_window, mode='same')

    norm = Normalize(vmin=np.min(smoothed), vmax=np.max(smoothed))
    cmap = plt.get_cmap('RdYlGn_r')

    t = np.arange(len(base))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Default height restored

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    # Plot 1: Signals with colored background
    for i in range(len(t)):
        color = cmap(norm(smoothed[i]))
        axs[0].axvspan(t[i] - 0.5, t[i] + 0.5, color=color, alpha=0.1)

    axs[0].plot(t, base, label='signal[t]', linewidth=1, color='blue')
    axs[0].plot(t, lag1, label=f'signal[t + {lag}]', linewidth=0.7, color='white')
    axs[0].plot(t, lag2, label=f'signal[t + {2*lag}]', linewidth=0.5, color='black')
    axs[0].set_title(f"Signal and Lagged Versions (lag = {lag}) with Correlation Background")
    axs[0].set_xlabel("Time Index")
    axs[0].set_ylabel("Value")
    axs[0].legend()
    axs[0].grid(True)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs[0], pad=0.01, aspect=30)
    cbar.set_label("Triple Product Strength")
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels(["Low", "High"])

    # Plot 2: Triple product and mean
    mean_product = np.mean(product)
    axs[1].plot(t, product, label='signal[t]*signal[t+lag]*signal[t+2*lag]', color='black', linewidth=0.5)
    axs[1].axhline(mean_product, color='red', linestyle='--', label=f'Mean = {mean_product:.4f}')
    axs[1].set_title("Triple Product and Mean")
    axs[1].set_xlabel("Time Index")
    axs[1].set_ylabel("Product Value")
    axs[1].legend()
    axs[1].grid(True)

    fig.tight_layout()
    return fig


def plot_change_quantiles(signal, params, bin_cat_0, div, ind_inside_corridor, label=None):
    ql = params.get('ql', 0)
    qh = params.get('qh', 1)
    isabs = params.get('isabs', False)
    f_agg = params.get('f_agg', 'mean')

    aggregator = getattr(np, f_agg)
    agg_value = aggregator(div[ind_inside_corridor]) if np.any(ind_inside_corridor) else np.nan

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    # Top plot: Original signal ---
    axs[0].plot(signal, label='Signal', color='black')
    axs[0].scatter(np.where(bin_cat_0)[0], signal[bin_cat_0],
                   color='orange', label='Quantile Range Points', zorder=3, s=2)
    axs[0].set_title(f"Signal with Quantile Range Points ({ql:.2f} to {qh:.2f})")
    axs[0].legend()

    # Bottom plot: div with aggregation ---
    ax2 = axs[1]
    x_div = np.arange(1, len(signal))
    ax2.plot(x_div, div, label='Change (div)', color='blue')

    # Mark points used in aggregation
    ax2.scatter(x_div[ind_inside_corridor], div[ind_inside_corridor],
                color='red', label='Included in Aggregation', zorder=3, s=2)

    # Aggregation line
    ax3 = ax2.twinx()
    agg_line = np.full_like(div, agg_value)
    ax3.plot(x_div, agg_line, color='green', linestyle='--',
             label=f'Aggregated ({f_agg})')

    # Labels, legends, titles
    ax2.set_ylabel("Difference")
    ax3.set_ylabel("Aggregated Value")
    abs_text = "Absolute" if isabs else "Signed"
    axs[1].set_title(f"{f_agg.capitalize()} of {abs_text} Changes in Quantile Corridor")
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper right')

    return fig


def plot_cid_ce(signal, normalize=True, label=None):
    signal = np.asarray(signal)

    if normalize:
        signal = (signal - np.mean(signal)) / np.std(signal)

    diff = np.diff(signal)
    squared_diffs = diff ** 2
    cumulative = np.cumsum(squared_diffs)
    cid_ce_val = np.sqrt(cumulative[-1])

    t = np.arange(len(signal))
    t_diff = np.arange(len(diff))

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    # Plot 1: Original signal
    axs[0].plot(t, signal, color='steelblue', linewidth=1.5)
    axs[0].set_title(f"Original Signal (cid_ce = {cid_ce_val:.4f})")
    axs[0].set_ylabel("Value")
    axs[0].grid(True)

    # Plot 2: Squared diffs (left) and cumulative sum (right)
    ax2 = axs[1]
    ax2_bar = ax2.twinx()

    bars = ax2.bar(t_diff, squared_diffs, color='lightcoral', label='(Δx)²')
    line = ax2_bar.plot(t_diff, cumulative, color='darkred', linewidth=2, label='Cumulative Sum')

    ax2.set_ylabel("(Δx)²", color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='lightcoral')

    ax2_bar.set_ylabel("Cumulative Sum", color='darkred')
    ax2_bar.tick_params(axis='y', labelcolor='darkred')

    # Final horizontal reference line and annotation
    ax2_bar.axhline(y=cumulative[-1], color='gray', linestyle='--', linewidth=1)
    ax2_bar.text(t_diff[-1], cumulative[-1], f' cid_ce² = {cumulative[-1]:.2f}',
                 va='bottom', ha='right', fontsize=10, color='gray')

    ax2.set_xlabel("Time Index")
    axs[1].set_title("Squared Differences and Cumulative cid_ce²")
    axs[1].grid(True)

    fig.tight_layout()
    return fig


def plot_cwt_coefficients(signal, param, delta_t=0.005, label=None):
    """
    Plots a CWT heatmap with frequency on the Y-axis.

    Parameters:
        signal (array-like): Input signal.
        parameter_combination (dict): Contains 'wavelet', 'level', 'coeff'.
        delta_t (float): Sampling interval in seconds (e.g., 0.005s).

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
    """

    widths = sorted({w for parameter_combination in param for w in parameter_combination.get("widths", (1,))})
    wavelet = param[0].get("wavelet", "mexh")
    scales = np.arange(1, max(widths)+1)

    wavelet = 'cmor1.5-1.0'  # CB testing

    # Perform CWT
    cwtmatr, frequencies = pywt.cwt(signal, scales=scales, wavelet=wavelet, sampling_period=delta_t)
    # cwt_power = (cwtmatr) / np.max(cwtmatr)  # Shape: [n_scales, n_times]
    cwt_power = np.abs(cwtmatr) ** 2  # CB testing
    cwt_power = np.log10(cwt_power)  # CB testing

    # Convert scales to frequencies
    fc = pywt.central_frequency(wavelet)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(
        cwt_power,
        extent=[0, len(signal) * delta_t, frequencies[-1], frequencies[0]],  # frequency decreases with scale
        cmap="coolwarm",
        aspect="auto"
    )

    # Overlay circles for each (width, coeff) pair
    # for p in param:
    #     scale = p.get("w", 1)-1  # (index is 1 less than returned value)

    #     time_idx = int(len(signal)/100*p.get("coeff", 0))  # temporary, until coeffs are redifined
    #     frequency = fc / (scale * delta_t)
    #     time = time_idx * delta_t

    #     ax.plot(time,
    #             frequency,
    #             'o',
    #             markersize=8,
    #             markeredgecolor='black',
    #             markeredgewidth=1.0,
    #             markerfacecolor='none',
    #             alpha=0.6,
    #             label=f"{cwtmatr[scale, time_idx]:.2f}")
    # ax.text(time + 0.1, frequency, f"{cwtmatr[scale, time_idx]:.2f}", fontsize=8, color='black', va='center')

    ax.set_title(f"CWT Heatmap (wavelet={wavelet})")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    # ax.legend(loc=-1)
    fig.colorbar(im, ax=ax, label="Wavelet Power Response")

    return fig


def plot_energy_ratio_by_chunks(signal, num_segments, segment_focus, energy_ratio, label=None):
    segment_len = len(signal) // num_segments

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    # Plot signal on primary y-axis
    ax1.plot(signal, label="Signal (mA)", color="tab:blue")
    ax1.set_xlabel("Time Index")
    ax1.set_ylabel("Current (mA)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # Draw segment boundaries on ax1
    for i in range(1, num_segments):
        ax1.axvline(i * segment_len, color="gray", linestyle="--", alpha=0.5)

    # Create secondary y-axis for energy ratio
    ax2 = ax1.twinx()
    ax2.set_ylabel("Energy Ratio", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")
    ax2.set_ylim(0, 1)

    # Plot energy ratio bar centered on segment_focus
    bar_center = (segment_focus + 0.5) * segment_len
    bar_width = segment_len * 0.8
    ax2.bar(bar_center, energy_ratio, width=bar_width, color='tab:orange', alpha=0.6,
            align='center', label=f"Energy Ratio (segment {segment_focus})")

    # Annotate the energy ratio value
    ax2.text(bar_center, energy_ratio + 0.03, f"{energy_ratio:.2f}",
             ha='center', va='bottom', color='tab:orange', fontsize=10)

    # Set title and tighten layout
    ax1.set_title(f"Signal and Energy Ratio (Segment {segment_focus})")
    fig.tight_layout()

    return fig


def plot_fft_aggregated(signal, aggtype='centroid', label=None):
    """
    Plot the FFT magnitude spectrum and overlay the tsfresh-style aggregated feature.

    Parameters:
        signal (array-like): The input time series.
        aggtype (str): One of 'centroid', 'variance', 'skew', 'kurtosis'.

    Returns:
        fig (matplotlib.figure.Figure): Figure with the plot.
    """
    signal = np.asarray(signal)
    fft_abs = np.abs(np.fft.rfft(signal))
    n = len(fft_abs)

    x = np.arange(n, dtype=float)

    def get_moment(y, moment):
        return np.dot(y, x**moment) / np.sum(y)

    def get_centroid(y): return get_moment(y, 1)
    def get_variance(y): return get_moment(y, 2) - get_centroid(y) ** 2

    def get_skew(y):
        var = get_variance(y)
        if var < 0.5:
            return np.nan
        return (get_moment(y, 3) - 3 * get_centroid(y) * var - get_centroid(y) ** 3) / var ** 1.5

    def get_kurtosis(y):
        var = get_variance(y)
        if var < 0.5:
            return np.nan
        return (
            get_moment(y, 4)
            - 4 * get_centroid(y) * get_moment(y, 3)
            + 6 * get_moment(y, 2) * get_centroid(y) ** 2
            - 3 * get_centroid(y)
        ) / var ** 2

    calculation = dict(
        centroid=get_centroid,
        variance=get_variance,
        skew=get_skew,
        kurtosis=get_kurtosis,
    )

    if aggtype not in calculation:
        raise ValueError(f"Invalid aggtype: {aggtype}")

    agg_value = calculation[aggtype](fft_abs)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    ax.plot(fft_abs, label="FFT Magnitude Spectrum")
    ax.set_title(f"FFT Aggregated: {aggtype}\nValue = {agg_value:.4f}")
    ax.set_xlabel("Frequency bin index")
    ax.set_ylabel("Magnitude")
    ax.axvline(agg_value, color='r', linestyle='--', label=f'{aggtype} = {agg_value:.2f}')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig


def plot_fft_coefficient(signal, param, delta_t=None, label=None):
    """
    Plot selected FFT coefficient attributes (real, imag, angle, abs).

    Args:
        signal (array-like): 1D input signal.
        param (list of dict): Each dict has 'coeff' (int) and 'attr' in ['real', 'imag', 'angle', 'abs'].
        delta_t (float, optional): Time step between samples. If provided, x-axis is in Hz.
        label (str, optional): Optional title label.

    Returns:
        matplotlib.figure.Figure
    """

    def complex_agg(x, agg):
        if agg == "real":
            return x.real
        elif agg == "imag":
            return x.imag
        elif agg == "abs":
            return np.abs(x)
        elif agg == "angle":
            return np.angle(x, deg=True)

    signal = np.asarray(signal)
    n = len(signal)
    fft_values = np.fft.rfft(signal)

    # Determine x-axis values
    if delta_t is not None:
        freqs = np.fft.rfftfreq(n, d=delta_t)
        x_vals = freqs
        x_label = "Frequency (Hz)"
    else:
        x_vals = np.fft.rfftfreq(n, d=1)  # cycles/sample
        x_label = "Cycles per Sample"

    # Determine which attributes are being requested
    attrs_requested = sorted(set(p['attr'] for p in param if p['attr'] in {'real', 'imag', 'angle', 'abs'}))
    num_plots = len(attrs_requested)

    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True)

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    if num_plots == 1:
        axs = [axs]

    fig.suptitle("FFT Coefficient Attributes")

    for i, attr in enumerate(attrs_requested):
        coeffs = [p['coeff'] for p in param if p['attr'] == attr]
        x_plot = [x_vals[c] for c in coeffs]
        values = [complex_agg(fft_values[c], attr) for c in coeffs]

        axs[i].stem(x_plot, values, basefmt=" ", linefmt='C0-', markerfmt='C0o')
        axs[i].set_ylabel(attr.capitalize())
        axs[i].grid(True)

    axs[-1].set_xlabel(x_label)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_mean_second_derivative_central(signal, label=None):
    signal = np.asarray(signal)

    if len(signal) < 3:
        raise ValueError("Signal too short for second derivative computation")

    # Compute second derivative using central difference
    second_deriv = signal[2:] - 2 * signal[1:-1] + signal[:-2]
    mid_x = np.arange(1, len(signal) - 1)

    # Full MSDC
    msdc_full = np.mean(second_deriv)

    # tsfresh approximation of MSDC
    msdc_tsfresh = (signal[-1] - signal[-2] - signal[1] + signal[0]) / (2 * (len(signal) - 2))

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Add label colour if provided
    if label is not None:
        color = 'red' if label == 0 else 'blue'
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(5)

    # Plot original signal
    ax1.plot(signal, color='blue')
    ax1.set_title("Original Signal")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    # Plot second derivative
    ax2.plot(mid_x, second_deriv, color='gray', label="Second Derivative")
    ax2.set_title("Second Derivative")
    ax2.set_xlabel("Time Index")
    ax2.set_ylabel("Second Derivative")
    ax2.grid(True)

    # Add secondary y-axis for MSDC results
    ax2b = ax2.twinx()
    ax2b.axhline(msdc_full, color='green', linestyle='--', linewidth=1, label=f'Full MSDC = {msdc_full:.4f}')
    ax2b.axhline(msdc_tsfresh, color='blue', linestyle='--', linewidth=1, label=f'tsfresh MSDC = {msdc_tsfresh:.4f}')
    ax2b.set_ylabel("MSDC Value")

    # Symmetrical y-limits for MSDC axis
    max_val = max(abs(msdc_full), abs(msdc_tsfresh))
    ax2b.set_ylim(-max_val * 1.1, max_val * 1.1)

    # Combine legends from both axes
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the suptitle
    fig.suptitle(f'Mean second derivative central: {msdc_tsfresh}', fontsize=16)

    return fig
