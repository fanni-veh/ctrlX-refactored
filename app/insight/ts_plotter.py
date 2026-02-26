import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')  # or 'Qt5Agg', 'WebAgg'


def plot_simple_ts(df, title=None, signal_sections_dict=None):
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Loop through each unique cycle_id and plot the corresponding data
    for cycle_id in df['cycle_id'].unique():
        cycle_data = df[df['cycle_id'] == cycle_id]
        # if the lable field is present, use color blue for label 1 or red for label 0
        if 'label' in cycle_data.columns:
            color = 'blue' if cycle_data['label'].iloc[0] == 1 else 'red'
            ax.plot(cycle_data['delta_time'], cycle_data['value'], label=f'Cycle {cycle_id}', color=color, linewidth=0.5)
        else:
            ax.plot(cycle_data['delta_time'], cycle_data['value'], label=f'Cycle {cycle_id}', linewidth=0.5)

    # If signal_sections_dict is provided, add horizontal levels and shaded regions
    if signal_sections_dict is not None:
        for section_name, (left_idx, right_idx) in signal_sections_dict.items():
            if 'ns' not in section_name:
                # Convert index to actual x-values
                left = cycle_data['delta_time'].iloc[left_idx]
                right = cycle_data['delta_time'].iloc[right_idx]

                # Shade the region between left and right
                ax.axvspan(left, right, alpha=0.2, label=section_name)

                # Optional lines at the bounds
                ax.axvline(left, color='gray', linestyle='--', linewidth=0.5)
                ax.axvline(right, color='gray', linestyle='--', linewidth=0.5)

                # Annotate the section name at the top center of the region
                label_x = left + 0.01 * (right - left)  # small offset from left
                label_y = ax.get_ylim()[1] * 0.98       # near the top of y-axis

                ax.text(label_x, label_y, section_name,
                        va='top', ha='left', fontsize=8, color='gray')

    # Customize the plot
    ax.set_xlabel('Delta Time [s]')
    ax.set_ylabel('Value [mA]')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Time Series Plot')
    ax.grid(True)
    # ax.legend()

    # Return the figure object
    return fig


def plot_stacked_ts(df, reference_signal=None,  title=None):

    unique_cycles = df['cycle_id'].unique()

    fig, ax = plt.subplots(1, 1)
    offset_step = 0.8

    # Plot reference signal
    if reference_signal is not None:
        ref = reference_signal.astype(float)
        norm = ref / np.nanmax(ref)
        ax.plot(norm, color='blue', label='Reference', linewidth=0.5)

    # Plot each cycle
    for idx, cycle_id in enumerate(unique_cycles, start=1):
        vals = df[df['cycle_id'] == cycle_id]['value'].values.astype(float)
        norm = vals / np.nanmax(vals)
        ax.plot(norm + idx * offset_step, color='black', label=f'C{cycle_id}', linewidth=0.5)

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Normalized Value + Offset')
    ax.grid(axis='x')  # Only show vertical grid lines
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is None:
        title = f'Normalized Time Series Plot ({len(unique_cycles)} cycles)'
    fig.suptitle(title)

    return fig


def plot_ts_and_xcorr(x, y, cc, lags, lag, vertex_x, vertex_y, x_fit, y_fit, window, title=None):

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), tight_layout=True)

    axs[0].plot(x, label='Reference Signal')
    axs[0].set_title('Reference Signal')
    axs[0].legend()

    axs[1].plot(y, label='Target Signal', color='orange')
    axs[1].set_title('Target Signal')
    axs[1].legend()

    axs[2].plot(lags, cc, label='Cross-correlation', color='green')
    axs[2].axvline(vertex_x, color='red', linestyle='--', label=f'Max Corr Lag = {lag}')
    axs[2].plot(x_fit, y_fit, '--', color='purple', label='Parabolic Fit')
    axs[2].plot(vertex_x, vertex_y, 'o', color='black', label=f'Interpolated Peak = {vertex_x:.2f}')
    axs[2].set_title('Cross-correlation')
    axs[2].legend()

    # --- Fourth plot: zoom on window points around the peak ---
    peak_idx = np.argmax(cc)
    start = max(peak_idx - window, 0)
    end = min(peak_idx + window + 1, len(cc))
    axs[3].plot(lags[start:end], cc[start:end], label='Cross-correlation (zoom)', color='green')
    axs[3].axvline(vertex_x, color='red', linestyle='--', label=f'Max Corr Lag = {lag}')
    # Only plot the fit and vertex if they are within the zoom window
    if np.any((x_fit >= lags[start]) & (x_fit <= lags[end-1])):
        axs[3].plot(x_fit, y_fit, '--', color='purple', label='Parabolic Fit')
    if lags[start] <= vertex_x <= lags[end-1]:
        axs[3].plot(vertex_x, vertex_y, 'o', color='black', label=f'Interpolated Peak = {vertex_x:.2f}')
    axs[3].set_title('Cross-correlation (Zoomed)')
    axs[3].legend()
    # ----------------

    # --- Fifth plot: overlapped signals with y offset by round(vertex_x) ---
    offset = int(round(vertex_x))
    if offset > 0:
        y_aligned = np.pad(y, (offset, 0), mode='constant', constant_values=np.nan)[:len(x)]
    elif offset < 0:
        y_aligned = np.pad(y, (0, -offset), mode='constant', constant_values=np.nan)[-offset:len(x)-offset]
    else:
        y_aligned = y[:len(x)]
    axs[4].plot(x, label='Reference Signal', color='black', linewidth=0.5)
    axs[4].plot(y_aligned, label=f'Target Signal (shifted by {vertex_x})', color='green', alpha=0.7)
    axs[4].set_title('Signals Overlapped (Target Shifted by Interpolated Lag)')
    axs[4].legend()
    # -----------------------------------------------------------------------

    if title:
        fig.suptitle(title, fontsize=14)

    return fig


# def plot_inferred_mechanics(df, torque_constant_mNm_per_A, inertia_kg_m2=1e-4, initial_velocity=0.0, initial_position=0.0):
#     """
#     Infers torque, angular velocity, and position from current signal.

#     Parameters:
#     - df: DataFrame with 'cycle_id', 'delta_time', 'value' (in mA)
#     - torque_constant_mNm_per_A: motor torque constant (mNm/A)
#     - inertia_kg_m2: moment of inertia (default 1e-4)
#     - initial_velocity: initial angular velocity in rad/s
#     - initial_position: initial angular position in rad

#     Returns:
#     - result_df: DataFrame with time, current (A), torque (Nm), angular velocity (rad/s), position (rad)
#     - and a plot of all quantities
#     """
#     # Select first cycle
#     first_cycle_id = df['cycle_id'].iloc[0]
#     cycle_df = df[df['cycle_id'] == first_cycle_id].copy()

#     # Convert current from mA to A
#     current = cycle_df['value'].values / 1000.0  # A

#     # Shift current so that the mean of the last 5% of the signal is zero
#     n_last = max(1, int(0.05 * len(current)))
#     current = current - np.mean(current[-n_last:])

#     # Convert delta_time to cumulative time vector in seconds
#     time = cycle_df['delta_time'].cumsum().values / 1000.0  # s

#     # Convert torque constant from mNm/A to Nm/A
#     torque_constant = torque_constant_mNm_per_A / 1000.0  # Nm/A

#     # Compute torque
#     torque = current * torque_constant  # Nm

#     # Angular acceleration
#     alpha = torque / inertia_kg_m2  # rad/s²

#     # Angular velocity (integrate alpha)
#     omega = cumtrapz(alpha, time, initial=0) + initial_velocity  # rad/s

#     # Angular position (integrate omega)
#     theta = cumtrapz(omega, time, initial=0) + initial_position  # rad

#     # Pack results in a DataFrame
#     result_df = pd.DataFrame({
#         'time_s': time,
#         'current_A': current,
#         'torque_Nm': torque,
#         'angular_velocity_rad_s': omega,
#         'angular_position_rad': theta
#     })

#     # Plot
#     fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

#     axs[0].plot(time, current)
#     axs[0].set_ylabel("Current [A]")
#     axs[0].grid(True)

#     axs[1].plot(time, torque)
#     axs[1].set_ylabel("Torque [Nm]")
#     axs[1].grid(True)

#     axs[2].plot(time, omega)
#     axs[2].set_ylabel("Angular Velocity [rad/s]")
#     axs[2].grid(True)

#     axs[3].plot(time, theta)
#     axs[3].set_ylabel("Position [rad]")
#     axs[3].set_xlabel("Time [s]")
#     axs[3].grid(True)

#     fig.suptitle("Inferred Mechanical Quantities from Current Signal")
#     plt.tight_layout()

#     return fig
