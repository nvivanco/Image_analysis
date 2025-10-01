import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, OptimizeWarning
from numba import jit

def plot_growth_analysis(all_cells_pd, cell_id, cell_id_col, time_col, length_col, prominence=0.3, distance=3, window_size=3):

    """Analyzes and plots cell growth data with exponential fits for detected phases."""

    cell_df = all_cells_pd[all_cells_pd[cell_id_col] == cell_id].copy()

    growth_phases = detect_growth_phases(cell_df, time_col, length_col, prominence, distance, window_size)

    cell_df = cell_df.assign(y_fit=np.nan, residuals=np.nan, r_squared=np.nan, growth_phase=None, growth_rate_constant=np.nan, doubling_time = np.nan)

    plt.figure(figsize=(10, 6))
    _plot_raw_data(cell_df, time_col, length_col, cell_id)

    if not growth_phases:
        plt.title("Cell Data (No Growth Phases Detected)")
        print("No growth phases detected.")
    else:
        for i, (start, end) in enumerate(growth_phases):
            cell_df = _process_growth_phase(cell_df, i + 1, start, end, time_col, length_col)

        plt.title("Cell Growth Analysis")

    plt.xlabel('Time Frame Index')
    plt.ylabel('Cell Length (pxls)')
    plt.legend()
    plt.show()

    return cell_df


def fit_cell_growth(df, time_col, length_col):
    """
    Fits an exponential function to cell growth data within a specified time range.

    Args:
        df (pd.DataFrame): DataFrame containing cell data.
        time_col (str): Name of the time column.
        length_col (str): Name of the length column.
        start_time (float): Start time of the growth phase.
        end_time (float): End time of the growth phase.

    Returns:
        tuple: (popt, pcov, y_fit, r_squared)
               popt: Optimized parameters from curve_fit.
               pcov: Covariance matrix from curve_fit.
               y_fit: Fitted y-values.
               r_squared: R-squared value of the fit.
               Returns (None, None, None, None) if fitting fails.
    """

    df_growth = df.copy()

    if df_growth.empty:
        print("Warning: No data within the specified time range.")
        return None, None, None, None

    x_data = df_growth[time_col].values
    y_data = df_growth[length_col].values

    # Remove NaN values
    mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_data = x_data[mask]
    y_data = y_data[mask]
    log_y_data = np.log(y_data)


    if len(x_data) < 3:
        print("Warning: Insufficient data points after NaN removal.")
        return None, None, None, None


    try:
        popt, pcov = curve_fit(linear_model, x_data, log_y_data)

        log_y_fit = linear_model(x_data, *popt)
        y_fit = np.exp(log_y_fit)
        r_squared = r2_score(y_data, y_fit)


        return popt, pcov, y_fit, r_squared

    except (RuntimeError, OptimizeWarning) as e:
        print(f"Error during curve fitting: {e}")
        return None, None, None, None
    except ValueError as e:
        print(f"ValueError during curve fitting: {e}")
        return None, None, None, None


def modified_exponential(x, a, b, c):
    exponent = b * x
    if np.any(exponent > 700):
        return np.inf
    return a * np.exp(exponent) + c

def linear_model(x, m, c):
    return m * x + c

def detect_growth_phases(df, time_col, length_col, prominence=0.5, distance=5, window_size=3):
    """Detects growth phases using peak detection on the smoothed derivative, refined by the second derivative."""

    derivative, second_derivative = calculate_smoothed_derivative(df, length_col, window_size)

    peaks, _ = find_peaks(derivative, prominence=prominence, distance=distance)

    time_values = df[time_col].to_numpy()

    growth_phases = refine_peaks(peaks, second_derivative, time_values)

    return growth_phases


def refine_peaks(peaks, second_derivative, time_values):
    """Refines peaks using the second derivative."""
    length = len(time_values)
    refined_indices = refine_peaks_numba(np.array(peaks), second_derivative, length) #use numba optimized function.
    refined_phases = []

    for start_index, end_index in refined_indices:
        start_time = time_values[start_index]
        end_time = time_values[end_index]
        refined_phases.append((start_time, end_time))

    return refined_phases


@jit(nopython=True)
def refine_peaks_numba(peaks, second_derivative, length):
    """Refines peaks using the second derivative (Numba-optimized)."""
    refined_phases = []
    last_end_index = -1

    for peak_index in peaks:
        if peak_index > last_end_index:
            # Refine start time using second derivative
            start_index = peak_index
            while start_index > 1 and second_derivative[start_index] > 0:
                start_index -= 1

            # Refine end time using second derivative
            end_index = peak_index
            while end_index < length - 2 and second_derivative[end_index] > 0:
                end_index += 1

            refined_phases.append((start_index, end_index))
            last_end_index = end_index

    return refined_phases

def calculate_smoothed_derivative(df, length_col, window_size=3):
    """Calculates the smoothed first and second derivatives."""
    df_smoothed = df[length_col].rolling(window=window_size, center=True).mean()
    derivative = df_smoothed.diff().to_numpy()
    second_derivative = pd.Series(derivative).diff().to_numpy()
    return derivative, second_derivative


def _plot_raw_data(cell_df, time_col, length_col, cell_id):
    """Plots the raw cell data."""
    plt.plot(cell_df[time_col], cell_df[length_col], 'o', color='grey', alpha=0.5, label=f'Cell {cell_id} Data')


def _plot_fitted_phase(phase_data, time_col, length_col, y_fit, phase_index, start, end):
    """Plots the fitted exponential curve for a growth phase."""
    plt.plot(phase_data[time_col], phase_data[length_col], 'o', label=f'Phase {phase_index} ({start}-{end})')
    plt.plot(phase_data[time_col], y_fit, '-', label=f'Fit {phase_index} ({start}-{end})')


def _process_growth_phase(cell_df, phase_index, start, end, time_col, length_col):
    """Processes a single growth phase, fits the data, and updates cell_df."""
    phase_data = cell_df[(cell_df[time_col] >= start) & (cell_df[time_col] <= end)]

    if len(phase_data) < 3:
        print(f"Skipping Phase {phase_index} ({start}-{end}): Insufficient data.")
        return cell_df

    popt, _, y_fit, r_squared = fit_cell_growth(phase_data, time_col, length_col)

    if popt is None or r_squared is None or (r_squared is not None and r_squared < 0.3):
        if r_squared is None:
            r_squared_str = "None"
        else:
            r_squared_str = f"{r_squared:.2f}"
        print(f"Skipping Phase {phase_index} ({start}-{end}): Fit failed or poor R-squared ({r_squared_str}).")
        return cell_df

    residuals = phase_data[length_col] - y_fit

    growth_rate_constant = popt[0]
    doubling_time = np.log(2) / growth_rate_constant

    cell_df.loc[phase_data.index, ['y_fit', 'residuals', 'r_squared', 'growth_phase', 'growth_rate_constant', 'doubling_time']] = \
        [y_fit, residuals, r_squared, f'Phase {phase_index} ({start}-{end})', growth_rate_constant, doubling_time]

    _plot_fitted_phase(phase_data, time_col, length_col, y_fit, phase_index, start, end)
    return cell_df