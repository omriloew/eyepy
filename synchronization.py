
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

EVENT_COL = "event_label"
TIME_COL  = "time_stamp"
FRAME_COL = "frame_index"

class SyncError(Exception):
    pass

def _get_numeric_data_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Get numeric columns, excluding specified columns and discrete columns (starting with _)."""
    exclude_set = set(exclude)
    return [c for c in df.columns if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c]) and not c.startswith('_')]

def _get_discrete_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Get discrete columns that start with _ and should not use window pooling."""
    exclude_set = set(exclude)
    return [c for c in df.columns if c not in exclude_set and c.startswith('_')]


def _fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y = a*x + b using least squares."""
    if x.size == 2:
        dx = x[1] - x[0]
        if dx == 0:
            raise SyncError("Identical device times - cannot fit affine transformation.")
        a = (y[1] - y[0]) / dx
        b = y[0] - a * x[0]
        return float(a), float(b)
    
    # Least squares
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def _extract_event_series(df: pd.DataFrame) -> List[Tuple[float, str]]:
    events = []
    for _, row in df.iterrows():
        event_label = str(row[EVENT_COL]).strip()
        if event_label and not pd.isna(event_label) and event_label != '' and event_label != 'nan':
            events.append((float(row[TIME_COL]), event_label))
    events = sorted(events, key=lambda x: x[0])
    print("--------------------------------")
    print(events)
    return events

def _plot_led_events(device_name: str, device_events: List[Tuple[float, str]], 
                     E_events: List[Tuple[float, str]], matched_pairs: List[Tuple[float, float]], 
                     a: float, b: float, save_path: str = None):
    """
    Plot LED events and affine transformation for debugging.
    
    Args:
        device_name: Name of the device being synchronized
        device_events: List of (time, event_label) tuples from device
        E_events: List of (time, event_label) tuples from E (LED events)
        matched_pairs: List of (device_time, E_time) tuples for matched events
        a: Affine transformation slope
        b: Affine transformation intercept
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Event matching visualization
    device_times = [e[0] for e in device_events]
    device_labels = [e[1] for e in device_events]
    E_times = [e[0] for e in E_events]
    E_labels = [e[1] for e in E_events]
    
    # Plot device events
    ax1.scatter(device_times, [1] * len(device_times), c='blue', s=100, label='Device Events', zorder=3)
    for i, (time, label) in enumerate(zip(device_times, device_labels)):
        ax1.annotate(label, (time, 1), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot E (LED) events
    ax1.scatter(E_times, [2] * len(E_times), c='red', s=100, label='LED Events (E)', zorder=3)
    for i, (time, label) in enumerate(zip(E_times, E_labels)):
        ax1.annotate(label, (time, 2), xytext=(5, -15), textcoords='offset points', fontsize=8)
    
    # Draw lines connecting matched pairs
    for dev_time, E_time in matched_pairs:
        ax1.plot([dev_time, E_time], [1, 2], 'g--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Event Type')
    ax1.set_yticks([1, 2])
    ax1.set_yticklabels(['Device', 'LED (E)'])
    ax1.set_title(f'LED Event Matching: {device_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Affine transformation visualization
    if len(matched_pairs) > 0:
        x_matched = [p[0] for p in matched_pairs]
        y_matched = [p[1] for p in matched_pairs]
        
        ax2.scatter(x_matched, y_matched, c='blue', s=100, label='Matched Events', zorder=3)
        
        # Plot the fitted line
        x_line = np.array([min(x_matched), max(x_matched)])
        y_line = a * x_line + b
        ax2.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fit: y = {a:.4f}x + {b:.4f}')
        
        # Add residuals
        y_predicted = [a * x + b for x in x_matched]
        for i, (x, y_obs, y_pred) in enumerate(zip(x_matched, y_matched, y_predicted)):
            ax2.plot([x, x], [y_obs, y_pred], 'g--', alpha=0.3, linewidth=1)
        
        ax2.set_xlabel('Device Time (seconds)')
        ax2.set_ylabel('LED Time (seconds)')
        ax2.set_title(f'Affine Transformation: {device_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"LED events plot saved to: {save_path}")
    else:
        plt.savefig(f'led_events_{device_name}.png', dpi=150, bbox_inches='tight')
        print(f"LED events plot saved to: led_events_{device_name}.png")
    
    plt.close()

def calculate_affine_transformations(df_device: pd.DataFrame, E_events: List[Tuple[float, str]], 
                                    device_name: str = None, plot: bool = True, 
                                    plot_save_path: str = None) -> Tuple[float, float]:
    """
    Calculate affine transformation from device time to E time.
    
    Args:
        df_device: Device dataframe with events
        E_events: List of (time, event_label) tuples from E (LED events)
        device_name: Name of the device (for plotting)
        plot: Whether to create a plot
        plot_save_path: Optional path to save the plot
    
    Returns:
        Tuple of (a, b) affine transformation parameters
    """
    device_events = _extract_event_series(df_device)
    if len(device_events) < 2:
        raise SyncError(f"Device has less than 2 events.")
    x = [device_event[0] for device_event in device_events]
    y = []
    matched_pairs = []
    curr_E_idx = 0
    for device_event in device_events:
        while device_event[1] != E_events[curr_E_idx][1]:
            curr_E_idx += 1
            if curr_E_idx >= len(E_events):
                raise SyncError(f"Device event {device_event[1]} not found in E events.")
        E_time = E_events[curr_E_idx][0]
        y.append(E_time)
        matched_pairs.append((device_event[0], E_time))
        curr_E_idx += 1
    a, b = _fit_affine(np.array(x), np.array(y))
    
    # Plot if requested
    if plot and device_name:
        _plot_led_events(device_name, device_events, E_events, matched_pairs, a, b, plot_save_path)
    
    return a, b

def get_affine_mapping(reference_name: str, devices: Dict[str, pd.DataFrame], df_E: pd.DataFrame, 
                       plot: bool = True, plot_dir: str = None) -> Dict[str, Tuple[float, float]]:
    """
    Synchronize multiple device dataframes to a reference timeline.
    
    Args:
        reference_name: Name of the reference device
        devices: Dictionary of device names to dataframes
        df_E: Events dataframe (LED events)
        plot: Whether to create plots for each device
        plot_dir: Directory to save plots (if None, saves in current directory)
    
    Returns:
        Dictionary mapping device names to (a, b) affine transformation parameters
    """
    E_events = _extract_event_series(df_E)
    affine_transformations = {}
    for device_name, device_df in devices.items():
        plot_path = None
        if plot and plot_dir:
            import os
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'led_events_{device_name}.png')
        elif plot:
            plot_path = f'led_events_{device_name}.png'
        
        a, b = calculate_affine_transformations(device_df, E_events, device_name=device_name, 
                                                plot=plot, plot_save_path=plot_path)
        affine_transformations[device_name] = (a, b)

    a_ref_to_E,b_ref_to_E = affine_transformations[reference_name]
    for device_name, device_df in devices.items():
        if device_name == reference_name:
            a_E_to_ref = 1/a_ref_to_E
            b_E_to_ref = -b_ref_to_E / a_ref_to_E
            affine_transformations[device_name] = (a_E_to_ref, b_E_to_ref)
            continue
        a_dev_to_E, b_dev_to_E = affine_transformations[device_name]
        a_dev_to_ref = a_dev_to_E / a_ref_to_E
        b_dev_to_ref = (b_dev_to_E - b_ref_to_E) / a_ref_to_E
        affine_transformations[device_name] = (a_dev_to_ref, b_dev_to_ref)
    return affine_transformations

def _extract_vals(df: pd.DataFrame, col: str):
    vals = df[col].dropna().to_numpy()
    times = df[TIME_COL][df[col].notna()].to_numpy(dtype=float)
    return zip(times, vals)

def _pool_discrete_col(reference_df: pd.DataFrame, device_df: pd.DataFrame, col: str, window_ms: float = 5.0) -> pd.DataFrame:
    device_vals_and_times = _extract_vals(device_df, col)
    reference_times = reference_df[TIME_COL].to_numpy(dtype=float)  
    window_sec = window_ms / 1000.0
    for time, val in device_vals_and_times:
        nearest_reference_time = np.argmin(np.abs(reference_times - time))
        if np.abs(reference_times[nearest_reference_time] - time) <= window_sec:
            reference_df.loc[nearest_reference_time, col] = val
        else:
            reference_df.loc[nearest_reference_time, col] = np.nan
    return reference_df

def _pool_numeric_col(reference_df: pd.DataFrame, device_df: pd.DataFrame, col: str, window_ms: float = 5.0) -> pd.DataFrame:
    device_times = device_df[TIME_COL].to_numpy(dtype=float)
    reference_times = reference_df[TIME_COL].to_numpy(dtype=float)
    window_sec = window_ms / 1000.0
    
    # Get device values for the column
    device_values = device_df[col].to_numpy(dtype=float)
    
    # Remove NaN values for interpolation
    valid_mask = ~np.isnan(device_values)
    if np.sum(valid_mask) < 2:
        # Not enough data points for interpolation, use window-based approach
        for reference_time in reference_times:
            times_in_window = device_times[np.abs(device_times - reference_time) <= window_sec]
            vals_in_window = device_df.loc[device_df[TIME_COL].isin(times_in_window), col].to_numpy(dtype=float)
            if len(vals_in_window) > 0:
                reference_df.loc[reference_df[TIME_COL] == reference_time, col] = np.mean(vals_in_window)
    else:
        # Use interpolation for continuous values
        valid_times = device_times[valid_mask]
        valid_values = device_values[valid_mask]
        
        # Interpolate to reference times
        interpolated_values = np.interp(reference_times, valid_times, valid_values, 
                                       left=np.nan, right=np.nan)
        
        # For points outside interpolation range, use window-based approach
        for i, reference_time in enumerate(reference_times):
            if np.isnan(interpolated_values[i]):
                times_in_window = device_times[np.abs(device_times - reference_time) <= window_sec]
                vals_in_window = device_df.loc[device_df[TIME_COL].isin(times_in_window), col].to_numpy(dtype=float)
                if len(vals_in_window) > 0:
                    interpolated_values[i] = np.mean(vals_in_window)
        
        reference_df[col] = interpolated_values
    
    return reference_df

def _pool_events(reference_df: pd.DataFrame, device_df: pd.DataFrame) -> pd.DataFrame:
    device_events = np.array(_extract_event_series(device_df))
    reference_events = np.array(_extract_event_series(reference_df))
    device_times = device_events[:, 0].astype(float)
    reference_times = reference_events[:, 0].astype(float)
    for device_time, device_event in zip(device_times, device_events[:, 1]):
        nearest_reference_event_time = np.argmin(np.abs(reference_times - device_time))
        nearest_reference_event = reference_events[nearest_reference_event_time, 1]
        reference_df.loc[nearest_reference_event, EVENT_COL] = f"{nearest_reference_event}; {device_event}"
    return reference_df

def _pool_into_reference_df(reference_df: pd.DataFrame, device_df: pd.DataFrame, window_ms: float = 5.0, pool_events: bool = False) -> pd.DataFrame:
    """
    Pool data from device_df into reference_df.
    
    Args:
        reference_df: Reference DataFrame to pool data into
        device_df: Device DataFrame to pool data from
        window_ms: Window size in milliseconds for pooling
        pool_events: If True, pool event_label column. If False, skip events (events should only come from reference device)
    """
    discrete_cols = _get_discrete_cols(device_df, exclude=[TIME_COL, EVENT_COL])
    numeric_cols = _get_numeric_data_cols(device_df, exclude=[TIME_COL, EVENT_COL])
    for col in discrete_cols:
        _pool_discrete_col(reference_df, device_df, col, window_ms)
    for col in numeric_cols:
        _pool_numeric_col(reference_df, device_df, col, window_ms)
    
    # Only pool events if explicitly requested (should only be True for reference device)
    if pool_events:
        _pool_discrete_col(reference_df, device_df, col="event_label", window_ms=window_ms)

    return reference_df

def synchronize_to_reference(reference_name: str, devices: Dict[str, pd.DataFrame], df_E: pd.DataFrame, 
                             window_ms: float = 5.0, plot_led_events: bool = True, 
                             plot_dir: str = None, device_time_offsets: Dict[str, float] = None) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Synchronize devices to a reference timeline using LED events.
    
    Args:
        reference_name: Name of the reference device
        devices: Dictionary of device names to dataframes
        df_E: Events dataframe (LED events)
        window_ms: Window size in milliseconds for pooling
        plot_led_events: Whether to plot LED event matching for debugging
        plot_dir: Directory to save LED event plots
        device_time_offsets: Dictionary mapping device names to time offsets in seconds.
                           These offsets are applied AFTER affine transformation to correct
                           for processing delays. Positive values shift data forward in time.
    
    Returns:
        Tuple of (synchronized dataframe, affine transformations dictionary)
    """
    if device_time_offsets is None:
        device_time_offsets = {}
    
    affine_transformations = get_affine_mapping(reference_name=reference_name, devices=devices, df_E=df_E, 
                                                plot=plot_led_events, plot_dir=plot_dir)
    
    for device_name, device_df in devices.items():
        if device_name == reference_name:
            continue
        a_dev_to_ref, b_dev_to_ref = affine_transformations[device_name]
        device_df[TIME_COL] = a_dev_to_ref * device_df[TIME_COL] + b_dev_to_ref
        _extract_event_series(device_df)
    
    # Apply manual offsets
    for device_name, device_df in devices.items():
        if device_name == reference_name:
            continue
        if device_name in device_time_offsets:
            offset = device_time_offsets[device_name]
            if offset != 0.0:
                device_df[TIME_COL] = device_df[TIME_COL] + offset
                _extract_event_series(device_df)  # Re-extract after offset

    # Create final synced output
    out = devices[reference_name].copy()
    a_E_to_ref, b_E_to_ref = affine_transformations[reference_name]
    df_E[TIME_COL] = a_E_to_ref * df_E[TIME_COL] + b_E_to_ref

    for device_name, device_df in devices.items():
        if device_name == reference_name:
            continue
        # Pool data columns but NOT events - events should only come from reference device
        _pool_into_reference_df(out, device_df, window_ms, pool_events=False)

    # Don't pool events from Events CSV - events should only come from reference device
    # Events are already in the reference device (out), so we don't need to pool them from anywhere else

    return out, affine_transformations


def main():
    df_E = pd.read_csv("rawResultsExample/E.csv")
    df_SWIR = pd.read_csv("rawResultsExample/IR.csv")
    df_VC = pd.read_csv("rawResultsExample/VC.csv")
    df_EL = pd.read_csv("rawResultsExample/EL.csv")
    df_MD = pd.read_csv("rawResultsExample/MD.csv")
    devices = {
        "VC": df_VC,
        "EL": df_EL,
        "MD": df_MD,
        "SWIR": df_SWIR,
    }
    
    synced_df, mappings = synchronize_to_reference(
        reference_name="SWIR",
        devices=devices,
        df_E=df_E,
        window_ms=5.0
    )
    
    print("Mappings:", mappings)

if __name__ == "__main__":
    main()
