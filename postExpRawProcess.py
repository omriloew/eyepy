from pickle import NONE
from tkinter.constants import FALSE
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import coloredPrint as cp
import synchronization as sync
import videoProccesing as vp
import config as config
import edfHandler as edf

always_keep_cols = ['time_stamp', 'event_label'] #DO NOT CHANGE THIS
#===============================================
# events configuration
#===============================================
events_csv_path = 'omri2/1/main/omri2_main_events.csv'
keep_events_cols = ['_event_code']
#===============================================
# swir configuration
#===============================================
process_swir = True
swir_csv_path = 'omri2/1/main/roi_intensity_results_omri_main_simple.csv'  # Path to SWIR CSV file (should have time_sec and stat_intensity columns)
keep_swir_cols = ['frame', 'dyn_intensity', 'dyn_darkness', 'stat_intensity', 'stat_darkness', 'roi_update']
swir_scale_factors = {'dyn_intensity': 100, 'dyn_darkness': 200}  # Multipliers for SWIR columns: {'column_name': multiplier}
swir_drift_factors = {'dyn_intensity': -8000.0, 'dyn_darkness': -30000.0}  # Drift/offset for SWIR columns: {'column_name': offset_value}
# LED event detection intervals: only detect events within these time ranges (in seconds)
# Format: list of tuples [(start1, end1), (start2, end2), ...]
# Events outside these intervals will be ignored
swir_led_detection_intervals = []  # Example: only detect events between 20s and 93s
swir_led_threshold_multiplier = 6  # Threshold multiplier for LED event detection (higher = more strict)
#===============================================
# medoc configuration
#===============================================
process_medoc = True
medoc_csv_path = 'omri2/1/main/omri2_main_medoc_events.csv'
keep_medoc_cols = ['temperature_c']
medoc_scale_factors = {'temperature_c': 100.0}  # Multipliers for MEDOC columns: {'column_name': multiplier}
medoc_drift_factors = {'temperature_c': 4000}  # Drift/offset for MEDOC columns: {'column_name': offset_value}
#===============================================
# eeg configuration
#===============================================
process_eeg = True
eeg_file_path = 'PILOT/PPM_MAIN_PILOT_20260203_115754.mff'
keep_eeg_cols = ["E4", "E2"]  # Keep all channels - empty list means keep all. E219, E25, E26 show best blink characteristics
eeg_scale_factors = {"E4": 10.0, "E2":5.0, "EMG Leg": 0.01}  # Multipliers for EEG columns: {'column_name': multiplier}
eeg_drift_factors = {"E4": -109000.0, "EMG Leg": 5000.0, "E2": -147000.0}  # Drift/offset for EEG columns: {'column_name': offset_value}
#===============================================
# eyelink configuration 
#===============================================
process_eyelink = True
eyelink_file_path = 'omri2/1/main/test.edf'
keep_eyelink_cols = ['ps'] # ps or ps_left or ps_right
eyelink_scale_factors = {'xpos': 1.0, 'ypos': 1.0, 'ps': 1.0}  # Multipliers for Eyelink columns: {'column_name': multiplier}
eyelink_drift_factors = {"ps": 4000.0}  # Drift/offset for Eyelink columns: {'column_name': offset_value}
#===============================================
# output configuration
#===============================================
figures_dir_path = 'output/figures'
output_dir = 'post_exp_raw_process_results'
#===============================================
# session configuration
#===============================================
session_type = 'main'
participant_id = 'omri2'
session_number = '1'
#===============================================
# synchronization configuration
#===============================================
sychronize_to = 'SWIR'
# Device-specific time offset corrections (in seconds) to account for processing delays
# Positive values shift device data forward in time, negative values shift backward
# These offsets are applied AFTER affine transformation but BEFORE pooling
# If ps (EL) events appear before dyn_darkness (SWIR) events, use negative EL offset
# Example: if ps drops 0.05s before dyn_darkness, set EL: -0.05 to align them
device_time_offsets = {
    'EL': 0.0,    # Eyelink offset (seconds) - negative to delay EL to match SWIR
    'SWIR': 0.0,    # SWIR offset (seconds) - reference device, usually 0
    'MEDOC': 0.0,   # MEDOC offset (seconds)
    'EEG': 0.0      # EEG offset (seconds)
}
#===============================================
# visualization configuration
#===============================================
columns_to_exclude_from_plot = ['frame_index', 'frame', 'roi_update', 'stat_darkness', 'dyn_intensity', 'stat_intensity']  # Columns to exclude from final visualization
# Additional CSV to plot alongside synced data (optional)
# CSV should have 'time_sec' column (will be matched to 'time_stamp' in synced data)
additional_csv_to_plot = NONE # Set to CSV file path or None to skip
additional_csv_multiplier = 100.0  # Multiplier to apply to all numeric columns from additional CSV
additional_csv_drift_factors = {}  # Drift/offset for additional CSV columns: {'column_name': offset_value}
additional_csv_exclude_cols = []  # Columns to exclude from additional CSV when plotting
# Example: additional_csv_to_plot = '2/4/pain_rating/roi_intensity_results_NRS4.csv'
# Example: additional_csv_multiplier = 0.001  # Scale down by 1000
# Example: additional_csv_exclude_cols = ['frame', 'frame_index', 'other_col']
POOLING_WINDOW_MS = 30.0

def configure_output_paths():
    
    paticipant_dir = f'{output_dir}/{participant_id}'
    os.makedirs(paticipant_dir, exist_ok=True)
    
    session_output_dir = f'{paticipant_dir}/{session_type}_{session_number}'
    if not os.path.exists(session_output_dir):
        os.makedirs(session_output_dir)
    else:
        cp.colored_print(f"Warning: Session output directory already exists: {session_output_dir}. Will overwrite existing files.", color=cp.Fore.YELLOW)
    
    output_csv_sufix = f'{participant_id}_{session_type}_{session_number}.csv'

    return session_output_dir, output_csv_sufix

def configure_input_paths():
    return events_csv_path, medoc_csv_path, eeg_file_path, eyelink_file_path, swir_csv_path

def swir_csv_to_df(session_output_dir=None, output_csv_sufix=None, swir_csv_path=None):
    print("================================================")
    cp.colored_print("== Converting SWIR CSV to DataFrame ==", color=cp.Fore.CYAN)
    
    # Load the CSV file
    if not os.path.exists(swir_csv_path):
        raise FileNotFoundError(f"SWIR CSV file not found: {swir_csv_path}")
    
    swir_df = pd.read_csv(swir_csv_path)
    cp.colored_print(f"Loaded {len(swir_df)} rows from {swir_csv_path}", color=cp.Fore.GREEN)
    
    # Check required columns
    if 'time_sec' not in swir_df.columns:
        raise ValueError(f"CSV must contain 'time_sec' column. Found columns: {swir_df.columns.tolist()}")
    if 'stat_intensity' not in swir_df.columns:
        raise ValueError(f"CSV must contain 'stat_intensity' column. Found columns: {swir_df.columns.tolist()}")
    
    # Rename time_sec to time_stamp
    swir_df = swir_df.rename(columns={'time_sec': 'time_stamp'})
    
    # Add event_label column (initialize as empty string)
    swir_df['event_label'] = ''
    
    # Extract LED events from stat_intensity column
    intensity_array = swir_df['stat_intensity'].values
    time_stamps = swir_df['time_stamp'].values
    
    cp.colored_print("Extracting LED events from stat_intensity column...", color=cp.Fore.YELLOW)
    light_events_indices = vp.extract_light_events_from_intensity(
        intensity_array, 
        time_stamps, 
        do_plot=True,
        show_plot=True,
        save_plot_path=None,
        detection_intervals=swir_led_detection_intervals,
        threshold_multiplier=swir_led_threshold_multiplier
    )
    led_on_indices = light_events_indices[0]  # LED_ON row indices
    led_off_indices = light_events_indices[1]  # LED_OFF row indices
    
    cp.colored_print(f"LED_ON row indices (auto-detected): {led_on_indices}", color=cp.Fore.BLUE)
    cp.colored_print(f"LED_OFF row indices (auto-detected): {led_off_indices}", color=cp.Fore.BLUE)
    
    # Mark LED_ON events
    for row_idx in led_on_indices:
        if 0 <= row_idx < len(swir_df):
            swir_df.iloc[row_idx, swir_df.columns.get_loc('event_label')] = "LED_ON"
    
    # Mark LED_OFF events
    for row_idx in led_off_indices:
        if 0 <= row_idx < len(swir_df):
            swir_df.iloc[row_idx, swir_df.columns.get_loc('event_label')] = "LED_OFF"
    
    # Save the updated CSV with LED events
    if session_output_dir and output_csv_sufix:
        save_path = f'{session_output_dir}/swir_{output_csv_sufix}'
        swir_df.to_csv(save_path, index=False)
        cp.colored_print(f"saved file: {save_path}", color=cp.Fore.BLUE)
    
    # Show preview of events
    led_events = swir_df[swir_df['event_label'].isin(['LED_ON', 'LED_OFF'])]
    if not led_events.empty:
        cp.colored_print(f"LED Events Preview:", color=cp.Fore.CYAN)
        cp.colored_print(led_events.head(10).to_string(index=False), color=cp.Fore.BLUE)
    else:
        cp.colored_print("Warning: No LED events detected", color=cp.Fore.YELLOW)
    
    print("================================================")
    return swir_df

def get_eyelink_df(eyelink_file_path, session_output_dir=None, output_csv_sufix=None):
    print("================================================")
    cp.colored_print("== Converting EYE LINK EDF to DataFrame ==", color=cp.Fore.CYAN)
    df = edf.edf_to_df(eyelink_file_path)
    keep_cols = always_keep_cols + keep_eyelink_cols
    df = df[keep_cols]
    cp.colored_print(f"EyeLink DataFrame: {df.head()}", color=cp.Fore.BLUE)
    # Save the CSV if output directory is provided
    if session_output_dir and output_csv_sufix:
        save_path = f'{session_output_dir}/eyelink_{output_csv_sufix}'
        df.to_csv(save_path, index=False)
        cp.colored_print(f"saved file: {save_path}", color=cp.Fore.BLUE)
    print("================================================")
    return df

def get_eeg_df(eeg_file_path):
    print("================================================")
    cp.colored_print("== Converting EEG EDF/MFF to DataFrame ==", color=cp.Fore.CYAN)
    df = edf.edf_to_df(eeg_file_path)
    # If keep_eeg_cols is empty, keep all columns (except always keep time_stamp and event_label)
    if len(keep_eeg_cols) > 0:
        keep_cols = always_keep_cols + keep_eeg_cols
        df = df[keep_cols]
    # else: keep all columns (they already include time_stamp and event_label)
    cp.colored_print(f"EEG DataFrame: {df.head()}", color=cp.Fore.BLUE)
    print("================================================")
    print(df.head())
    return df

def get_medoc_df(medoc_csv_path):
    print("================================================")
    cp.colored_print("== Converting MEDOC CSV to DataFrame ==", color=cp.Fore.CYAN)
    df = pd.read_csv(medoc_csv_path)
    
    df = df.rename(columns={'command_id': 'event_label'})
        # Keep only the desired columns
    keep_cols = always_keep_cols + keep_medoc_cols
    df = df[keep_cols]
    # Adjust event_label as specified
    def adjust_event_label(val):
        if val == 'GET_STATUS':
            return ""
        else:
            return f"MEDOC_{val}"

    df['event_label'] = df['event_label'].apply(adjust_event_label)
    
    # Convert MEDOC timestamps from Unix epoch to relative time (seconds from first event)
    # MEDOC timestamps are Unix timestamps, but EVENTS timestamps are relative from session start
    # We'll make MEDOC timestamps relative by subtracting the first timestamp
    if len(df) > 0:
        first_timestamp = df['time_stamp'].min()
        df['time_stamp'] = df['time_stamp'] - first_timestamp
    
    cp.colored_print(f"MEDOC DataFrame: {df.head()}", color=cp.Fore.BLUE)
    print("================================================")
    return df

def get_events_df(events_csv_path):
    df = pd.read_csv(events_csv_path)
    # Keep only the desired columns
    keep_cols = always_keep_cols + keep_events_cols
    df = df[keep_cols]

    print(df.head())
    return df

def print_def_columns_and_unique_values(df, df_name="DATAFRAME"):
    print(f"\n--- {df_name}: COLUMN HEADERS AND UNIQUE VALUES ---")
    for col in df.columns:
        unique_vals = df[col].unique()
        print(f"Column: '{col}'")
        print(f"Unique values ({len(unique_vals)}): {unique_vals[:20]}")
        if len(unique_vals) > 20:
            print("...(truncated)...")
        print("------")

def plot_df_in_time(df, columns_to_plot=[], scale_factors=None, drift_factors=None, df_name="DATAFRAME", save_path=None):
    """
    Plots specified columns against time on a single graph.
    
    Args:
        df: DataFrame with 'time_stamp' and optionally 'event_label' columns
        columns_to_plot: List of column names to plot
        scale_factors: Dict mapping column name -> float factor to scale the values
        drift_factors: Dict mapping column name -> float offset to add to values (for visual separation)
        df_name: Name for the plot title
        save_path: Optional path to save the plot as PNG (if None, shows the plot)
    
    Behavior:
        - Always plots time_stamp on x-axis
        - Always plots event_label as discrete markers (if present)
        - Columns starting with '_' are plotted as discrete (markers where values exist)
        - Other columns are plotted as continuous (lines)
        - All plots on the same graph
        - Extreme values (>10000 or <-10000) are clipped for visualization to prevent plot distortion
    """
    if scale_factors is None:
        scale_factors = {}
    if drift_factors is None:
        drift_factors = {}
    
    time_col = 'time_stamp'
    
    # Validate time_stamp exists
    if time_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{time_col}' column")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Track if we have events to plot (for legend)
    has_events = False
    
    # Separate columns into discrete and continuous
    discrete_cols = []
    continuous_cols = []
    
    for col in columns_to_plot:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame, skipping.")
            continue
        
        if col.startswith('_'):
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)
    
    # Plot continuous columns as lines
    for col in continuous_cols:
        scale = scale_factors.get(col, 1.0)
        drift = drift_factors.get(col, 0.0)
        values = df[col] * scale + drift
        label_parts = []
        if scale != 1.0:
            label_parts.append(f"×{scale:.2f}")
        if drift != 0.0:
            label_parts.append(f"+{drift:.1f}" if drift > 0 else f"{drift:.1f}")
        label = f"{col} ({', '.join(label_parts)})" if label_parts else col
        ax.plot(df[time_col], values, label=label, linewidth=1.5, alpha=0.8)
    
    # Plot discrete columns as markers
    for col in discrete_cols:
        scale = scale_factors.get(col, 1.0)
        drift = drift_factors.get(col, 0.0)
        # Get rows where this column has non-null/non-zero values
        discrete_df = df[df[col].notna() & (df[col] != 0)].copy()
        if len(discrete_df) > 0:
            values = discrete_df[col] * scale + drift
            label_parts = []
            if scale != 1.0:
                label_parts.append(f"×{scale:.2f}")
            if drift != 0.0:
                label_parts.append(f"+{drift:.1f}" if drift > 0 else f"{drift:.1f}")
            label = f"{col} ({', '.join(label_parts)})" if label_parts else col
            ax.scatter(discrete_df[time_col], values, 
                      marker='o', s=50, alpha=0.6, 
                      label=label,
                      zorder=4)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Values")
    ax.set_title(f"{df_name}: All Signals vs Time")
    ax.grid(True, alpha=0.3)
    
    # Always plot event_label as discrete if it exists (after setting labels so we can get y-limits)
    # Plot events as vertical lines spanning the plot height with text labels
    if 'event_label' in df.columns:
        event_df = df[df['event_label'] != ''].copy()
        if len(event_df) > 0:
            has_events = True
            # Get y-axis range for label positioning (after all data is plotted)
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            label_y = y_max - (y_range * 0.05)  # Position labels near top
            
            # Plot vertical lines and labels for each event
            for idx, row in event_df.iterrows():
                ax.axvline(x=row[time_col], color='red', linestyle='--', 
                          alpha=0.5, linewidth=1, zorder=3)
                # Add text label
                event_text = str(row['event_label']).strip()
                if event_text:  # Only add label if not empty
                    ax.text(row[time_col], label_y, event_text, 
                           rotation=90, fontsize=8, ha='right', va='bottom',
                           color='red', alpha=0.8, zorder=6)
    
    # Add events to legend if present
    if has_events:
        from matplotlib.lines import Line2D
        event_line = Line2D([0], [0], color='red', linestyle='--', 
                           alpha=0.5, linewidth=1, label='Events')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(event_line)
        labels.append('Events')
        ax.legend(handles, labels, loc='best', fontsize=8)
    else:
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    # Configure paths
    session_output_dir, output_csv_sufix = configure_output_paths()
    events_csv_path, medoc_csv_path, eeg_file_path, eyelink_file_path, swir_csv_path = configure_input_paths()
    devices = {}
    if process_swir:
        swir_df = swir_csv_to_df(session_output_dir, output_csv_sufix, swir_csv_path)
        devices["SWIR"] = swir_df
    if process_eyelink:
        eyelink_df = get_eyelink_df(eyelink_file_path, session_output_dir, output_csv_sufix)
        devices["EL"] = eyelink_df
    if process_eeg:
        eeg_df = get_eeg_df(eeg_file_path)
        # Save EEG CSV before synchronization
        eeg_pre_sync_path = f'{session_output_dir}/EEG_pre_sync_{output_csv_sufix}'
        eeg_df.to_csv(eeg_pre_sync_path, index=False)
        cp.colored_print(f"EEG (pre-sync) saved to: {eeg_pre_sync_path}", color=cp.Fore.GREEN)
        devices["EEG"] = eeg_df
    if process_medoc:
        medoc_df = get_medoc_df(medoc_csv_path)
        devices["MEDOC"] = medoc_df

    events_df = get_events_df(events_csv_path)

    synced_df, mappings = sync.synchronize_to_reference(
        reference_name=sychronize_to,
        devices=devices,
        df_E=events_df,
        window_ms=POOLING_WINDOW_MS,
        plot_led_events=True,
        plot_dir=session_output_dir,
        device_time_offsets=device_time_offsets
    )

    # Save each individual device's synced DataFrame
    for device_name, device_df in devices.items():
        save_path = f'{session_output_dir}/synced_{device_name}_{output_csv_sufix}'
        device_df.to_csv(save_path, index=False)
        cp.colored_print(f"Synced {device_name} saved to: {save_path}", color=cp.Fore.BLUE)
    
    # Save synced events DataFrame
    events_save_path = f'{session_output_dir}/synced_events_{output_csv_sufix}'
    events_df.to_csv(events_save_path, index=False)
    cp.colored_print(f"Synced events saved to: {events_save_path}", color=cp.Fore.BLUE)
    
    # Save the final combined synced result (all devices pooled together)
    final_synced_path = f'{session_output_dir}/synced_combined_{output_csv_sufix}'
    synced_df.to_csv(final_synced_path, index=False)
    cp.colored_print(f"Final combined synced data saved to: {final_synced_path}", color=cp.Fore.GREEN)
    
    # Visualize the final combined synced result
    # Get all numeric columns (excluding time_stamp, event_label, and configured exclusions) for plotting
    exclude_cols = ['time_stamp', 'event_label'] + columns_to_exclude_from_plot
    numeric_cols = [col for col in synced_df.columns 
                    if col not in exclude_cols 
                    and synced_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    # Limit to reasonable number of columns for visualization
    if len(numeric_cols) > 10:
        # Prioritize common columns
        priority_cols = ['xpos', 'ypos', 'ps', 'temperature_c']
        plot_cols = [col for col in priority_cols if col in numeric_cols]
        plot_cols.extend([col for col in numeric_cols if col not in plot_cols][:10-len(plot_cols)])
    else:
        plot_cols = numeric_cols
    
    # Load and merge additional CSV if provided
    if additional_csv_to_plot is not None and os.path.exists(additional_csv_to_plot):
        print(f"Loading additional CSV: {additional_csv_to_plot}")
        additional_df = pd.read_csv(additional_csv_to_plot)
        
        # Rename time_sec to time_stamp for consistency
        if 'time_sec' in additional_df.columns:
            additional_df = additional_df.rename(columns={'time_sec': 'time_stamp'})
        
        # Get numeric columns from additional CSV (excluding time_stamp and configured exclusions)
        exclude_additional_cols = ['time_stamp'] + additional_csv_exclude_cols
        additional_numeric_cols = [col for col in additional_df.columns 
                                   if col not in exclude_additional_cols 
                                   and additional_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        # Merge additional columns into synced_df by interpolating to match time_stamp
        for col in additional_numeric_cols:
            # Interpolate additional data to synced time points and apply multiplier
            synced_df[col] = np.interp(synced_df['time_stamp'], 
                                       additional_df['time_stamp'], 
                                       additional_df[col]) * additional_csv_multiplier
            # Add to plot columns
            if col not in plot_cols:
                plot_cols.append(col)
        
        cp.colored_print(f"Added {len(additional_numeric_cols)} columns from additional CSV", color=cp.Fore.GREEN)
    
    # Build scale factors and drift factors dictionaries from all device configurations
    scale_factors = {}
    scale_factors.update(eyelink_scale_factors)
    scale_factors.update(medoc_scale_factors)
    scale_factors.update(eeg_scale_factors)
    scale_factors.update(swir_scale_factors)
    
    drift_factors = {}
    drift_factors.update(eyelink_drift_factors)
    drift_factors.update(medoc_drift_factors)
    drift_factors.update(eeg_drift_factors)
    drift_factors.update(swir_drift_factors)
    
    # Add drift factors for additional CSV columns if provided
    if additional_csv_to_plot is not None and os.path.exists(additional_csv_to_plot):
        for col in additional_numeric_cols:
            if col in additional_csv_drift_factors:
                drift_factors[col] = additional_csv_drift_factors[col]
    
    plot_save_path = f'{session_output_dir}/synced_combined_plot_{output_csv_sufix}'.replace('.csv', '.png')
    plot_df_in_time(synced_df, columns_to_plot=plot_cols, scale_factors=scale_factors, drift_factors=drift_factors, df_name="SYNCED COMBINED", save_path=plot_save_path)
    
    print(f"Mappings: {mappings}")


def test():
    medoc_df = get_medoc_df('log_files/omri/1/main/omri_main_medoc_events.csv')
    events_df = get_events_df('log_files/omri/1/main/omri_main_events.csv')
    plot_df_in_time(events_df,df_name="EVENTS")
    plot_df_in_time(medoc_df, columns_to_plot=["temperature_c"], df_name="MEDOC")
    swir_df = swir_csv_to_df(swir_csv_path=swir_csv_path)
    plot_df_in_time(swir_df, df_name="SWIR")
    eyelink_df = get_eyelink_df(eyelink_file_path)
    plot_df_in_time(eyelink_df, columns_to_plot=["xpos", "ypos", "ps"], df_name="EYELINK")
    

if __name__ == "__main__":
    main()