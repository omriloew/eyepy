import pandas as pd
import matplotlib.pyplot as plt
import os
import coloredPrint as cp
import synchronization as sync
import videoProccesing as vp
import config as config
import edfHandler as edf

always_keep_cols = ['time_stamp', 'event_label'] #DO NOT CHANGE THIS
#===============================================
# events configuration
#===============================================
events_csv_path = 'plr/omri_plr_events.csv'
keep_events_cols = ['_event_code']
#===============================================
# swir configuration
#===============================================
process_swir = True
swir_avi_file_path = 'plr/output.avi'
keep_swir_cols = ['event_label']
#===============================================
# medoc configuration
#===============================================
process_medoc = False
medoc_csv_path = 'plr/omri_plr_medoc_events.csv'
keep_medoc_cols = ['temperature_c']
#===============================================
# eeg configuration
#===============================================
process_eeg = False
eeg_file_path = 'rawFilesToTestWith/STM_Visualization_Clean_v2_EN.ipynb.edf'
keep_eeg_cols = []
#===============================================
# eyelink configuration 
#===============================================
process_eyelink = True
eyelink_file_path = 'plr/test.edf'
keep_eyelink_cols = ['xpos', 'ypos', 'ps']
#===============================================
# output configuration
#===============================================
figures_dir_path = 'output/figures'
output_dir = 'post_exp_raw_process_results'
#===============================================
# session configuration
#===============================================
session_type = 'main'
participant_id = 'omri'
session_number = '1'
#===============================================
# synchronization configuration
#===============================================
sychronize_to = 'SWIR'
#===============================================
# visualization configuration
#===============================================
columns_to_exclude_from_plot = ['frame_index']  # Columns to exclude from final visualization


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
    return events_csv_path, medoc_csv_path, eeg_file_path, eyelink_file_path, swir_avi_file_path

def swir_avi_to_df(session_output_dir=None, output_csv_sufix=None, swir_avi_file_path=None):
    print("================================================")
    cp.colored_print("== Converting SWIR AVI to DataFrame ==", color=cp.Fore.CYAN)
    # Convert the video to a DataFrame
    save_path = f'{session_output_dir}/swir_{output_csv_sufix}' if session_output_dir and output_csv_sufix else None
    swir_df = vp.video_to_df(swir_avi_file_path, save_path=save_path)
    # Extract light events from the video
    save_plot_path = f'{session_output_dir}/light_events.png' if session_output_dir else None
    light_events_frame_indices = vp.extract_light_events_time(swir_avi_file_path, save_plot_path=save_plot_path)
    led_on_frames = light_events_frame_indices[0]  # LED_ON frame indices
    led_off_frames = light_events_frame_indices[1]  # LED_OFF frame indices
    cp.colored_print(f"LED_ON frame indices: {led_on_frames}", color=cp.Fore.BLUE)
    cp.colored_print(f"LED_OFF frame indices: {led_off_frames}", color=cp.Fore.BLUE)
    # Add LED event labels to the DataFrame
    swir_df['event_label'] = swir_df['event_label'].astype(str)  # Ensure it's string type
    # Mark LED_ON events
    for frame_idx in led_on_frames:
        if frame_idx < len(swir_df):
            swir_df.loc[frame_idx, 'event_label'] = "LED_ON"
    # Mark LED_OFF events  
    for frame_idx in led_off_frames:
        if frame_idx < len(swir_df):
            swir_df.loc[frame_idx, 'event_label'] = "LED_OFF"
    # Save the updated CSV with LED events
    if session_output_dir and output_csv_sufix:
        swir_df.to_csv(f'{session_output_dir}/swir_{output_csv_sufix}', index=False)
        cp.colored_print(f"saved file: {f'{session_output_dir}/swir_{output_csv_sufix}'}", color=cp.Fore.BLUE)
    
    # Show preview of events
    led_events = swir_df[swir_df['event_label'].isin(['LED_ON', 'LED_OFF'])]
    if not led_events.empty:
        cp.colored_print(f"LED Events Preview:", color=cp.Fore.CYAN)
        cp.colored_print(led_events.head(10).to_string(index=False), color=cp.Fore.BLUE)
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
        
        # Visualize the raw Eyelink data
        plot_save_path = f'{session_output_dir}/eyelink_raw_plot_{output_csv_sufix}'.replace('.csv', '.png')
        plot_df_in_time(df, columns_to_plot=['xpos', 'ypos', 'ps'], df_name="EYELINK RAW", save_path=plot_save_path)
        cp.colored_print(f"Raw Eyelink plot saved to: {plot_save_path}", color=cp.Fore.BLUE)
    print("================================================")
    return df

def get_eeg_df(eeg_file_path):
    print("================================================")
    cp.colored_print("== Converting EEG EDF to DataFrame ==", color=cp.Fore.CYAN)
    df = edf.edf_to_df(eeg_file_path)
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

def plot_df_in_time(df, columns_to_plot=[], scale_factors=None, df_name="DATAFRAME", save_path=None):
    """
    Plots specified columns against time on a single graph.
    
    Args:
        df: DataFrame with 'time_stamp' and optionally 'event_label' columns
        columns_to_plot: List of column names to plot
        scale_factors: Dict mapping column name -> float factor to scale the values
        df_name: Name for the plot title
        save_path: Optional path to save the plot as PNG (if None, shows the plot)
    
    Behavior:
        - Always plots time_stamp on x-axis
        - Always plots event_label as discrete markers (if present)
        - Columns starting with '_' are plotted as discrete (markers where values exist)
        - Other columns are plotted as continuous (lines)
        - All plots on the same graph
    """
    if scale_factors is None:
        scale_factors = {}
    
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
        values = df[col] * scale
        ax.plot(df[time_col], values, label=f"{col} (×{scale:.2f})" if scale != 1.0 else col, 
                linewidth=1.5, alpha=0.8)
    
    # Plot discrete columns as markers
    for col in discrete_cols:
        scale = scale_factors.get(col, 1.0)
        # Get rows where this column has non-null/non-zero values
        discrete_df = df[df[col].notna() & (df[col] != 0)].copy()
        if len(discrete_df) > 0:
            values = discrete_df[col] * scale
            ax.scatter(discrete_df[time_col], values, 
                      marker='o', s=50, alpha=0.6, 
                      label=f"{col} (×{scale:.2f})" if scale != 1.0 else col,
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
    events_csv_path, medoc_csv_path, eeg_file_path, eyelink_file_path, swir_avi_file_path = configure_input_paths()
    devices = {}
    if process_swir:
        swir_df = swir_avi_to_df(session_output_dir, output_csv_sufix, swir_avi_file_path)
        devices["SWIR"] = swir_df
    if process_eyelink:
        eyelink_df = get_eyelink_df(eyelink_file_path, session_output_dir, output_csv_sufix)
        devices["EL"] = eyelink_df
    if process_eeg:
        eeg_df = get_eeg_df(eeg_file_path)
        devices["EEG"] = eeg_df
    if process_medoc:
        medoc_df = get_medoc_df(medoc_csv_path)
        devices["MEDOC"] = medoc_df

    events_df = get_events_df(events_csv_path)

    synced_df, mappings = sync.synchronize_to_reference(
        reference_name=sychronize_to,
        devices=devices,
        df_E=events_df,
        window_ms=30.0
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
    
    plot_save_path = f'{session_output_dir}/synced_combined_plot_{output_csv_sufix}'.replace('.csv', '.png')
    plot_df_in_time(synced_df, columns_to_plot=plot_cols, df_name="SYNCED COMBINED", save_path=plot_save_path)
    
    print(f"Mappings: {mappings}")


def test():
    medoc_df = get_medoc_df('log_files/omri/1/main/omri_main_medoc_events.csv')
    events_df = get_events_df('log_files/omri/1/main/omri_main_events.csv')
    plot_df_in_time(events_df,df_name="EVENTS")
    plot_df_in_time(medoc_df, columns_to_plot=["temperature_c"], df_name="MEDOC")
    swir_df = swir_avi_to_df(swir_avi_file_path=swir_avi_file_path)
    plot_df_in_time(swir_df, df_name="SWIR")
    eyelink_df = get_eyelink_df(eyelink_file_path)
    plot_df_in_time(eyelink_df, columns_to_plot=["xpos", "ypos", "ps"], df_name="EYELINK")
    

if __name__ == "__main__":
    main()