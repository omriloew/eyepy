import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

import synchronization as sync
import videoProccesing as vp


def main():

    avi_file_path = 'rawFilesToTestWith/1-9_M_30exp.avi'
    edf_file_path = 'out.edf'
    medoc_csv_path = 'output/medoc.csv'
    figures_dir_path = 'output/figures'
    output_dir = 'post_exp_raw_process_results'

    session_type = 'main'
    participant_id = '002'

    events_csv_path = f'log_files/{session_type}/{participant_id}/{participant_id}_{session_type}_events.csv'
    output_csv_sufix = f'{participant_id}_{session_type}.csv'

    paticipant_dir = f'{output_dir}/{participant_id}'
    os.makedirs(paticipant_dir, exist_ok=True)
    
    session_output_dir = f'{paticipant_dir}/{session_type}'
    if not os.path.exists(session_output_dir):
        os.makedirs(session_output_dir)
    else:
        session_number = 1
        while os.path.exists(os.path.join(paticipant_dir, f"{session_type}_{session_number}")):
            session_number += 1
        session_output_dir = os.path.join(paticipant_dir, f"{session_type}_{session_number}")
        os.makedirs(session_output_dir)

    parser = argparse.ArgumentParser(description='Post-experiment raw data processing')
    parser.add_argument('--input_avi', default='', help='Path to input AVI video file')
    parser.add_argument('--output_csv', default='', help='Path to output CSV file')
    parser.add_argument('--fps', type=float, help='Frames per second (default: auto-detect from video)')
    parser.add_argument('--empty-event-label', default='', help='Label for event column (default: "frame")')
    parser.add_argument('--preview', action='store_true', help='Show first few rows of output CSV')
    args = parser.parse_args()

    swir_df = vp.video_to_df(avi_file_path, save_path=f'{session_output_dir}/swir_{output_csv_sufix}')
    light_events_frame_indices = vp.extract_light_events_time(avi_file_path, save_plot_path=f'{session_output_dir}/light_events.png')
        
    led_on_frames = light_events_frame_indices[0]  # LED_ON frame indices
    led_off_frames = light_events_frame_indices[1]  # LED_OFF frame indices
    
    print("--------------------------------")
    print(f"LED_ON frame indices: {led_on_frames}")
    print(f"LED_OFF frame indices: {led_off_frames}")
    print("--------------------------------")
    
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
    swir_df.to_csv(f'{session_output_dir}/swir_{output_csv_sufix}', index=False)
    
    print(f"\nUpdated CSV saved with LED events!")
    print(f"  - LED_ON events: {len(led_on_frames)}")
    print(f"  - LED_OFF events: {len(led_off_frames)}")
    print(f"  - Output file: {f'{session_output_dir}/swir_{output_csv_sufix}'}")
    
    # Show preview of events
    led_events = swir_df[swir_df['event_label'].isin(['LED_ON', 'LED_OFF'])]
    if not led_events.empty:
        print(f"\nLED Events Preview:")
        print(led_events.head(10).to_string(index=False))

    E_df = pd.read_csv(events_csv_path)
    
    devices = {
    }

    synced_df, mappings = sync.synchronize_to_SWIR(
        df_SWIR=swir_df,
        devices=devices,
        df_E=E_df,
        window_ms=5.0
    )

    synced_df.to_csv(f'{session_output_dir}/synced_{output_csv_sufix}', index=False)
    print(f"Synced data saved to: {f'{session_output_dir}/synced_{output_csv_sufix}'}")
    print(f"Mappings: {mappings}")

    

if __name__ == "__main__":
    main()