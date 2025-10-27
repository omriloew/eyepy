#!/usr/bin/env python3
"""
Video to CSV Converter

This script converts an AVI video file to a CSV file with the following columns:
- time_stamp: Timestamp of each frame in seconds
- event_label: Label for the event (can be customized)
- frame_index: Index of the frame (0-based)

Usage:
    python videoToCsv.py input_video.avi output.csv [--fps FPS] [--event-label LABEL]
"""

import cv2
import pandas as pd
import argparse
import os
import sys
from datetime import datetime
from tqdm import tqdm

def video_to_csv(video_path, output_path, fps=None, event_label=""):
    """
    Convert AVI video to CSV with time_stamp, event_label, frame_index columns.
    
    Args:
        video_path (str): Path to input AVI video file
        output_path (str): Path to output CSV file
        fps (float, optional): Frames per second. If None, will be detected from video
        event_label (str): Label for the event column
    """
    
    # Check if input file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    # Use provided fps or detected fps
    if fps is None:
        fps = video_fps
    
    print(f"Video properties:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Video FPS: {video_fps:.2f}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Using FPS: {fps:.2f}")
    
    # Prepare data lists
    timestamps = []
    event_labels = []
    frame_indices = []
    
    # Create progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Calculate timestamp
        timestamp = len(timestamps) / fps
        
        # Add data
        timestamps.append(timestamp)
        event_labels.append(event_label)
        frame_indices.append(len(timestamps) - 1)
        
        # Update progress bar
        progress_bar.update(1)
    
    # Close progress bar and release video capture
    progress_bar.close()
    cap.release()
    
    # Create DataFrame
    df = pd.DataFrame({
        'time_stamp': timestamps,
        'event_label': event_labels,
        'frame_index': frame_indices
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\nConversion completed!")
    print(f"  - Output file: {output_path}")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Time range: {df['time_stamp'].min():.3f}s - {df['time_stamp'].max():.3f}s")
    
    return df


def main():
    """Main function to handle command line arguments and run conversion."""
    
    parser = argparse.ArgumentParser(
        description="Convert AVI video to CSV with time_stamp, event_label, frame_index columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python videoToCsv.py input.avi output.csv
  python videoToCsv.py input.avi output.csv --fps 30
  python videoToCsv.py input.avi output.csv --event-label "stimulus"
        """
    )
    
    parser.add_argument('input_video', help='Path to input AVI video file')
    parser.add_argument('output_csv', help='Path to output CSV file')
    parser.add_argument('--fps', type=float, help='Frames per second (default: auto-detect from video)')
    parser.add_argument('--event-label', default='frame', help='Label for event column (default: "frame")')
    parser.add_argument('--preview', action='store_true', help='Show first few rows of output CSV')
    
    args = parser.parse_args()
    
    try:
        # Convert video to CSV
        df = video_to_csv(
            video_path=args.input_video,
            output_path=args.output_csv,
            fps=args.fps,
            event_label=args.event_label
        )
        
        # Show preview if requested
        if args.preview:
            print(f"\nPreview of output CSV:")
            print(df.head(10).to_string(index=False))
            print(f"\n... (showing first 10 rows of {len(df)} total rows)")

        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
