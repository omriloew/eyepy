import cv2
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import convolve, find_peaks, butter, filtfilt
import time
import os
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable, Iterable, Union
import argparse
import sys


def convert_video_into_frames_series(_video_path):
    frames_series = []
    cap = cv2.VideoCapture(str(_video_path))

    if not cap.isOpened():
        raise Exception(f'the video at: {_video_path}, could not be open')

    progress_bar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT), desc='Converting video to frames')
    while cap.isOpened():
        return_value, frame = cap.read()
        if return_value:
            frames_series.append(frame)
            progress_bar.update()
        else:
            break

    cap.release()
    progress_bar.close()
    return np.array(frames_series)


def convert_images_series_into_video(images_list, save_path,
                                     video_name=None,
                                     fps=31, frame_size=None,
                                     is_color=False):
    if video_name is None:
        current_date_time = time.strftime("%Y.%m.%d-%H.%M")
        video_name = f'{current_date_time}_stitched_video.mp4'
    if frame_size is None:
        frame_size = images_list[0].shape[:2]

    video_out = cv2.VideoWriter(filename=str(Path(save_path, video_name)), fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                fps=fps, frameSize=frame_size, isColor=is_color)
    for frame in images_list:
        # TODO - understand when is necessary
        if not is_color:
            frame = cv2.convertScaleAbs(frame, alpha=2, beta=50)
        video_out.write(frame)
    video_out.release()


def preview_frames(_video_path, frame_start=0, time_start=0.0, num_frames=10,
                   save_path=None, num_cols=5, is_rolling=False):
    num_rows = int(np.ceil(num_frames / num_cols))

    cap = cv2.VideoCapture(str(_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_start > 0 and time_start == 0:
        start_index = frame_start
    elif frame_start == 0 and time_start > 0:
        start_index = int(fps * time_start)
    else:
        start_index = frame_start

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)

    if is_rolling:
        plt.figure(figsize=(7, 7))
        for i in range(num_frames):
            res, frame = cap.read()
            if res:
                frame_index = start_index + i
                plt.title(f'frame index: {frame_index}')
                # plt.imshow(cv2.convertScaleAbs(frame, alpha=2, beta=50), cmap='gray')
                plt.imshow(frame)
                plt.pause(1)  # in seconds
        plt.close()
    else:
        plt.figure(figsize=(19, 10))
        for i in range(num_frames):
            res, frame = cap.read()
            if res:
                frame_index = start_index + i
                plt.subplot(num_rows, num_cols, i + 1)
                plt.title(f'frame index: {frame_index}')
                # plt.imshow(cv2.convertScaleAbs(frame, alpha=2, beta=50), cmap='gray')
                plt.imshow(frame)

        if save_path is not None and Path(save_path).exists():
            plt.savefig(save_path)

        # plt.tight_layout()
        plt.show()


def add_time_marker_on_frames_series(frames, mark_size=10, jump=10):
    mark_value = frames.max() + frames.std()
    num_frames, width, height = frames.shape[:3]
    marked_frames = frames.copy()
    for frame_index in range(num_frames):
        row = ((frame_index // ((width - mark_size) // jump)) % ((height - mark_size) // jump)) * jump
        col = (frame_index % ((width - mark_size) // jump)) * jump
        marked_frames[frame_index, row:row + mark_size, col:col + mark_size] = mark_value

    return marked_frames


def plot_video_mean_brightness(_video_path, return_output=False):
    cap = cv2.VideoCapture(str(_video_path))
    brightness = np.zeros(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    i = 0
    while True:
        res, frame = cap.read()
        if res:
            brightness[i] = frame.mean()
            i += 1
        else:
            break

    b, a = butter(5, 0.1, 'lowpass')
    brightness = filtfilt(b, a, brightness, padtype=None)

    if return_output:
        return brightness

    plt.plot(brightness)
    plt.title('Video Brightness')
    plt.xlabel('frame index')
    plt.ylabel('mean brightness')
    plt.show()


def extract_light_events_time(_video_path, do_plot=True, show_plot=False, save_plot_path=None):
    
    fps = int(cv2.VideoCapture(str(_video_path)).get(cv2.CAP_PROP_FPS))
    start_padding = 3 * fps
    end_padding = 60 * fps

    brightness = plot_video_mean_brightness(_video_path, return_output=True)
    brightness_derivative = convolve(brightness, np.array([1, -1]), 'same')
    brightness_derivative[0] = brightness_derivative[1]
    
    # Adjust padding to fit video length
    total_frames = len(brightness_derivative)
    start_padding = min(start_padding, total_frames // 4)  # Max 25% of video
    end_padding = min(end_padding, total_frames // 2)  # Max 50% of video
    
    # Ensure slice is valid (not empty)
    if start_padding >= total_frames - end_padding:
        start_padding = max(1, total_frames // 10)  # Use 10% if too large
        end_padding = max(1, total_frames // 10)
    
    # Use a smaller slice just for computing statistics (exclude noisy start/end)
    derivative_slicing = slice(start_padding, -end_padding if end_padding > 0 else None)
    mean_brightness = brightness_derivative[derivative_slicing].mean()
    std_brightness = brightness_derivative[derivative_slicing].std()
    # Use higher threshold (3.5*std) to reduce false positives
    brightness_threshold = mean_brightness + 3.5 * std_brightness

    # Search in the FULL array (don't exclude end, just exclude very noisy start)
    derivative_peaks_slicing = slice(start_padding, None)  # Only exclude start, not end
    search_array = brightness_derivative[derivative_peaks_slicing]
    light_duration_and_interval_in_seconds = 3
    frames_peak_to_peak = light_duration_and_interval_in_seconds * fps

    def get_events(_brightness_threshold, event_type='lights_on'):
        while True:
            # Find peaks directly in positive or negative derivative
            # Use smaller distance to allow close peaks, but still prevent noise
            min_distance = max(1, int(fps * 0.5))  # At least 0.5 seconds apart
            
            if event_type == 'lights_on':
                # Find peaks in positive derivative (lights turning on)
                events, _ = find_peaks(search_array,
                                      distance=min_distance,
                                      height=_brightness_threshold,
                                      prominence=0.2 * std_brightness)
            else:  # lights_off
                # Find peaks in negative derivative (inverted, so we look for positive peaks)
                events, _ = find_peaks(-search_array,
                                      distance=min_distance,
                                      height=_brightness_threshold,
                                      prominence=0.2 * std_brightness)

            if events.size >= 3:
                # Adjust indices back to full array coordinates
                return events + start_padding

            # Moderate threshold reduction
            if _brightness_threshold > mean_brightness + 0.5 * std_brightness:
                _brightness_threshold -= 0.15 * std_brightness
            else:
                break

        # Return empty array if no events found
        return np.array([], dtype=int)

    lights_on_events = get_events(brightness_threshold, event_type='lights_on')
    lights_off_events = get_events(brightness_threshold, event_type='lights_off')
    
    # Sort events by frame index and limit to first 6 of each kind
    lights_on_events = np.sort(lights_on_events)
    lights_off_events = np.sort(lights_off_events)
    
    max_events = 6
    if lights_on_events.size > max_events:
        lights_on_events = lights_on_events[:max_events]  # Take first 6
    
    if lights_off_events.size > max_events:
        lights_off_events = lights_off_events[:max_events]  # Take first 6

    if do_plot:
        p1, = plt.plot(brightness_derivative, label='derivative')
        plt.axhline(brightness_threshold, linestyle='--')
        plt.axhline(-brightness_threshold, linestyle='--')
        plt.scatter(lights_on_events, brightness_derivative[lights_on_events], label='lights on')
        plt.scatter(lights_off_events, brightness_derivative[lights_off_events], label='lights off')
        ax = plt.gca()
        twin_ax = ax.twinx()
        p2, = twin_ax.plot(brightness, c='tab:purple', label='brightness')
        twin_ax.legend(loc='lower right')
        twin_ax.tick_params(axis='y', colors=p2.get_color())
        ax.set_title('Video Brightness Derivative')
        ax.set_xlabel('frame index')
        ax.set_ylabel('brightness derivative')
        ax.legend(loc='upper right')
        if save_plot_path is not None:
            plt.savefig(save_plot_path)
        elif show_plot:
            plt.show()

    return lights_on_events, lights_off_events


def extract_light_events_from_intensity(intensity_array, time_stamps, do_plot=True, show_plot=False, 
                                         save_plot_path=None, detection_intervals=None, 
                                         threshold_multiplier=4.5):
    """
    Extract LED ON/OFF events from stat_intensity column (similar to extract_light_events_time but for 1D array).
    
    Args:
        intensity_array: 1D numpy array of intensity values
        time_stamps: 1D numpy array of corresponding time stamps (for calculating fps)
        do_plot: Whether to create a plot
        show_plot: Whether to display the plot
        save_plot_path: Optional path to save the plot
        detection_intervals: List of tuples [(start1, end1), (start2, end2), ...] specifying time ranges
                           (in seconds) where LED events should be detected. Events outside these intervals
                           will be ignored. If None, detects events in the entire signal.
        threshold_multiplier: Multiplier for the detection threshold (higher = more strict, default 4.5)
    
    Returns:
        Tuple of (lights_on_indices, lights_off_indices) as numpy arrays
    """
    # Calculate approximate fps from time stamps
    if len(time_stamps) > 1:
        avg_dt = np.mean(np.diff(time_stamps))
        fps = 1.0 / avg_dt if avg_dt > 0 else 30.0  # Default to 30 if can't calculate
    else:
        fps = 30.0
    
    start_padding = int(3 * fps)
    end_padding = int(60 * fps)
    
    # Apply lowpass filter to intensity (similar to brightness processing)
    b, a = butter(5, 0.1, 'lowpass')
    intensity_filtered = filtfilt(b, a, intensity_array, padtype=None)
    
    # Calculate derivative
    intensity_derivative = convolve(intensity_filtered, np.array([1, -1]), 'same')
    intensity_derivative[0] = intensity_derivative[1]
    
    # Adjust padding to fit array length
    total_frames = len(intensity_derivative)
    start_padding = min(start_padding, total_frames // 4)  # Max 25% of array
    end_padding = min(end_padding, total_frames // 2)  # Max 50% of array
    
    # Ensure slice is valid (not empty)
    if start_padding >= total_frames - end_padding:
        start_padding = max(1, total_frames // 10)  # Use 10% if too large
        end_padding = max(1, total_frames // 10)
    
    # Create a mask for valid detection intervals
    total_frames = len(intensity_derivative)
    valid_mask = np.zeros(total_frames, dtype=bool)
    
    if detection_intervals is not None and len(detection_intervals) > 0:
        # Convert time intervals to frame indices
        for interval_start, interval_end in detection_intervals:
            # Find frame indices within this interval
            interval_mask = (time_stamps >= interval_start) & (time_stamps <= interval_end)
            valid_mask |= interval_mask
    else:
        # If no intervals specified, use default padding (exclude noisy start/end)
        valid_mask[start_padding:total_frames - end_padding] = True
    
    # Find the first and last valid indices for statistics calculation
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        # Fallback: use default padding if no valid intervals
        valid_mask[start_padding:total_frames - end_padding] = True
        valid_indices = np.where(valid_mask)[0]
    
    # Use valid region for computing statistics
    mean_intensity = intensity_derivative[valid_indices].mean()
    std_intensity = intensity_derivative[valid_indices].std()
    # Use configurable threshold multiplier (default 4.5, higher = more strict)
    intensity_threshold = mean_intensity + threshold_multiplier * std_intensity
    
    # Search only in valid intervals
    search_array = intensity_derivative[valid_mask]
    search_indices = np.where(valid_mask)[0]  # Original indices for mapping back
    light_duration_and_interval_in_seconds = 3
    frames_peak_to_peak = light_duration_and_interval_in_seconds * fps
    
    def get_events(_intensity_threshold, event_type='lights_on'):
        while True:
            # Find peaks directly in positive or negative derivative
            # Use smaller distance to allow close peaks, but still prevent noise
            min_distance = max(1, int(fps * 0.5))  # At least 0.5 seconds apart
            
            if event_type == 'lights_on':
                # Find peaks in positive derivative (lights turning on)
                events, _ = find_peaks(search_array,
                                      distance=min_distance,
                                      height=_intensity_threshold,
                                      prominence=0.2 * std_intensity)
            else:  # lights_off
                # Find peaks in negative derivative (inverted, so we look for positive peaks)
                events, _ = find_peaks(-search_array,
                                      distance=min_distance,
                                      height=_intensity_threshold,
                                      prominence=0.2 * std_intensity)
            
            if events.size >= 3:
                # Map events back to original array indices
                return search_indices[events]
            
            # Moderate threshold reduction
            if _intensity_threshold > mean_intensity + 0.5 * std_intensity:
                _intensity_threshold -= 0.15 * std_intensity
            else:
                break
        
        # Return empty array if no events found
        return np.array([], dtype=int)
    
    lights_on_events = get_events(intensity_threshold, event_type='lights_on')
    lights_off_events = get_events(intensity_threshold, event_type='lights_off')
    
    # Sort events by index and limit to first 6 of each kind
    lights_on_events = np.sort(lights_on_events)
    lights_off_events = np.sort(lights_off_events)
    
    max_events = 6
    if lights_on_events.size > max_events:
        lights_on_events = lights_on_events[:max_events]  # Take first 6
    
    if lights_off_events.size > max_events:
        lights_off_events = lights_off_events[:max_events]  # Take first 6
    
    if do_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Intensity and detected events over time
        time_array = time_stamps
        ax1.plot(time_array, intensity_filtered, 'b-', label='Intensity (filtered)', linewidth=1.5, alpha=0.7)
        ax1.plot(time_array, intensity_array, 'gray', label='Intensity (raw)', linewidth=0.5, alpha=0.3)
        
        # Mark detection intervals (valid regions) and ignored regions
        if detection_intervals is not None and len(detection_intervals) > 0:
            # Show valid detection intervals in green
            for i, (interval_start, interval_end) in enumerate(detection_intervals):
                ax1.axvspan(interval_start, interval_end, alpha=0.15, color='green', 
                           label='Detection Interval' if i == 0 else '')
            
            # Show ignored regions (everything outside intervals) in yellow
            intervals_sorted = sorted(detection_intervals)
            # Before first interval
            if intervals_sorted[0][0] > time_array[0]:
                ax1.axvspan(time_array[0], intervals_sorted[0][0], alpha=0.2, color='yellow', 
                           label='Ignored Region')
            # Between intervals
            for i in range(len(intervals_sorted) - 1):
                if intervals_sorted[i][1] < intervals_sorted[i+1][0]:
                    ax1.axvspan(intervals_sorted[i][1], intervals_sorted[i+1][0], alpha=0.2, color='yellow')
            # After last interval
            if intervals_sorted[-1][1] < time_array[-1]:
                ax1.axvspan(intervals_sorted[-1][1], time_array[-1], alpha=0.2, color='yellow')
        else:
            # If no intervals specified, show default padding regions as ignored
            if start_padding > 0:
                ignore_start_time = time_array[start_padding] if start_padding < len(time_array) else time_array[0]
                ax1.axvspan(time_array[0], ignore_start_time, alpha=0.2, color='yellow', label='Ignored (start)')
            if end_padding > 0:
                ignore_end_time = time_array[-end_padding] if end_padding < len(time_array) else time_array[-1]
                ax1.axvspan(ignore_end_time, time_array[-1], alpha=0.2, color='yellow', label='Ignored (end)')
        
        # Mark LED_ON events
        if len(lights_on_events) > 0:
            led_on_times = time_array[lights_on_events]
            led_on_intensities = intensity_filtered[lights_on_events]
            ax1.scatter(led_on_times, led_on_intensities, c='green', s=150, marker='^', 
                       label=f'LED_ON ({len(lights_on_events)} events)', zorder=5, edgecolors='black', linewidths=1.5)
            for i, (t, val) in enumerate(zip(led_on_times, led_on_intensities)):
                ax1.annotate(f'ON{i+1}', (t, val), xytext=(5, 10), textcoords='offset points', 
                           fontsize=9, fontweight='bold', color='green')
        
        # Mark LED_OFF events
        if len(lights_off_events) > 0:
            led_off_times = time_array[lights_off_events]
            led_off_intensities = intensity_filtered[lights_off_events]
            ax1.scatter(led_off_times, led_off_intensities, c='red', s=150, marker='v', 
                       label=f'LED_OFF ({len(lights_off_events)} events)', zorder=5, edgecolors='black', linewidths=1.5)
            for i, (t, val) in enumerate(zip(led_off_times, led_off_intensities)):
                ax1.annotate(f'OFF{i+1}', (t, val), xytext=(5, -15), textcoords='offset points', 
                           fontsize=9, fontweight='bold', color='red')
        
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Intensity', fontsize=12)
        ax1.set_title('LED Events Detection from SWIR Intensity', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Derivative with thresholds
        ax2.plot(time_array, intensity_derivative, 'b-', label='Intensity Derivative', linewidth=1.5)
        ax2.axhline(intensity_threshold, linestyle='--', color='orange', linewidth=2, label=f'Threshold (+{intensity_threshold:.2f})')
        ax2.axhline(-intensity_threshold, linestyle='--', color='orange', linewidth=2, label=f'Threshold (-{intensity_threshold:.2f})')
        ax2.axhline(0, linestyle='-', color='black', linewidth=0.5, alpha=0.3)
        
        # Mark detection intervals and ignored regions in derivative plot too
        if detection_intervals is not None and len(detection_intervals) > 0:
            # Show valid detection intervals in green
            for interval_start, interval_end in detection_intervals:
                ax2.axvspan(interval_start, interval_end, alpha=0.15, color='green')
            
            # Show ignored regions in yellow
            intervals_sorted = sorted(detection_intervals)
            if intervals_sorted[0][0] > time_array[0]:
                ax2.axvspan(time_array[0], intervals_sorted[0][0], alpha=0.2, color='yellow')
            for i in range(len(intervals_sorted) - 1):
                if intervals_sorted[i][1] < intervals_sorted[i+1][0]:
                    ax2.axvspan(intervals_sorted[i][1], intervals_sorted[i+1][0], alpha=0.2, color='yellow')
            if intervals_sorted[-1][1] < time_array[-1]:
                ax2.axvspan(intervals_sorted[-1][1], time_array[-1], alpha=0.2, color='yellow')
        else:
            # Default padding regions
            if start_padding > 0:
                ignore_start_time = time_array[start_padding] if start_padding < len(time_array) else time_array[0]
                ax2.axvspan(time_array[0], ignore_start_time, alpha=0.2, color='yellow')
            if end_padding > 0:
                ignore_end_time = time_array[-end_padding] if end_padding < len(time_array) else time_array[-1]
                ax2.axvspan(ignore_end_time, time_array[-1], alpha=0.2, color='yellow')
        
        # Mark detected peaks in derivative
        if len(lights_on_events) > 0:
            led_on_times = time_array[lights_on_events]
            led_on_derivatives = intensity_derivative[lights_on_events]
            ax2.scatter(led_on_times, led_on_derivatives, c='green', s=150, marker='^', 
                      label='LED_ON peaks', zorder=5, edgecolors='black', linewidths=1.5)
        
        if len(lights_off_events) > 0:
            led_off_times = time_array[lights_off_events]
            led_off_derivatives = intensity_derivative[lights_off_events]
            ax2.scatter(led_off_times, led_off_derivatives, c='red', s=150, marker='v', 
                       label='LED_OFF peaks', zorder=5, edgecolors='black', linewidths=1.5)
        
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Intensity Derivative', fontsize=12)
        ax2.set_title('Derivative with Detection Thresholds', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        elif save_plot_path is not None:
            plt.savefig(save_plot_path, dpi=150, bbox_inches='tight')
            print(f"LED events plot saved to: {save_plot_path}")
            plt.close()
        else:
            plt.close()
    
    return lights_on_events, lights_off_events


def play_video_with_frame_indices(_video_path, indices_location=(15, 35),
                                  font_scale=1,
                                  callbacks=None):
    video_name = Path(_video_path).name

    def apply_callbacks(_frame, _callbacks: Optional[Union[Callable, Iterable[Callable]]]) -> np.ndarray:
        if _callbacks is not None:
            if isinstance(_callbacks, (list, tuple)):
                for callback in _callbacks:
                    _frame = callback(_frame)
            else:
                _frame = _callbacks(_frame)
        return _frame

    def get_frame_time_representation(frame_index: int, _fps: int, with_hours: bool = True) -> str:
        time_in_ms = 1000 / _fps * frame_index
        datetime_obj = datetime.fromtimestamp(time_in_ms / 1000.0, tz=timezone.utc)
        hours = (datetime_obj.day - 1) * 24 + datetime_obj.hour
        if with_hours:
            return f'{hours:02}:{datetime_obj:%M:%S.%f}'[:-3]
        minutes = hours * 60 + datetime_obj.minute
        return f'{minutes:02}:{datetime_obj:%S.%f}'[:-3]

    def show_frame(_frame: np.ndarray, frame_index: Optional[int] = None, num_frames: Optional[int] = None,
                   fps: Optional[int] = None) -> None:
        if frame_index is not None:
            height, width, depth = _frame.shape
            header = np.zeros((max([int(height * 0.10), 20]), width, depth), dtype='uint8')
            _frame = np.vstack([header, _frame])

            (x, y) = indices_location
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 0, 0)  # Blue color in BGR
            thickness = 2
            frame_description = str(frame_index) if num_frames is None else str(frame_index).zfill(len(str(num_frames)))
            if num_frames is not None:
                frame_description += f'/{num_frames - 1}'
                if fps is not None:
                    with_hours = (frame_index / fps) >= 60 * 60
                    frame_time_description = f'{get_frame_time_representation(frame_index, fps, with_hours)}/{get_frame_time_representation(num_frames - 1, fps, with_hours)}'
                    cv2.putText(_frame, frame_time_description, (200 + x, y), font, font_scale, color, thickness)
            cv2.putText(_frame, frame_description, (x, y), font, font_scale, color, thickness)
        cv2.imshow(video_name, _frame)

    def get_current_frame_index(_cap: cv2.VideoCapture) -> int:
        return int(_cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

    def backward(_cap: cv2.VideoCapture, jump_size: int = 1) -> None:
        current_frame_index = get_current_frame_index(_cap)
        _cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index - jump_size)

    def forward(_cap: cv2.VideoCapture, jump_size: int = 1) -> None:
        current_frame_index = get_current_frame_index(_cap)
        _cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index + jump_size)

    def controls(_cap: cv2.VideoCapture, _fps: int, _is_paused: bool, _did_quit: bool) -> tuple[bool, bool, bool]:
        _refresh = False

        key = cv2.waitKeyEx(1000 // _fps) & 0xFF
        if key == ord(' '):
            return not _is_paused, _did_quit, _refresh
        elif key in (ord('a'), ord('A')):
            backward(_cap)
            _refresh = True
        elif key in (ord('d'), ord('D')):
            forward(_cap)
            _refresh = True
        elif key in (ord('s'), ord('S')):
            backward(_cap, 10)
            _refresh = True
        elif key in (ord('w'), ord('W')):
            forward(_cap, 10)
            _refresh = True
        elif key in [ord('q'), ord('Q'), 27]:  # 27 -> Esc
            _did_quit = True
        return _is_paused, _did_quit, _refresh

    # read and show video
    cap = cv2.VideoCapture(str(_video_path))

    # validate the video was opened
    if not cap.isOpened():
        raise Exception(f'the video at: {_video_path}, could not be open')

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(video_name, frames_width, frames_height)

    is_paused, did_quit = False, False
    while cap.isOpened() and not did_quit:
        return_value, frame = cap.read()
        if return_value:
            frame = apply_callbacks(frame, callbacks)
            show_frame(frame.copy(), get_current_frame_index(cap), frames_count, fps)

            is_paused, did_quit, refresh = controls(cap, fps, is_paused, did_quit)
            while is_paused and not did_quit:
                if refresh:
                    return_value, frame = cap.read()
                    if return_value:
                        frame = apply_callbacks(frame, callbacks)
                        show_frame(frame.copy(), get_current_frame_index(cap), frames_count, fps)
                is_paused, did_quit, refresh = controls(cap, fps, is_paused, did_quit)
        else:
            break

    # print summary messages and close all
    cap.release()
    cv2.destroyAllWindows()
    print(f'there were {frames_count} frames, which means for {fps}fps, the video took: {frames_count / fps}sec')


def measure_distances(_video_path, selected_frame=None):
    def print_click_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'{x=}, {y=}')

    cap = cv2.VideoCapture(_video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if selected_frame is None:
        selected_frame = np.random.randint(num_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
    ret, frame = cap.read()
    window_name = f'frame {selected_frame} display'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)

    while True:
        key = cv2.waitKey(25)
        if key in [ord('q'), ord('Q'), 27]:  # 27 -> Esc
            break
        cv2.setMouseCallback(window_name, print_click_coordinates)

    cap.release()
    cv2.destroyAllWindows()

def video_to_df(video_path, fps=None, event_label="", save_path=None):
    """
    Convert AVI video to CSV with time_stamp, event_label, frame_index columns.
    
    Args:
        video_path (str): Path to input AVI video file
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
    df.to_csv(save_path, index=False)
    
    print(f"\nConversion completed!")
    print(f"  - Output file: {save_path}")
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
    parser.add_argument('--event-label', default='', help='Label for event column (default: "frame")')
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

        
        df2 = extract_light_events_time(
            _video_path=args.input_video,
            do_plot=True
        )
        print("--------------------------------")
        print(f"Light events time: {df2[0]}")
        print(f"Light events time: {df2[1]}")
        print(f"Light events time: {df2}")
        print("--------------------------------")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

if __name__ == '__main__':
    # video_path = r"H:\DavidH\DLC\projects\Network_metrics-MichalT-2023-11-28\videos_to_analayze\AA91_PLR_2\video_of_test_images_testsize_0.2_numframes_1.mp4"
    # video_path = r"H:\Omer_DLC\PLR_SWIR_exp-omer-2023-06-19\videos\DH_PLR_1.avi"
    video_path = r"rawFilesToTestWith/1-9_M_30exp.avi"
    #location_path = r"H:\DavidH\DLC\projects\Network_metrics-MichalT-2023-11-28\analyzed_videos\video_of_test_images_testsize_0.2_numframes_1DLC_resnet101_Network_metricsNov28shuffle1_30000.h5"
    #video_out_path = r"H:\DavidH\DLC\projects\Network_metrics-MichalT-2023-11-28\videos_to_analayze\0.2 test\video_of_test_images_testsize_0.2_numframes_1_output.mp4"

    # draw_pupil_metrics_over_video(video_path, location_path)

    # preview_frames(video_path, 300, num_frames=100, num_cols=10)
    # preview_frames(video_path, 300, num_frames=10, is_rolling=True)

    # plot_video_mean_brightness(video_path)
    # print(extract_light_events_time(video_path, do_plot=True))
    #video_path = r"H:\DavidH\PupilSizeTrackerNetwork\Videos\PLR\OBB_PLR_1.avi"
    #video_path = r"H:\DavidH\DLC\projects\Gray_cropped-DavidH-2024-07-17\analyzed_videos\auditory_expiriment_trial_OBB03.avi"
    #measure_distances(video_path)
    # play_video_with_frame_indices(r"H:\DavidH\PupilSizeTrackerNetwork\Videos\Gray\OBB91_PLR_2.avi")
    play_video_with_frame_indices(video_path)