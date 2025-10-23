import numpy as np
from pathlib import Path
from psychopy import core, visual
from psychopy.event import getKeys
from random import randrange
from screeninfo import get_monitors
import serial  # pip install pyserial
import serial.tools.list_ports as list_ports
import sys
import time
from typing import Union


def get_monitor_size(screen_index):
    available_monitors = get_monitors()
    selected_screen = available_monitors[screen_index]
    return np.array((selected_screen.width, selected_screen.height), dtype=int)


def check_quit():
    keys = getKeys()
    if 'q' in [key.lower() for key in keys]:
        sys.exit('IMPORTANT: EXIT CODE DUE TO USER INTERRUPTION !!!')


def append_time_event(message, _clock, _clock_times):
    print(message)
    _clock_times.append(message)
    _clock_times.append(str(_clock.getTime()))


def light_led(duration_in_sec, port_name=list_ports.comports()[0].device, baud_rate=9800, timeout=1):
#def light_led(duration_in_sec, port_name, baud_rate=9800, timeout=1):
    ser = serial.Serial(port_name, baud_rate, timeout=timeout)
    for _ in range(duration_in_sec):
        ser.writelines(b'H')  # send a byte
        time.sleep(0.5)  # wait 0.5 seconds
        ser.writelines(b'L')  # send a byte
        time.sleep(0.5)
    ser.close()


def led_and_update_times(duration_in_sec, _clock, _clock_times):
    append_time_event('LED is ON', _clock, _clock_times)
    light_led(duration_in_sec)
    append_time_event('LED is OFF', _clock, _clock_times)


def draw_stimulus(_window: visual.Window, stimulus: Union[visual.ImageStim, visual.TextStim], duration: float):
    stimulus.draw()
    _window.flip()
    core.wait(duration)


def exp_and_frames_rec(_clock: core.Clock, time_output_dir: Union[str, Path], screen_index: int = 0) -> None:
    window = visual.Window(fullscr=True, units='pix', screen=screen_index)
    screen_size = get_monitor_size(screen_index)
    window.recordFrameIntervals = True

    try_to_stay_focused_msg = visual.TextStim(window, text="X", anchorHoriz='center', anchorVert='center',
                                              antialias=False, languageStyle='RTL')
    open_your_beautiful_msg = visual.TextStim(window, text="Without Moving, Please Open Your Eyes", anchorHoriz='center', anchorVert='center',
                                              antialias=False, languageStyle='RTL')

    black_stim_path = np.full((*screen_size[::-1], 3), 0)
    black_stim = visual.ImageStim(window, image=black_stim_path, size=screen_size)

    white_stim_path = np.full((*screen_size[::-1], 3), 255)
    white_stim_path = r"D:\user\Desktop\michal\greys\255.jpg"
    white_stim = visual.ImageStim(window, image=white_stim_path, size=screen_size)

    clock_times = []

    # start_exp
    _clock.reset()
    append_time_event('try to stay focused starts', _clock, clock_times)
    draw_stimulus(window, try_to_stay_focused_msg, 3.0)

    num_led_flashes = 3
    for i in range(num_led_flashes):
        led_and_update_times(3, _clock, clock_times)
        core.wait(3.0)

    append_time_event('try to stay focused ends', _clock, clock_times)
    check_quit()

    initial_dark_habituation_duration = 20
    append_time_event('Black_Time_start', _clock, clock_times)
    draw_stimulus(window, black_stim, initial_dark_habituation_duration)
    append_time_event('Black_Time_end', _clock, clock_times)

    no_trials = 10
    for trial_index in np.arange(no_trials) + 1:
        check_quit()
        print(f"DEBUG: Current Trial - {trial_index}")

        white_screen_duration_seconds = 2
        append_time_event('White_Time_start', _clock, clock_times)
        white_stim_path = r"D:\user\Desktop\michal\greys\255.jpg"
        white_stim = visual.ImageStim(window, image=white_stim_path, size=screen_size)
        draw_stimulus(window, white_stim, white_screen_duration_seconds)
        append_time_event('White_Time_end', _clock, clock_times)

        check_quit()
        append_time_event('Black_Time_start', _clock, clock_times)
        black_stimulus_duration = randrange(18, 22)
        #black_stimulus_duration = 20
        draw_stimulus(window, black_stim, black_stimulus_duration)
        append_time_event('Black_Time_end', _clock, clock_times)

    check_quit()
    append_time_event('open_your_beautiful_eyes_start', _clock, clock_times)
    draw_stimulus(window, open_your_beautiful_msg, 3.0)
    append_time_event('open_your_beautiful_eyes_end', _clock, clock_times)

    window.close()
    Path(time_output_dir, TIMES_FILE_NAME).write_text('\n'.join(clock_times))




if __name__ == '__main__':
    TIMES_FILE_NAME = '11-8_old_newlens_K.txt'
    OUTPUT_DIR = Path(r'D:\user\Pupil experiments\Kfir')
    SCREEN_INDEX = 1

    clock = core.Clock()
    exp_and_frames_rec(clock, OUTPUT_DIR, SCREEN_INDEX)
