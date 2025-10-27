
# ==============================================
# MEDOC PARAMETERS
# ==============================================
medoc_on = 0
medoc_host = '10.101.119.124'
medoc_port = 20121
medoc_default_program = '00010010'
medoc_programs = {
    'default': medoc_default_program,
    'main_1': '00010001', 
    'main_2': '00010010', 
    'main_3': '00010011',
    'main_4': '00010100', 
    'main_5': '00010101',
    'main_demo': '00010001',
    'threshold': '00010001',
    'threshold_demo': '00010001',}

# ==============================================
# VIDEO CAMERA PARAMETERS
# ==============================================
video_camera_on = 0
vc_fps = 30

# ==============================================
# IR CAMERA PARAMETERS
# ==============================================
IR_camera_on = 0
IR_fps = 30

# ==============================================
# LED PARAMETERS
# ==============================================
led_on = True
led_baud_rate = 9800
led_timeout = 1
led_on_duration_in_sec = 3
led_num_flashes = 3
led_interval_in_sec = 0.3

# ==============================================
# EYELINK PARAMETERS
# ==============================================
eyelink_on = 0

# ==============================================
# EEG PARAMETERS
# ==============================================
eeg_on = 0
eeg_port_address = 0x03EFC


# ==============================================
# experiment parameters
# ==============================================
full_screen = False
main_trails_number = 5
threshold_trails_number = 4
main_trail_duration = 5
demo_trail_duration = 2
use_vas = True
timeout = 100 #8   #skip question if too long in seconds
is_hebrew = 0
threshold_demo_trails_number = 2
main_demo_trails_number = 2

# ==============================================
# VAS (Visual Analog Scale) parameters
# ==============================================
vas_show_value = True  # Show the current numeric value while choosing
vas_max = 10  # Maximum value on the scale
vas_min = 0  # Minimum value on the scale
vas_granularity = 1  # Step size for the scale
vas_keyboard_step = 1  # How much to jump when using arrow keys
vas_start_value = 5  # Starting position of the slider
vas_timeout = None  # Timeout in seconds
vas_marker_style = 'slider'  # Options: 'slider', 'circle', 'triangle', 'glow'
vas_marker_color = 'red'  # Color of the marker/cursor

# ==============================================
# intervals generation parameters
# ==============================================
intervals_generation_method = "exponential"
intervals_mean = 1
intervals_std = 0.5
intervals_min_time = 0.5
intervals_max_time = 3

# ==============================================
# folder paths
# ==============================================
folder_laptop = r'/Users/omrilo/Desktop/school/EyeDoc/examples/code_backup'
datapath = 'log_files'  # directory to save data
eyelink_edf_prefix = 'eyelink_'
vc_csv_prefix = 'VC_'
el_csv_prefix = 'EL_'
md_csv_prefix = 'MD_'
ir_csv_prefix = 'IR_'

# ==============================================
# experiment info
# ==============================================
exp_name = 'eye pain'
exp_info = {
    'participant': '',
    'gender': ('male', 'female', 'other'),
    'session': ('threshold', 'main', 'threshold_demo', 'main_demo'),
    'intensity': ('1', '2', '3', '4', '5'),
    'age': '',
    'dominant eye': ('right', 'left', 'both'),
    'left-handed': False
}

# ==============================================
# event types
# ==============================================
session_start_msg = "SESSION_START"
session_end_msg = "SESSION_END"
trail_start_msg = "TRIAL_START"
trail_end_msg = "TRIAL_END"
vas_rating_msg = "VAS_RATING"
program_start_msg = "PROGRAM_START"
marker_msg = "MARKER"
led_on_msg = "LED_ON"
led_off_msg = "LED_OFF"

events = {
    session_start_msg: 10,
    session_end_msg: 20,
    trail_start_msg: 30,
    trail_end_msg: 40,
    vas_rating_msg: 50,
    program_start_msg: 60,
    marker_msg: 70,
    led_on_msg: 80,
    led_off_msg: 90,
}

# ==============================================
# AESTHETHICS
# ==============================================

#screen size
scrsize = (1920, 1080)  # screen size in pixels
#scrsize = (3840, 2160)  # screen size in pixels -> lab
#text
txt_size = scrsize[0]/27  # text size
#colors
txt_color = 'black'
#cue size
pic_size = 3.5   # as 1/X % of screen
#cue position
pic_pos = [0, 0.3*scrsize[1]]
# text presentation orientation
if is_hebrew == 1:
    txt_align = 'center'
else:
    txt_align = 'left'
