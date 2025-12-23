
# ==============================================
# MEDOC PARAMETERS
# ==============================================
medoc_on = 0
medoc_host = '192.168.1.10'
medoc_port = 20121
medoc_programs = {
    'main': '01010000',
    'main_demo': '00010001',
    'threshold': '01010101',
    'threshold_demo': '00010001',
    'pain_rating': '00001111',
    'pain_rating_demo': '00010001',
    'cpm': '10101010',
    'cpm_demo': '00010001',
    'custom': '00010001',
    'custom_demo': '00010001',
}
medoc_base_temperature = 32.0

# ==============================================
# VIDEO CAMERA PARAMETERS
# ==============================================
video_camera_on = 0

# ==============================================
# LED PARAMETERS
# ==============================================
led_on = 1
led_baud_rate = 9600
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
eeg_port_address = 0x03FF8

# ==============================================
# experiment parameters
# ==============================================
full_screen = 0
timeout = 100 
is_hebrew = 0
debug = True
manual_threshold_temperature_data = [40.0, 32.0, 36.0, 44.0, 48.0]
manual_pain_rating_temperature_data = [40.0, 32.0, 36.0, 44.0, 48.0]
manual_temperature_data = {
    'threshold': manual_threshold_temperature_data,
    'pain_rating': manual_pain_rating_temperature_data,
}
desired_ratings = [20, 40, 60]
each_tmp_rep_num = 4
break_points_main_session = [5, 10]
threshold_reduction_factor = 2
trail_dur_sec = {
    'main': 15,
    'cpm': 30
}
num_trails = {
    'threshold': 4,
    'pain_rating': len(desired_ratings),
    'main': (len(desired_ratings) + 1) * each_tmp_rep_num,
    'cpm': 2,
    'plr' : 10,
    'custom': 0,
    'threshold_demo': 3,
    'main_demo': 3,
    'pain_rating_demo': 3,
    'custom_demo': 0,
    'cpm_demo': 3,
}

# ==============================================
# VAS (Visual Analog Scale) parameters
# ==============================================
vas_max = 100  # Maximum value on the scale
vas_min = 0  # Minimum value on the scale
vas_granularity = 1  # Step size for the scale
vas_keyboard_step = 1  # How much to jump when using arrow keys
vas_start_value = 50  # Starting position of the slider
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
participent = 'omri'
session_number = '1'

exp_name = 'eye pain'
exp_info = {
    'participant': (participent),
    'session_number': (session_number),
    'gender': ('male', 'female', 'other'),
    'session': ('threshold', 'pain_rating' , 'main', 'cpm', 'custom'),
    'age': '',
    'dominant eye': ('right', 'left', 'both'),
    'left-handed': False,
    'demo': False,
}

# ==============================================
# post experiment raw process parameters
# ==============================================
process_swir = True

# ==============================================
# event types
# ==============================================
session_start_msg = "SESSION_START"
session_end_msg = "SESSION_END"
led_on_msg = "LED_ON"
led_off_msg = "LED_OFF"
vas_rating_msg = "VAS_RATING"
rest_start_msg = "REST_START"
rest_end_msg = "REST_END"
white_screen_on_msg = "WHITE_SCREEN_ON"
white_screen_off_msg = "WHITE_SCREEN_OF"

events = {
    session_start_msg: 20,
    session_end_msg: 30,
    led_on_msg: 40,
    led_off_msg: 50,
    vas_rating_msg: 60,
    rest_start_msg: 60,
    rest_end_msg: 70,
    white_screen_on_msg: 80,
    white_screen_off_msg: 90,
    'MEDOC_GET_STATUS': 0,
    'MEDOC_SELECT_TP': 1,
    'MEDOC_START': 2,
    'MEDOC_PAUSE': 3,
    'MEDOC_TRIGGER': 4,
    'MEDOC_STOP': 5,
    'MEDOC_ABORT': 6,
    'MEDOC_YES': 7,       
    'MEDOC_NO': 8,         
    'MEDOC_COVAS': 9,
    'MEDOC_VAS': 10,
    'MEDOC_SPECIFY_NEXT': 11,
    'MEDOC_T_UP': 12,      
    'MEDOC_T_DOWN': 13,    
    'MEDOC_KEYUP': 14,     
    'MEDOC_UNNAMED': 15
}

# ==============================================
# AESTHETHICS
# ==============================================

#screen size
scrsize = (1024, 768)  # screen size in pixels
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
