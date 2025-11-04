
# ==============================================
# MEDOC PARAMETERS
# ==============================================
medoc_on = 0
medoc_host = '192.168.1.10'
medoc_port = 20121
medoc_programs = {
    'main': '00010001', 
    'main_demo': '00010001',
    'threshold': '00010001',
    'threshold_demo': '00010001',
    'pain_rating': '00010001',
    'pain_rating_demo': '00010001',}
medoc_base_temperature = 32.0

# ==============================================
# VIDEO CAMERA PARAMETERS
# ==============================================
video_camera_on = 0

# ==============================================
# SWIR CAMERA PARAMETERS
# ==============================================
swir_camera_on = 0

# ==============================================
# LED PARAMETERS
# ==============================================
led_on = 0
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
timeout = 100 #8   #skip question if too long in seconds
is_hebrew = 0
debug = True

num_trails = {
    'threshold': 4,
    'pain_rating': 5,
    'main': 4,
    'custom': 0,
    'threshold_demo': 3,
    'main_demo': 3,
    'pain_rating_demo': 3,
    'custom_demo': 0,
}

manual_temperature_data = [40.0, 32.0, 36.0, 44.0, 48.0]
manual_rating_data = [40, 0, 20, 60, 80]
desired_ratings = [0, 20, 40, 60]

# ==============================================
# VAS (Visual Analog Scale) parameters
# ==============================================
vas_show_value = True  # Show the current numeric value while choosing
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

exp_name = 'eye pain'
exp_info = {
    'participant': (participent),
    'session_number': '',
    'gender': ('male', 'female', 'other'),
    'session': ('threshold', 'pain_rating'  , 'main', 'custom'),
    'age': '',
    'dominant eye': ('right', 'left', 'both'),
    'left-handed': False,
    'demo': False,
}

# ==============================================
# event types
# ==============================================
session_start_msg = "SESSION_START"
session_end_msg = "SESSION_END"
led_on_msg = "LED_ON"
led_off_msg = "LED_OFF"
vas_rating_msg = "VAS_RATING"

events = {
    session_start_msg: 20,
    session_end_msg: 30,
    led_on_msg: 40,
    led_off_msg: 50,
    vas_rating_msg: 60,
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
