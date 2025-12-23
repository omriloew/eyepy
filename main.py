from __future__ import division
from psychopy import visual, event, core, gui, data

import config
import expirimentUtils
import eegHandler
import MedocHandler
import VideoCamera
import LED
import eyelinkHandler
import dataRecordHandler
import drawingHandler
import os
import shutil
import coloredPrint as cp
import random

# ==============================================
# SET EXPERIMENT INFO
# ==============================================
def get_exp_info():
    """Get experiment info from GUI."""
    exp_info = config.exp_info
    while True:
        dlg = gui.DlgFromDict(dictionary=exp_info, title=config.exp_name)
        if dlg.OK == False:
            core.quit()
        output_dir = os.path.join(config.datapath, exp_info['participant'], str(exp_info['session_number']), exp_info['session'])
        if os.path.exists(output_dir):
            # Create a simple Yes/No dialog using PsychoPy's gui
            overwrite_dlg = gui.Dlg(title="Warning", labelButtonOK="Yes", labelButtonCancel="No")
            overwrite_dlg.addText("A session with the same session type, number and participant already exists.\nThis will overwrite existing data.\n\nDo you want to continue?")
            overwrite_dlg.show()
            if overwrite_dlg.OK:  # User clicked Yes - proceed to overwrite
                break
            # User clicked No - loop back to get new session info
        else:
            # Directory doesn't exist - proceed
            break

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    exp_info['output_dir'] = output_dir
    exp_info['date'] = data.getDateStr()
    exp_info['exp_name'] = config.exp_name
    exp_info['screen_size'] = config.scrsize
    if exp_info['demo']:
        exp_info['session'] = exp_info['session'] + '_demo'
    return exp_info

#get session type
exp_info = get_exp_info()
curr_session = exp_info['session']
not_demo = not exp_info['demo']
num_trails = config.num_trails[curr_session]
log_info = cp.logger("INFO")
log_info("got exp info")

# ==============================================
# INIT EXPERIMENT
# ==============================================
clock = core.Clock()
win = visual.Window(size=config.scrsize,screen=2 , color='grey', units='pix', fullscr=config.full_screen)
mouse = event.Mouse(visible=False)
draw = drawingHandler.Draw(win)

eeg = eegHandler.init()
medoc = MedocHandler.init()
led = LED.init()

log = dataRecordHandler.DataRecordHandler(clock=clock, eeg=eeg, led=led)
medoc.log = log
el = eyelinkHandler.init(log.output_dir)
log.el = el
vc = VideoCamera.init(log.output_dir)
log_info("initialized devices")
# ==============================================
# HELPER FUNCTIONS
# ==============================================

def generate_wait_times(trails_number=num_trails):
    return expirimentUtils.generate_wait_times(trails_number=trails_number)


def start_session():
    """Start a session and initialize data recording."""
    clock.reset()
    draw.blank()
    draw.top_instructions("Welcome and thank you for participating!")
    draw.bottom_instructions("Press SPACE to start calibration")
    draw.show()
    wait_for_space_or_escape()
    if not_demo:
        draw.bottom_instructions("calibration in progress...")
        draw.show()
        el.calibrate(win)
        el.start_recording()
        vc.start_recording()
        draw.blank()
        draw.instructions("Press SPACE to start the session")
        draw.show()
        wait_for_space_or_escape()
        draw.blank()
        draw.instructions("synchronizing... please wait")
        draw.show()
        log.start_session()

def finish_session():       
    """Finish a session and save all recorded data."""
    if not_demo:
        draw.instructions("finishing session...")
        draw.show()
        el.stop_recording()
        vc.stop_recording()
        log.finish_session()
        summary = log.get_summary()
        cp.print_session("\n=== Session Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    draw.blank()
    draw.instructions("Thank you for your time!")
    draw.show()
    wait_for_space()

def wait_for_space(log_medoc=True):
    log_info("waiting for space")
    while not check_for_space():
        core.wait(0.02)
        if log_medoc:
            log.medoc_event(medoc.get_status())

def wait_for_space_or_escape(log_medoc=True):
    log_info("waiting for space or escape")
    while not check_for_space_or_escape():
        core.wait(0.02)
        if log_medoc:
            log.medoc_event(medoc.get_status())

def wait_for_space_escape_or_quit(log_medoc=True):
    log_info("waiting for space or escape or quit")
    res, key = check_for_space_escape_or_quit()
    while not res:
        res, key = check_for_space_escape_or_quit()
        core.wait(0.02)
        if log_medoc:
            log.medoc_event(medoc.get_status())
    return key == 'space'

def wait(seconds, log_medoc=True):
    log_info(f"waiting for {seconds} seconds")
    if log_medoc:
        for _ in range(int(seconds*100)):
            core.wait(0.01)
            medoc.get_status()
    else:
        core.wait(seconds)
    clear_keys()
    
def check_for_space():
    keys = event.getKeys(keyList=['space'])
    if 'space' in keys:
        return True
    return False

def check_for_escape():
    keys = event.getKeys(keyList=['escape'])
    if 'escape' in keys:
        abort_session()

def check_for_space_or_escape():
    keys = event.getKeys(keyList=['space', 'escape'])
    if 'escape' in keys:
        abort_session()
    if 'space' in keys:
        return True
    return False

def check_for_space_escape_or_quit():
    keys = event.getKeys(keyList=['space', 'escape', 'q'])
    if 'escape' in keys:
        abort_session()
    if 'q' in keys:
        return True, 'quit'
    if 'space' in keys:
        return True, 'space'
    return False, None

def abort_session():
    medoc.abort()
    cp.print_error("Escape key pressed, aborting session")
    core.quit() 

def genarate_main_temperatures(th_temps, rate_temps):
    """
    Generate temperatures for main session.
    args:
        th_temps: list of threshold temperatures
        rate_temps: list of pain rating temperatures
    returns:
        list of temperatures for main session
    """
    temperatures = []
    th_mean = sum(th_temps) / len(th_temps)
    temperatures.append(th_mean - config.threshold_reduction_factor)
    for rate_temp in rate_temps:
        temperatures.append(rate_temp)

    result = []
    for temp in temperatures:
        result.extend([temp] * config.each_tmp_rep_num)
    random.shuffle(result)
    return result

def clear_keys():
    event.clearEvents()

# ==============================================
# THRESHOLD SESSION
# ==============================================
def threshold_session():
    # generate wait times
    intervals = generate_wait_times(trails_number=num_trails)
    
    #start medoc program
    medoc.start_thermal_program()
    # start instructions screen
    draw.top_instructions("Pain stimulus will be applied to your hand.")
    draw.middle_instructions("Please press SPACE to start, then SPACE again when you feel pain.")
    draw.show()
    wait_for_space_or_escape()

    # trails
    for i in range(num_trails):
        trial_num = i + 1
        # pain trial instructions
        draw.bottom_instructions("Press SPACE when you feel pain.")
        draw.fixation_cross()
        draw.show()
        wait(0.2)

        wait(intervals[i]) # wait for interval
        medoc.start_threshold_trial() 
        wait_for_space_or_escape() #wait for user to press space
        temperature = medoc.stop_threshold_trial(trial_num)
        log.trial(trial_number=trial_num, wait_time=intervals[i], temperature=temperature)
    
    medoc.stop()

# ==============================================
# PAIN RATING SESSION
# ==============================================

def pain_rating_session():
    # generate wait times
    intervals = generate_wait_times(trails_number=num_trails)
    
    #start medoc program
    medoc.start_thermal_program()

    # trails
    for i in range(num_trails):
        trial_num = i + 1
        # pain trial instructions
        draw.vas_scale(percentage=config.desired_ratings[i], left_label='0', right_label='100')
        draw.fixation_cross()
        draw.bottom_instructions("Please press SPACE to start, then SPACE again when the pain reaches the rating on the scale above")
        draw.show()
        wait_for_space_or_escape()
        wait(intervals[i]) # wait for interval
        medoc.start_threshold_trial() 
        wait_for_space_or_escape() #wait for user to press space
        temperature = medoc.stop_threshold_trial(trial_num)
        log.trial(trial_number=trial_num, wait_time=intervals[i], temperature=temperature)
    
    medoc.stop()

# ==============================================
# MAIN SESSION
# ==============================================
def main_session():
    intervals = generate_wait_times() #generate wait times
    th_temps = log.load_session_trials_data(session_type="threshold")
    rate_temps = log.load_session_trials_data(session_type="pain_rating")
    th_temps = [float(temperature) for temperature in th_temps]
    
    rate_temps = [float(temperature) for temperature in rate_temps]
    tempretures = genarate_main_temperatures(th_temps, rate_temps)
    # start instructions screen
    draw.blank()
    draw.top_instructions("A pain stimulus will be applied to your hand.")
    draw.fixation_cross()
    draw.bottom_instructions("Please fixate on the cross")
    draw.show()
    wait_for_space_or_escape()

    #start medoc program
    medoc.start_thermal_program()
    # trails
    medoc.skip_initial_pain_stimulus()
    for i in range(num_trails):
        if i in config.break_points_main_session:
            draw.blank()
            draw.top_instructions("Adjust the thermod position")
            draw.fixation_cross()
            draw.bottom_instructions("Press SPACE to continue")
            draw.show()
            wait_for_space_or_escape()
        trial_num = i + 1

        # pain trial instructions
        draw.fixation_cross()
        draw.show()
        # pain trial
        wait(intervals[i]) #wait for interval
        medoc.pain_stimulus_trial(temperature=tempretures[i])
        # log trial data
        log.trial(trial_number=trial_num, wait_time=intervals[i], temperature=tempretures[i])

    medoc.stop() #stop medoc program

# ==============================================
# CPM SESSION
# ==============================================

def cpm_session():
    
    tempretures = log.load_session_trials_data(session_type="pain_rating")
    tempretures = [float(temperature) for temperature in tempretures]
    temp_high = tempretures[-1]

    # start instructions screen
    draw.blank()
    draw.top_instructions("A pain stimulus will be applied to your hand.")
    draw.fixation_cross()
    draw.bottom_instructions("Please fixate on the cross")
    draw.show()
    wait_for_space_or_escape()

    #start medoc program
    medoc.start_thermal_program()
    medoc.skip_initial_pain_stimulus()
    clear_keys()
    # trails
    trial_num = 1
    while True:
        draw.blank()
        draw.fixation_cross()
        draw.show()

        # pain trial
        wait(1)
        medoc.pain_stimulus_trial(temperature=temp_high)
        # log trial data
        log.trial(trial_number=trial_num, wait_time=1, temperature=temp_high)
        trial_num += 1
        draw.blank()
        draw.instructions("SPACE to continue, Q to quit")
        draw.show()
        if not wait_for_space_escape_or_quit():
            draw.blank()
            draw.show()
            break
    
    medoc.stop() #stop medoc program

def custom_session():
   pass


# ==============================================
# PLR SESSION
# ==============================================

def plr_session():
    """
    PLR (Pupil Light Reflex) test session.
    Starts and ends with rest periods, with 10 sudden white screen flash trials in between.
    Events are logged to all devices for synchronization testing.
    """
    # Generate wait times for intervals between trials
    intervals = generate_wait_times(trails_number=num_trails)

    # Start with rest period
    draw.blank()
    draw.instructions("Rest period - Please relax and fixate on the center")
    draw.show()
    log.event(config.rest_start_msg)
    wait(2, log_medoc=False)  # 2 second rest period at start

    # Instructions
    draw.blank()
    draw.top_instructions("White screen flashes will appear.")
    draw.bottom_instructions("Please keep your eyes open and fixate on the center")
    draw.show()
    wait_for_space_or_escape(log_medoc=False)

    # Create a white rectangle that covers the entire screen for the flash
    white_rect = visual.Rect(win, width=config.scrsize[0], height=config.scrsize[1],
                             fillColor='white', lineColor='white', pos=(0, 0))

    # Create a black rectangle that covers the entire screen for the background
    black_rect = visual.Rect(win, width=config.scrsize[0], height=config.scrsize[1],
                             fillColor='black', lineColor='black', pos=(0, 0))

    # Create a white fixation cross for visibility on black background
    white_fixation = visual.TextStim(win, text='+', color='white', height=150,
                                     alignText='center', antialias=False)

    # Flash duration in seconds (can be adjusted)
    flash_duration = 1.0

    # Generate longer wait times for PLR (minimum 2 seconds, mean 3 seconds)
    # This ensures enough time between flashes for proper PLR measurement
    plr_intervals = expirimentUtils.generate_wait_times(
        trails_number=num_trails,
        mean=3.0,  # Mean of 3 seconds
        std=1.0,  # Standard deviation of 1 second
        min_time=2.0,  # Minimum 2 seconds between flashes
        max_time=5.0  # Maximum 5 seconds between flashes
    )

    # Show black background with white fixation immediately BEFORE first flash
    black_rect.draw()  # Draw black background
    white_fixation.draw()  # Draw white fixation cross
    win.flip()  # Show black screen with fixation
    wait(1, log_medoc=False)  # Brief pause before starting flashes

    # 10 white screen flash trials
    for i in range(num_trails):
        trial_num = i + 1

        # Wait for interval before flash (using longer PLR intervals)
        wait(plr_intervals[i], log_medoc=False)

        # White screen flash - draw white rectangle covering entire screen
        white_rect.draw()
        win.flip()
        log.event(config.white_screen_on_msg)  # Log event to all devices
        core.wait(flash_duration)

        # Return to black screen - draw black rectangle and fixation cross
        black_rect.draw()  # Draw black background
        white_fixation.draw()  # Show fixation cross again
        win.flip()  # Show black screen with fixation
        log.event(config.white_screen_off_msg)  # Log event to all devices

        # Log trial data
        log.trial(trial_number=trial_num, wait_time=plr_intervals[i], trial_duration=flash_duration)

    # End with rest period - restore grey background
    win.color = 'grey'  # Restore grey background
    draw.blank()
    draw.instructions("Rest period - Please relax")
    draw.show()
    log.event(config.rest_end_msg)
    wait(2, log_medoc=False)  # 2 second rest period at end


if __name__ == "__main__":
    start_session()
    if curr_session.startswith("threshold"):
        threshold_session()
    elif curr_session.startswith("main"):
        main_session()
    elif curr_session.startswith("pain_rating"):
        pain_rating_session()
    elif curr_session.startswith("cpm"):
        cpm_session()
    elif curr_session.startswith("custom"):
        custom_session()
    elif curr_session.startswith("plr"):
        plr_session()
    finish_session()








 




