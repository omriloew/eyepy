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
import coloredPrint as cp

# ==============================================
# SET EXPERIMENT INFO
# ==============================================
def get_exp_info():
    """Get experiment info from GUI."""
    exp_info = config.exp_info
    dlg = gui.DlgFromDict(dictionary=exp_info, title=config.exp_name)
    if dlg.OK == False:
        core.quit()
    output_dir = os.path.join(config.datapath, exp_info['participant'], str(exp_info['session_number']), exp_info['session'])
    while os.path.exists(output_dir):
        gui.warnDlg(title="A session with the same session type, number and participant already exists", prompt="A session with the same session type, number and participant already exists, Please enter a new session info")
        dlg = gui.DlgFromDict(dictionary=exp_info, title=config.exp_name)
        if dlg.OK == False:
            core.quit()
        output_dir = os.path.join(config.datapath, exp_info['participant'], str(exp_info['session_number']), exp_info['session'])
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

# ==============================================
# INIT EXPERIMENT
# ==============================================
clock = core.Clock()
win = visual.Window(size=config.scrsize, color='grey', units='pix', fullscr=config.full_screen)
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

# ==============================================
# HELPER FUNCTIONS
# ==============================================

def generate_wait_times(trails_number=num_trails):
    return expirimentUtils.generate_wait_times(trails_number=trails_number)

def vas(instructions=None):
    final_rating, reaction_time = expirimentUtils.pain_vas(win, instructions=instructions)
    if not_demo:
        log.event(config.vas_rating_msg, config.vas_rating_msg + "_" + str(final_rating))
    return final_rating, reaction_time

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
        el.calibrate()
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
    cp.print_info("[LOG]", end="")
    print("waiting for space")
    while not check_for_space():
        core.wait(0.02)
        if log_medoc:
            log.medoc_event(medoc.get_status())

def wait_for_space_or_escape(log_medoc=True):
    cp.print_info("[LOG]", end="")
    print("waiting for space or escape")
    while not check_for_space_or_escape():
        core.wait(0.02)
        if log_medoc:
            log.medoc_event(medoc.get_status())

def wait(seconds, log_medoc=True):
    cp.print_info("[LOG]", end="")
    print("waiting for", seconds, "seconds")
    if log_medoc:
        for _ in range(int(seconds*100)):
            core.wait(0.01)
            medoc.get_status()
    else:
        core.wait(seconds)
    
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

def abort_session():
    medoc.abort()
    cp.print_error("Escape key pressed, aborting session")
    core.quit() 

def genarate_main_temperatures(ratings, temperetures):
    #TODO - implement a corect method
    desired_ratings = config.desired_ratings
    pares = zip(ratings, temperetures)
    pares = sorted(pares, key=lambda x: x[0])
    resulting_temperatures = []
    for rating in desired_ratings:
        i = 0
        while i < len(pares) and pares[i][0] < rating:
            i += 1
        if i == len(pares):
            resulting_temperatures.append(pares[-1][1])
        else:
            resulting_temperatures.append(pares[i][1])
    return resulting_temperatures




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
    intervals = generate_wait_times() #generate wait times
    tempretures, _ = log.load_session_trials_data(session_type="threshold")
    tempretures = [float(temperature) for temperature in tempretures]
    mean_threshold_temperature = sum(tempretures) / len(tempretures)
    max_temperature = 50
    initial_temperature = mean_threshold_temperature - 5
    temperature_step = (max_temperature - initial_temperature) / num_trails
    draw.top_instructions("A pain stimulus will be applied to your hand.")
    draw.fixation_cross()
    draw.bottom_instructions("Please fixate on the cross")
    draw.show()
    wait_for_space_or_escape()

    #start medoc program
    draw.top_instructions("starting session...")
    draw.fixation_cross()
    draw.show()
    medoc.start_thermal_program()
    medoc.skip_initial_pain_stimulus()
    # trails
    for i in range(num_trails):
        trial_num = i + 1 
        temperature = initial_temperature + i * temperature_step
        # pain trial instructions
        draw.fixation_cross()
        draw.show()

        # pain trial
        wait(intervals[i]) #wait for interval
        medoc.pain_stimulus_trial(temperature=temperature) 
        # log trial data
        pain_rating, reaction_time = vas(instructions="Rate your pain level using the scale below")
        log.trial(trial_number=trial_num, pain_rating=pain_rating, reaction_time=reaction_time, wait_time=intervals[i], temperature=temperature)

    medoc.stop() #stop medoc program


# ==============================================
# MAIN SESSION
# ==============================================
def main_session():
    intervals = generate_wait_times() #generate wait times
    tempretures, ratings = log.load_session_trials_data(session_type="pain_rating")
    tempretures = [float(temperature) for temperature in tempretures]
    ratings = [float(rating) for rating in ratings]
    tempretures = genarate_main_temperatures(ratings, tempretures)

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
    for i in range(num_trails):
        trial_num = i + 1

        # pain trial instructions
        draw.fixation_cross()
        draw.show()

        # pain trial
        wait(intervals[i]) #wait for interval
        medoc.pain_stimulus_trial(temperature=tempretures[i])
        # log trial data
        pain_rating, reaction_time = vas(instructions="Rate your pain level using the scale below")
        log.trial(trial_number=trial_num, pain_rating=pain_rating, reaction_time=reaction_time, wait_time=intervals[i], temperature=tempretures[i])

    medoc.stop() #stop medoc program



def custom_session():
    #TODO - implement
    pass


if __name__ == "__main__":
    start_session()
    if curr_session.startswith("threshold"):
        threshold_session()
    elif curr_session.startswith("main"):
        main_session()
    elif curr_session.startswith("pain_rating"):
        pain_rating_session()
    elif curr_session.startswith("custom"):
        custom_session()
    finish_session()








 




