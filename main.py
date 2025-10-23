from __future__ import division
from psychopy import visual, event, core, gui, data

import config
import expirimentUtils
import eegHandler
import MedocHandler
import VideoCamera
import IRCamera
import LED
import eyelinkHandler
import dataRecordHandler
import drawingHandler
import flow

# ==============================================
# SET EXPERIMENT INFO
# ==============================================

#get expiriment info from gui
exp_info = config.exp_info
dlg = gui.DlgFromDict(dictionary=exp_info, title=config.exp_name)
if dlg.OK == False:
    core.quit()

#get info from system
exp_info['date'] = data.getDateStr()
exp_info['exp_name'] = config.exp_name
exp_info['screen_size'] = config.scrsize

#get session type
curr_session = exp_info['session']
not_demo = not curr_session.endswith('demo')

# ==============================================
# INIT EXPERIMENT
# ==============================================
clock = core.monotonicClock
win = visual.Window(size=config.scrsize, color='grey', units='pix', fullscr=config.full_screen)
mouse = event.Mouse(visible=False)
draw = drawingHandler.Draw(win)

eeg = eegHandler.EEG()
medoc = MedocHandler.Medoc(dry_run=True) 
led = LED.init()

if not_demo:
    log = dataRecordHandler.DataRecordHandler(clock=clock, eeg=eeg, led=led)
    dataPath = log.output_dir
    el = eyelinkHandler.Eyelink(dataPath)
    log.el = el
    ir = IRCamera.init(log.output_dir)
    vc = VideoCamera.init(log.output_dir)
else:
    log = ir = vc = None

# ==============================================
# HELPER FUNCTIONS
# ==============================================
def generate_wait_times(trails_number=config.main_trails_number):
    return expirimentUtils.generate_wait_times(trails_number=trails_number)

def vas(instructions=None):
    final_rating, reaction_time = expirimentUtils.pain_vas(win, instructions=instructions)
    if not_demo:
        log.event(config.vas_rating_msg, config.vas_rating_msg + "_" + str(final_rating), led_flash=False)
    return final_rating, reaction_time

def start_session():
    """Start a session and initialize data recording."""
    if curr_session == 'main':
        el.calibrate()
        el.start_recording()
        ir.calibrate()
        ir.start_recording()
        vc.start_recording()
    if not_demo:
        log.start_session()

def finish_session():       
    """Finish a session and save all recorded data."""
    if not_demo:
        log.finish_session()
        summary = log.get_summary()
        print("\n=== Session Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    if curr_session == 'main':
        el.stop_recording()
        ir.stop_recording()
        vc.stop_recording()

def start_trial(trial_num):
    medoc.trigger()
    if not_demo:
        log.event(config.trail_start_msg, f"trail_start_{trial_num}")

def finish_trial(trial_num):
    if not_demo:
        log.event(config.trail_end_msg, f"trail_end_{trial_num}")



start_session()
# ==============================================
# THRESHOLD SESSION
# ==============================================
if curr_session == "threshold":
    # generate wait times
    intervals = generate_wait_times(trails_number=config.threshold_trails_number)

    # welcome screen
    draw.instructions("Welcome and thank you for participating!")
    draw.show()
    flow.wait_for_space()
    
    #start medoc program
    medoc.start_program("threshold") 
    log.event(config.program_start_msg)

    # start instructions screen
    draw.top_instructions("Pain stimulus will be applied to your hand.")
    draw.middle_instructions("Please press SPACE to start, then SPACE again when you feel pain.")
    draw.show()
    flow.wait_for_space()

    # trails
    for i in range(config.threshold_trails_number):
        trial_num = i + 1

        # pain trial instructions
        draw.bottom_instructions("Press SPACE when you feel pain.")
        draw.fixation_cross()
        draw.show()

        # pain trial 
        flow.wait(intervals[i]) # wait for interval
        start_trial(trial_num) #trigger medoc trail and log trial start
        flow.wait_for_space() #wait for user to press space
        medoc.yes() #trigger medoc yes to stop trail
        finish_trial(trial_num) #log trial end

        # log trial data
        if config.use_vas: #if use vas, ask for pain rating and log trial data
            pain_rating, reaction_time = vas(instructions="Rate your pain level using the scale below")
            log.trial(trial_number=trial_num, pain_rating=pain_rating, reaction_time=reaction_time, wait_time=intervals[i])
        else: #if not just log wait time
            log.trial(trial_number=trial_num, wait_time=intervals[i])
    
    # thank you screen
    draw.blank() 
    draw.instructions("Thank you for your time!")
    draw.show()
    flow.wait_for_space()

    medoc.stop_program("threshold") #stop medoc program


# ==============================================
# MAIN SESSION
# ==============================================
elif curr_session == "main":
    intensity = exp_info["intensity"] #get intensity
    intervals = generate_wait_times() #generate wait times

    # welcome screen
    draw.instructions("Welcome and thank you for participating!")
    draw.show()
    flow.wait_for_space()

    # start instructions screen
    draw.top_instructions("A pain stimulus will be applied to your hand.")
    draw.fixation_cross()
    draw.bottom_instructions("Please fixate on the cross")
    draw.show()
    flow.wait_for_space()

    #start medoc program
    medoc.start_program("main_"+intensity)
    log.event(config.program_start_msg)

    # trails
    for i in range(config.main_trails_number):
        trial_num = i + 1

        # pain trial instructions
        draw.fixation_cross()
        draw.show()

        # pain trial
        flow.wait(intervals[i]) #wait for interval
        start_trial(trial_num) #trigger medoc trail and log trial start
        medoc.wait_for_trail_end() #wait for trail end
        finish_trial(trial_num) #log trial end

        # log trial data
        if config.use_vas: #if use vas, ask for pain rating and log trial data
            pain_rating, reaction_time = vas(instructions="Rate your pain level using the scale below")
            log.trial(trial_number=trial_num, pain_rating=pain_rating, reaction_time=reaction_time, wait_time=intervals[i])
        else: #if not just log wait time
            log.trial(trial_number=trial_num, wait_time=intervals[i])

    # thank you screen
    draw.blank()
    draw.instructions("Thank you for your time!")
    draw.show()
    flow.wait_for_space()

    medoc.stop_program("main_"+intensity) #stop medoc program

# ==============================================
# THRESHOLD DEMO SESSION
# ==============================================
elif curr_session == "threshold_demo":
    intervals = generate_wait_times(trails_number=config.threshold_demo_trails_number)

    # welcome screen
    draw.top_instructions("Welcome and thank you for participating!")
    draw.bottom_instructions("this is a demo session")
    draw.show()
    flow.wait_for_space()
    
    #start medoc program
    medoc.start_program("threshold_demo") 

    # start instructions screen
    draw.top_instructions("Pain stimulus will be applied to your hand.")
    draw.middle_instructions("Please press SPACE to start, then SPACE again when you feel pain.")
    draw.show()
    flow.wait_for_space()

    # trails
    for i in range(config.threshold_demo_trails_number):
        trial_num = i + 1

        # pain trial instructions
        draw.bottom_instructions("Press SPACE when you feel pain.")
        draw.fixation_cross()
        draw.show()

        # pain trial 
        flow.wait(intervals[i]) # wait for interval
        start_trial(trial_num) #trigger medoc trail 
        flow.wait_for_space() #wait for user to press space
        medoc.yes() #trigger medoc yes to stop trail

        # vas
        if config.use_vas: #if use vas, ask for pain rating 
            vas(instructions="Rate your pain level using the scale below")
    
    # thank you screen
    draw.blank() 
    draw.instructions("Thank you for your time!")
    draw.show()
    flow.wait_for_space()

    medoc.stop_program("threshold_demo") #stop medoc program

# ==============================================
# MAIN DEMO SESSION
# ==============================================
elif curr_session == "main_demo":
    intervals = generate_wait_times() #generate wait times

    # welcome screen
    draw.top_instructions("Welcome and thank you for participating!")
    draw.bottom_instructions("this is a demo session")
    draw.show()
    flow.wait_for_space()

    # start instructions screen
    draw.top_instructions("A pain stimulus will be applied to your hand.")
    draw.fixation_cross()
    draw.bottom_instructions("Please fixate on the cross")
    draw.show()
    flow.wait_for_space()

    #start medoc program
    medoc.start_program("main_demo")

    # trails
    for i in range(config.main_demo_trails_number):
        trial_num = i + 1

        # pain trial instructions
        draw.fixation_cross()
        draw.show()

        # pain trial
        flow.wait(intervals[i]) #wait for interval
        start_trial(trial_num) #trigger medoc trail 
        medoc.wait_for_trail_end() #wait for trail end

        # vas
        if config.use_vas: #if use vas, ask for pain rating 
            vas(instructions="Rate your pain level using the scale below")

    # thank you screen
    draw.blank()
    draw.instructions("Thank you for your time!")
    draw.show()
    flow.wait_for_space()

    medoc.stop_program("main_demo") #stop medoc program


finish_session()



    








 




