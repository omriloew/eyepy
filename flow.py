from psychopy import event, core

def wait_for_space():
    print("waiting for space")
    event.waitKeys(keyList=['space'])

def wait_for_space_or_escape():
    print("waiting for space or escape")
    keys = event.waitKeys(keyList=['space', 'escape'])
    if 'escape' in keys:
        core.quit()

def wait(seconds):
    print("waiting for", seconds, "seconds")
    core.wait(seconds)
