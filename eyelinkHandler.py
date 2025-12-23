"""
Eyelink placeholder module for testing without hardware.
"""
import psychopy
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
import pylink as pl
import config
import coloredPrint as cp

if config.exp_info['demo'] or not config.eyelink_on:
    print_el_log = cp.logger("EYELINK", place_holder=True)
else:
    print_el_log = cp.logger("EYELINK")

def init(dataPath):
    if not config.eyelink_on or config.exp_info['demo']:
        return Eyelinkplaceholder()
    return Eyelink(dataPath)

class Eyelink:

    def __init__(self, dataPath):
        self.edfFileName = "test.edf"
        self.edfFilePath = dataPath + '\\' + self.edfFileName

        self.eye_link = pl.EyeLink()
        self.eye_link.sendCommand("screen_pixel_coords = 0 0 %d %d" % config.scrsize)
        self.eye_link.sendMessage("DISPLAY_COORDS  0 0 %d %d" % config.scrsize)

        if self.eye_link.getTrackerVersion() == 3:
            tvstr = self.eye_link.getTrackerVersionString()
            vindex = tvstr.find("EYELINK CL")
            tracker_software_ver = int(float(tvstr[(vindex + len("EYELINK CL")):].strip()))
            print_el_log(f"Tracker software version: {tracker_software_ver}")

        self.eye_link.sendCommand("select_parser_configuration 0")
        self.eye_link.sendCommand("pupil_size_diameter = %s" % ("YES"))

        # not sure what this is
        self.eye_link.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON")
        # this for the file
        self.eye_link.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,HTARGET")
        # not sure what this is
        self.eye_link.sendCommand("button_function 5 'accept_target_fixation'");
        print_el_log("Initialized")

    def calibrate(self, win):
        print_el_log("Calibrating")

        el_tracker = self.eye_link  # this is your pl.EyeLink object

        # Create the graphics environment for calibration
        genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
        # open the calibration graphics
        pl.openGraphicsEx(genv)

        # run the standard calibration / validation UI
        el_tracker.doTrackerSetup()

        # close the graphics when done
        #pl.closeGraphics()

        print_el_log("Calibration complete")

    def start_recording(self):
        """Start eye tracking recording."""
        print_el_log("Starting recording")
        self.eye_link.openDataFile(self.edfFileName)
        self.eye_link.startRecording(1, 1, 1, 1)

    def stop_recording(self):
        print_el_log("Stopping recording")
        self.eye_link.setOfflineMode()
        psychopy.core.wait(0.5)
        self.eye_link.closeDataFile()
        self.eye_link.receiveDataFile(self.edfFileName, self.edfFilePath)
        pl.closeGraphics()
        self.eye_link.close()


    
    def send_message(self, message):
        print_el_log(f"Sending message: {message}")
        self.eye_link.sendMessage(message)
       
    
class Eyelinkplaceholder:

    def __init__(self):
        print_el_log("Initialized as placeholder")

    def send_message(self, message):
        print_el_log(f"Sending message: {message}")

    def calibrate(self,win):
        print_el_log("Calibrating")

    def start_recording(self):
        print_el_log("Starting recording")

    def stop_recording(self):
        print_el_log("Stopping recording")

    def edf_to_csv(self):
        print_el_log("Converting EDF to CSV")