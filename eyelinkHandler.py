"""
Eyelink placeholder module for testing without hardware.
"""
import psychopy
import pylink as pl
import config
import coloredPrint as cp

def init(dataPath):
    if not config.eyelink_on or config.exp_info['demo']:
        return Eyelinkplaceholder()
    return Eyelink(dataPath)

class Eyelink:
    
    def __init__(self, dataPath):
        self.edfFileName = config.eyelink_edf_prefix + config.exp_info['participant'] + '_' + config.exp_info['session'] + '.edf'
        self.edfFilePath = dataPath + '\\' + self.edfFileName

        self.eye_link = pl.EyeLink()
        self.eye_link.sendCommand("screen_pixel_coords = 0 0 %d %d" % config.scrsize)
        self.eye_link.sendMessage("DISPLAY_COORDS  0 0 %d %d" % config.scrsize)

        if self.eye_link.getTrackerVersion() == 3:
            tvstr = self.eye_link.getTrackerVersionString()
            vindex = tvstr.find("EYELINK CL")
            tracker_software_ver = int(float(tvstr[(vindex + len("EYELINK CL")):].strip()))
            cp.print_info(tracker_software_ver)

        self.eye_link.sendCommand("select_parser_configuration 0")
        self.eye_link.sendCommand("pupil_size_diameter = %s" % ("YES"))

        # not sure what this is
        self.eye_link.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON")
        # this for the file
        self.eye_link.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,HTARGET")
        # not sure what this is
        self.eye_link.sendCommand("button_function 5 'accept_target_fixation'");
        cp.print_success("[Eyelink]", end="")
        print(" - Initialized")



    def calibrate(self):
        cp.print_info("[Eyelink]", end="")
        print(" - Calibrating")
        sp = (config.scrsize[0], config.scrsize[1])
        cd = 32
        pl.pylink_c.openGraphics(sp, cd)
        pl.pylink_c.setCalibrationColors((255, 255, 255), (0, 0, 0))
        pl.pylink_c.setTargetSize(int(config.scrsize[0] / 70), int(config.scrsize[1] / 300))
        pl.pylink_c.setCalibrationSounds("", "", "")
        pl.pylink_c.setDriftCorrectSounds("", "off", "off")
        self.eye_link.doTrackerSetup(self.eye_link)
        pl.closeGraphics()
        cp.print_success("[Eyelink]", end="")
        print(" - Calibration complete")
    
    def start_recording(self):
        """Start eye tracking recording."""
        cp.print_info("[Eyelink]", end="")
        print(" - Starting recording")
        self.eye_link.openDataFile(self.edfFileName)
        self.eye_link.startRecording(1, 1, 1, 1)
    
    def stop_recording(self):
        cp.print_info("[Eyelink]", end="")
        print(" - Stopping recording")
        self.eye_link.setOfflineMode()
        psychopy.core.wait(0.5)
        self.eye_link.closeDataFile()
        self.eye_link.receiveDataFile(self.edfFileName, self.edfFilePath)
        self.eye_link.close()

    def edf_to_csv(self):
        if not self.connected:
            return
        return
        #TODO: implement this
    
    def send_message(self, message):
        cp.print_info("[Eyelink]", end="")
        print(f" - Sending message: {message}")
        self.eye_link.sendMessage(message)
       
    
class Eyelinkplaceholder:

    def __init__(self):
        cp.print_warning("[Eyelink] [placeholder]", end="")
        print(" - Initialized")

    def send_message(self, message):
        cp.print_warning("[Eyelink] [placeholder]", end="")
        print(f" - Sending message: {message}")

    def calibrate(self):
        cp.print_warning("[Eyelink] [placeholder]", end="")
        print(" - Calibrating")

    def start_recording(self):
        cp.print_warning("[Eyelink] [placeholder]", end="")
        print(" - Starting recording")

    def stop_recording(self):
        cp.print_warning("[Eyelink] [placeholder]", end="")
        print(" - Stopping recording")

    def edf_to_csv(self):
        cp.print_warning("[Eyelink] [placeholder]", end="")
        print(" - Converting EDF to CSV")