import coloredPrint as cp
import config

if config.exp_info['demo'] or not config.video_camera_on:
    print_vc_log = cp.logger("VIDEO CAMERA", place_holder=True)
else:
    print_vc_log = cp.logger("VIDEO CAMERA")

def init(data_path):
    return VideoCameraPlaceholder(data_path)


class VideoCameraPlaceholder:
    """Placeholder for Video Camera functionality."""
    
    def __init__(self, data_path):
        print_vc_log("Initialized as placeholder")
        self.data_path = data_path
    
    def start_recording(self):
        print_vc_log("Video recording started")
      
    
    def stop_recording(self):
        print_vc_log("Video recording stopped")

