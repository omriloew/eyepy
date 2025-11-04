import coloredPrint as cp

def init(data_path):
    return VideoCameraPlaceholder(data_path)

class VideoCameraPlaceholder:
    """Placeholder for Video Camera functionality."""
    
    def __init__(self, data_path):
        cp.print_warning("[VC] [placeholder]", end="")
        print(" - Initialized")
        self.data_path = data_path
    
    def start_recording(self):
        cp.print_warning("[VC]", end="")
        print(" - Video recording started")
      
    
    def stop_recording(self):
        cp.print_warning("[VC]", end="")
        print(" - Video recording stopped")

