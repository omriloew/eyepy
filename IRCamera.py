"""
IR Camera placeholder module for testing without hardware.
"""
from datetime import datetime

def init(data_path):
    """Initialize IR Camera placeholder."""
    print("IR Camera placeholder initialized")
    return IRCameraPlaceholder(data_path)

class IRCameraPlaceholder:
    """Placeholder for IR Camera functionality."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.recording = False
        self.filename = None

    def generate_filename(self):
        """Generate filename for IR recording."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"ir_recording_{timestamp}.avi"
    
    def calibrate(self):
        """Calibrate IR camera."""
        print("IR camera calibrated (placeholder)")
        return True
    
    def start_recording(self):
        """Start IR recording."""
        print("IR recording started (placeholder)")
        self.recording = True
        return True

    
    def stop_recording(self):
        """Stop IR recording."""
        print("IR recording stopped (placeholder)")
        self.recording = False
        return True

    def save_recording(self):
        """Save IR recording."""
        print("IR recording saved (placeholder)")
        return True
    
    def is_recording(self):
        """Check if recording."""
        return self.recording
