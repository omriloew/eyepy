"""
IR Camera placeholder module for testing without hardware.
"""
from datetime import datetime

def init(data_path):
    """Initialize SWIR Camera."""
    print("SWIR Camera initialized")
    return SWIRCamera(data_path)

class SWIRCamera:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.recording = False
        self.filename = None

    def generate_filename(self):
        """Generate filename for SWIR recording."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"swir_recording_{timestamp}.avi"
    
    def calibrate(self):
        """Calibrate SWIR camera."""
        print("SWIR camera calibrated (placeholder)")
        return True
    
    def start_recording(self):
        """Start SWIR recording."""
        print("SWIR recording started (placeholder)")
        self.recording = True
        return True

    
    def stop_recording(self):
        """Stop SWIR recording."""
        print("SWIR recording stopped (placeholder)")
        self.recording = False
        return True

    def save_recording(self):
        """Save SWIR recording."""
        print("SWIR recording saved (placeholder)")
        return True
    
    def is_recording(self):
        """Check if recording."""
        return self.recording
