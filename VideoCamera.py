"""
Video Camera placeholder module for testing without hardware.
"""

def init(data_path):
    """Initialize Video Camera placeholder."""
    print("Video Camera placeholder initialized")
    return VideoCameraPlaceholder(data_path)

class VideoCameraPlaceholder:
    """Placeholder for Video Camera functionality."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.recording = False
    
    def start_recording(self):
        """Start video recording."""
        print("Video recording started (placeholder)")
        self.recording = True
        return True
    
    def stop_recording(self):
        """Stop video recording."""
        print("Video recording stopped (placeholder)")
        self.recording = False
        return True
    
    def is_recording(self):
        """Check if recording."""
        return self.recording
