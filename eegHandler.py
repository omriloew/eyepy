from psychopy import visual, event, core, gui, data
import config
try:
    from psychopy import parallel
    parallel_available = True
except ImportError:
    parallel_available = False
    print("Warning: parallel port not available - EEG triggers disabled")

class EEG:
    
    def __init__(self):
        self.connected = config.eeg_on
        self.para_port = None
        if config.eeg_on and parallel_available:
            self.para_port = parallel.ParallelPort(address=config.eeg_port_address)
            self.para_port.setData(0)
    
    def send_trigger(self, trigger_code):  
        if self.connected and self.para_port is not None:
            self.para_port.setData(trigger_code)
            core.wait(0.1)
            self.para_port.setData(0)
    
