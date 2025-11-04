from psychopy import core
import config
import coloredPrint as cp
try:
    from psychopy import parallel
    parallel_available = True
except ImportError:
    parallel_available = False
    cp.print_warning("Warning: parallel port not available - EEG triggers disabled")

def init():
    if not config.eeg_on or not parallel_available or config.exp_info['demo']:
        return EEGplaceholder()
    return EEG()

class EEG:
    
    def __init__(self):
        self.para_port = parallel.ParallelPort(address=config.eeg_port_address)
        self.para_port.setData(0)
        cp.print_success("[EEG]", end="")
        print(" - Initialized")
    
    def send_trigger(self, trigger_code):  
        cp.print_success("[EEG]", end="")
        print(f" - Sending trigger: {trigger_code}")
        self.para_port.setData(trigger_code)
        core.wait(0.1)
        self.para_port.setData(0)

class EEGplaceholder:

    def __init__(self):
        cp.print_warning("[EEG] [placeholder]", end="")
        print(" - Initialized")

    def send_trigger(self, trigger_code):
        cp.print_warning("[EEG] [placeholder]", end="")
        print(f" - Sending trigger: {trigger_code}")
