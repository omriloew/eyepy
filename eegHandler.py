from psychopy import core, parallel
import config
import coloredPrint as cp

try:
    from psychopy import parallel
    parallel_available = True
except ImportError:
    parallel_available = False
    cp.print_warning("Warning: parallel port not available - EEG triggers disabled")

if config.exp_info['demo'] or not config.eeg_on or not parallel_available:
    print_eeg_log = cp.logger("EEG", place_holder=True)
else:
    print_eeg_log = cp.logger("EEG")

def init():
    if not config.eeg_on or not parallel_available or config.exp_info['demo']:
        return EEGplaceholder()
    return EEG()

class EEG:
    
    def __init__(self):
        print(parallel.ParallelPort)
        parallel.setPortAddress(config.eeg_port_address)
        print(parallel.ParallelPort)
        self.para_port = parallel.ParallelPort(address=config.eeg_port_address)
        self.para_port.setData(0)
        print_eeg_log("Initialized")
    
    def send_trigger(self, trigger_code):  
        print_eeg_log(f"Sending trigger: {trigger_code}")
        self.para_port.setData(trigger_code)
        core.wait(0.1)
        self.para_port.setData(0)

class EEGplaceholder:

    def __init__(self):
        print_eeg_log("Initialized as placeholder")

    def send_trigger(self, trigger_code):
        print_eeg_log(f"Sending trigger: {trigger_code}")
