
import serial
import serial.tools.list_ports as list_ports
from psychopy import core
import config
import coloredPrint as cp


def init():
    if not config.led_on or config.exp_info['demo']:
        return LEDplaceholder()
    return LED()

class LED:
    """LED functionality."""
    
    def __init__(self):
        self.port_name = list_ports.comports()[0].device
        self.baud_rate = config.led_baud_rate
        self.timeout = config.led_timeout
        self.on_duration_in_sec = config.led_on_duration_in_sec
        cp.print_success("[LED]", end="")
        print(" - Initialized")

    def flash(self):
        cp.print_info("[LED]", end="")
        print(" - Flashing")
        ser = serial.Serial(self.port_name, self.baud_rate, timeout=self.timeout)
        for _ in range(self.on_duration_in_sec):
            ser.writelines(b'H')  # send a byte
            core.wait(0.5)  # wait 0.5 seconds
            ser.writelines(b'L')  # send a byte
            core.wait(0.5)
        ser.close()
        cp.print_success("[LED]", end="")
        print(" - Flashing complete")


class LEDplaceholder:
    """LED placeholder."""
    
    def __init__(self):
        self.on_duration_in_sec = config.led_on_duration_in_sec
        cp.print_warning("[LED] [placeholder]", end="")
        print(" - Initialized")

    def flash(self):
        cp.print_warning("[LED] [placeholder]", end="")
        print(" - Flashing")
        core.wait(self.on_duration_in_sec)
        cp.print_warning("[LED] [placeholder]", end="")
        print(" - Flashing complete")