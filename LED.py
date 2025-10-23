"""
LED placeholder module for testing without hardware.
"""
import serial
import serial.tools.list_ports as list_ports
from psychopy import core
import config

def init():
    """Initialize LED."""
    print("LED initialized")
    return LED()

class LED:
    """LED functionality."""
    
    def __init__(self):
        self.port_name = list_ports.comports()[0].device
        self.baud_rate = config.led_baud_rate
        self.timeout = config.led_timeout
        self.on_duration_in_sec = config.led_on_duration_in_sec

    def flash(self):
        if not config.led_on:
            return
        ser = serial.Serial(self.port_name, self.baud_rate, timeout=self.timeout)
        for _ in range(self.on_duration_in_sec):
            ser.writelines(b'H')  # send a byte
            core.wait(0.5)  # wait 0.5 seconds
            ser.writelines(b'L')  # send a byte
            core.wait(0.5)
        ser.close()
