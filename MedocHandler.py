
"""
Medoc pain device interface.
Handles TCP communication with Medoc pain machine.
"""

from typing import Dict
import config
from psychopy import core, event
import socket, struct, time, binascii
from dataclasses import dataclass
import random
import dataRecordHandler
import LED
import coloredPrint as cp
from medockResponse import MedocResponse, check_for_escape, _build_command
# ===== פרוטוקול Open Control (על פי הדוגמה שצרפת) =====

if config.exp_info['demo'] or not config.medoc_on:
    print_medoc_log = cp.logger("MEDOC", place_holder=True)
else:
    print_medoc_log = cp.logger("MEDOC")

def init(log: dataRecordHandler.DataRecordHandler = None):
    if not config.medoc_on or config.exp_info['demo']:
        return Medocplaceholder(log=log)
    return Medoc(log=log)

class Medoc:
    def __init__(self, log: dataRecordHandler.DataRecordHandler = None, host: str = config.medoc_host, port: int = config.medoc_port, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.program = config.medoc_programs.get(config.exp_info['session'], '00010001')
        self.log = log
        self.last_temp = config.medoc_base_temperature
        print_medoc_log("Initialized")

    def _xfer(self, payload: bytes) -> MedocResponse:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(self.timeout)
            s.connect((self.host, self.port))
            s.sendall(payload)
            # קריאה ראשונה
            buf = s.recv(1024)
            # המשך קריאה עד שאין עוד (הדוגמה המקורית קראה עוד 17-בייטים בלולאה)
            while True:
                try:
                    more = s.recv(4096)
                except socket.timeout:
                    break
                if not more:
                    break
                buf += more
        return MedocResponse.parse(buf)

    # ===== פקודות נוחות =====

    def cmd(self, command: str, **kwargs) -> MedocResponse:
        respones =  self._xfer(_build_command(command, **kwargs))
        if command != 'GET_STATUS' and command != 'TRIGGER':
            self.log.medoc_event(respones)
            self.log.event("MEDOC_" + command, temperature=respones.temperature_c)
        print_medoc_log(f"Command: {command} sent")
        if config.debug:
            cp.print_debug(f" - Command: {command} response: {respones}")
        return respones

    def get_status(self) -> MedocResponse:
        response = self._xfer(_build_command('GET_STATUS'))
        self.log.medoc_event(response)
        return response

    def select_tp(self, program_bits: str) -> MedocResponse:
        # program_bits דוגמת '00011100'
        return self.cmd('SELECT_TP', param=program_bits)

    def start(self) -> MedocResponse:
        return self.cmd('START')

    def pause(self) -> MedocResponse:
        return self.cmd('PAUSE')

    def trigger(self, log: bool = True) -> MedocResponse:
        response = self.cmd('TRIGGER')
        if log:
            self.log.medoc_event(response)
            self.log.event("MEDOC_TRIGGER", temperature=response.temperature_c)
        return response

    def stop(self) -> MedocResponse:
        return self.cmd('STOP')

    def abort(self) -> MedocResponse:
        return self.cmd('ABORT')

    def yes(self) -> MedocResponse:
        return self.cmd('YES')

    def no(self) -> MedocResponse:
        return self.cmd('NO')

    def keyup(self) -> MedocResponse:
        return self.cmd('KEYUP')

    def covas(self, value: int) -> MedocResponse:
        # שלח ערך 0..255
        return self.cmd('COVAS', param=value)

    def vas(self, value: int) -> MedocResponse:
        # שלח ערך 0..10 (לפי הדוגמה)
        return self.cmd('VAS', param=value)

    def t_up(self, delta_c: float) -> MedocResponse:
        # נשלח Δ°C * 100 כשלם
        return self.cmd('T_UP', param=float(delta_c))

    def t_down(self, delta_c: float) -> MedocResponse:
        return self.cmd('T_DOWN', param=float(delta_c))

    def wait(self, seconds: float, log_medoc: bool = True):
        if log_medoc:
            for i in range(int(seconds*100)):
                core.wait(0.01)
                self.get_status()
        else:
            core.wait(seconds)
        event.clearEvents()

    
    def wait_until_status_is(self, status: str):
        print_medoc_log(f"Waiting until status is: {status}")
        response = self.get_status()
        while response.system_state != status:
            core.wait(0.01)
            response = self.get_status()
            if check_for_escape():
                self.abort()
                cp.print_error("Escape key pressed, aborting session")
                core.quit()

    def wait_until_test_state_is(self, test_state: str):
        print_medoc_log(f"Waiting until test state is: {test_state}")
        response = self.get_status()
        while response.test_state != test_state:
            core.wait(0.01)
            response = self.get_status()
            if check_for_escape():
                self.abort()
                cp.print_error("Escape key pressed, aborting session")
                core.quit()

    def wait_until_temperature_is_below(self, temperature: float):
        print_medoc_log(f"Waiting until temperature is below: {temperature}")
        response = self.get_status()
        while response.temperature_c > temperature:
            core.wait(0.01)
            response = self.get_status()
            if check_for_escape():
                self.abort()
                cp.print_error("Escape key pressed, aborting session")
                core.quit()

    def start_thermal_program(self):
        print_medoc_log(f"Starting thermal program {self.program}")
        self.select_tp(self.program) #select the program
        self.wait(1)
        self.start() #start pretest
        self.wait(1)
        self.start() #start test
        self.wait(1)

    def set_program(self, program: str):
        print_medoc_log(f"Setting program to: {program}")
        self.program = config.medoc_programs.get(program, '00010001')

    def start_threshold_trial(self):
        print_medoc_log("Starting threshold trial")
        self.trigger()

    def stop_threshold_trial(self, trial_num: int):
        print_medoc_log(f"Stopping threshold trial {trial_num}")
        temperature = self.yes().temperature_c #trigger medoc yes to stop trail
        return temperature

    def skip_initial_pain_stimulus(self):
        """
        In the vas search program, the first trail is fixed,
        so we want to skip it to choose the temperature by the user
        """
        print_medoc_log("Skipping initial pain stimulus")
        self.trigger(log=False)
        self.wait(config.trail_dur_sec[config.exp_info['session']] + 1)

    def pain_stimulus_trial(self, temperature: float):
        print_medoc_log(f"Pain stimulus trial at temperature: {temperature}")
        t_up = temperature - self.last_temp 
        self.last_temp = temperature
        self.t_up(t_up)
        self.wait(0.2)
        self.trigger()
        self.wait(1)
        self.wait_until_temperature_is_below(config.medoc_base_temperature + 0.5)

class Medocplaceholder(Medoc):

    def _mock_response(self, command: str, test_state: str, temperature: float = None):
        if temperature is None:
            temperature = self._random_temperature()
        return MedocResponse(length=0, timestamp=round(time.time(), 6), command_id=command, system_state=0, test_state=test_state, resp_code=0, test_time_s=0, temperature_c=temperature, covas=0, yes=0, no=0, message=b'')

    def _random_temperature(self):
        return random.uniform(30, 50)
    
    def cmd(self, command: str, **kwargs) -> MedocResponse:
        respones =  self._mock_response(command, 'READY')
        if command != 'GET_STATUS' and command != 'TRIGGER':
            self.log.medoc_event(respones)
            self.log.event("MEDOC_" + command, temperature=respones.temperature_c)
        print_medoc_log(f"Command: {command} sent")
        return respones

    def get_status(self):
        response = self._mock_response('GET_STATUS', 'READY')
        self.log.medoc_event(response)
        return response



def sanity_check():
    # TODO: add sanity check
    pass

if __name__ == '__main__':
     
    sanity_check()
