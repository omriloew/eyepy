
"""
Medoc pain device interface.
Handles TCP communication with Medoc pain machine.
"""

import socket
import time
from typing import Optional, Dict, Any
import config
from psychopy import core


class Medoc:
    """Medoc pain device interface."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.host = config.medoc_host
        self.port = config.medoc_port
        self.default_program = config.medoc_default_program
        self.programs = config.medoc_programs
        
        # Current active program
        self.current_program = self.default_program
        
        # TCP connection
        self.socket = None
        
        if not dry_run:
            self._init_connection()
    
    def _init_connection(self):
        """Initialize TCP connection to Medoc."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # 5 second timeout
            self.socket.connect((self.host, self.port))
            print(f"Connected to Medoc at {self.host}:{self.port}")
        except Exception as e:
            print(f"Warning: Could not connect to Medoc: {e}")
            self.socket = None
    
    def _send_command(self, cmd: str) -> str:
        """Send command to Medoc and return response."""
        if self.dry_run:
            print(f"[DRY-RUN] Medoc command: {cmd}")
            return "OK"
        
        if not self.socket:
            print(f"[MOCK] Medoc command: {cmd}")
            return "OK"
        
        try:
            # Send command
            self.socket.send(f"{cmd}\r\n".encode('ascii'))
            
            # Receive response
            response = self.socket.recv(1024).decode('ascii').strip()
            return response
        except Exception as e:
            print(f"Warning: Medoc communication error: {e}")
            return "ERROR"
    
    def select_tp(self, program: str = None) -> bool:
        """Select thermal program."""
        if program is None:
            program = self.default_program
        
        # Check if program is a named program
        if program in self.programs:
            program_code = self.programs[program]
        else:
            # Assume it's a direct program code
            program_code = program
        
        cmd = f"SELECT_TP {program_code}"
        response = self._send_command(cmd)
        
        if "OK" in response:
            self.current_program = program_code
            print(f"Selected Medoc program: {program} ({program_code})")
        
        return "OK" in response
    
    def get_program(self, name: str = None) -> str:
        """Get program code by name."""
        if name is None:
            return self.current_program
        
        return self.programs.get(name, name)
    
    def list_programs(self) -> Dict[str, str]:
        """List all available programs."""
        return self.programs.copy()
    
    def start(self) -> bool:
        """Start pain stimulation."""
        cmd = "START"
        response = self._send_command(cmd)
        return "OK" in response
    
    def stop(self) -> bool:
        """Stop pain stimulation."""
        cmd = "STOP"
        response = self._send_command(cmd)
        return "OK" in response

    def yes(self) -> bool:
        """Set yes."""
        cmd = "YES"
        response = self._send_command(cmd)
        print(f"Medoc command: {cmd} responded with: {response}")
        return "OK" in response

    def set_intensity(self, intensity: int) -> bool:
        """Set pain intensity."""
        cmd = f"SET_INTENSITY {intensity}"
        response = self._send_command(cmd)
        return "OK" in response

    def trigger(self) -> bool:
        """Trigger pain stimulation."""
        cmd = "TRIGGER"
        response = self._send_command(cmd)
        return "OK" in response
    
    def get_status(self) -> Dict[str, Any]:
        """Get device status."""
        cmd = "STATUS"
        response = self._send_command(cmd)
        
        # Parse status response
        status = {
            'connected': self.socket is not None,
            'response': response
        }
        
        # Try to parse specific status fields
        if "OK" in response:
            status['ready'] = True
        else:
            status['ready'] = False
        
        return status
    
    def send(self, cmd: str, param: Any = None) -> str:
        """Send custom command to Medoc."""
        if param is not None:
            full_cmd = f"{cmd} {param}"
        else:
            full_cmd = cmd
        
        return self._send_command(full_cmd)
    
    def close(self):
        """Close Medoc connection."""
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                print(f"Warning: Could not close Medoc connection: {e}") 

    def start_program(self, program: str):
        """Start program."""
        self.select_tp(program)
        self.start()

    def stop_program(self, program: str):
        """Stop program."""
        self.select_tp(program)
        self.stop()
    
    def wait_for_trail_start(self):
        """Wait for trail start."""
        if self.dry_run:
            core.wait(0.5)
            return
        while self.get_status()['response'] != "RUNNING":
            time.sleep(0.01)
        print("Medoc running")

    def wait_for_trail_end(self):
        """Wait for trail end."""
        if self.dry_run:
            core.wait(0.5)
            return
        self.wait_for_trail_start()
        while self.get_status()['response'] != "READY":
            time.sleep(0.01)
        print("Medoc ready")