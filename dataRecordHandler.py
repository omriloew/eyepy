import csv
import os
import json
from datetime import datetime
import config
from psychopy import core, gui
import coloredPrint as cp

from medockResponse import MedocResponse

print_log = cp.logger("LOG")

class DataRecordHandler:
    """
    Handles recording and logging of experimental data to CSV files.
    Tracks trials, events, pain ratings, and timestamps for experimental sessions.
    """
    
    def __init__(self,clock, led, eeg, el=None):
        """
        Initialize the data record handler.
        
        Args:
            clock: Clock object
            led: LED object
            el: Eyelink object
            eeg: EEG object
        """
        self.participant_id = config.exp_info['participant']
        self.session_type = config.exp_info['session']
        self.session_number = config.exp_info['session_number']
        self.exp_info = config.exp_info
        self.session_start_time = None
        self.trial_data = []
        self.event_log = []
        self.medoc_event_log = []
        self.el = el
        self.eeg = eeg
        self.led = led
        self.clock = clock
        self.inter_session_data = {'temperatures': {}, 'ratings': {}, 'durations': {}}
        

        self.output_dir = config.exp_info['output_dir']
        self.trials_filename = self._generate_filename('trials')
        self.events_filename = self._generate_filename('events')
        self.exp_info_filename = self._generate_filename('exp_info')
        self.medoc_events_filename = self._generate_filename('medoc_events')
        # Save experiment info immediately
        self.save_exp_info()
        
    def _generate_filename(self, data_type):
        """Generate a unique filename with timestamp."""
        return f"{self.participant_id}_{self.session_type}_{data_type}.csv"
    
    def start_session(self):
        """Mark the start of a session."""
        self.session_start_time = self.clock.getTime()
        self.event(config.session_start_msg)
        core.wait(0.5)
        for _ in range(config.led_num_flashes):
            self.event(config.led_on_msg)
            self.led.flash()
            self.event(config.led_off_msg)
            core.wait(config.led_interval_in_sec)
        print_log(f"Session started: {self.session_type} for participant {self.participant_id}")
    
    def load_session_trials_data(self, session_type: str):
        """Load session data from CSV files."""
        filename = f"{self.participant_id}_{session_type}_trials.csv"
        output_dir = os.path.join(config.datapath, self.participant_id, str(self.session_number), session_type)
        filepath = os.path.join(output_dir, filename)
        temperature_data = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    temperature_data.append(row['temperature'])
        else:
            temperature_data = config.manual_temperature_data[session_type]
        print_log(f"Loaded temperature data: {temperature_data}")
        return temperature_data

    
    def trial(self, trial_number, pain_rating=None, reaction_time=None, 
                  wait_time=None, trial_duration=None, temperature=None, **kwargs):
        """
        Log data for a single trial.
        
        Args:
            trial_number: The trial number (1-indexed)
            pain_rating: Pain rating from VAS (0-100 or None)
            reaction_time: Time taken to respond to VAS in seconds
            wait_time: Wait time before this trial in seconds
            trial_duration: Duration of the trial in seconds
            **kwargs: Additional trial-specific data
        """
        timestamp = self.clock.getTime()
        time_from_session_start = timestamp - self.session_start_time if self.session_start_time else 0
        
        trial_entry = {
            'participant_id': self.participant_id,
            'session_type': self.session_type,
            'trial_number': trial_number,
            'pain_rating': pain_rating,
            'reaction_time': reaction_time,
            'temperature': temperature,
            'wait_time_before_trial': wait_time,
            'trial_duration': trial_duration,
            'time_from_session_start': round(time_from_session_start, 6),
            **kwargs
        }
        print_log(f"Trial {trial_number} logged")
        if config.debug:
            cp.print_debug(f"Trial {trial_number} logged - {trial_entry}")
        self.trial_data.append(trial_entry)

    
    def event(self, event_type ,event_label=None, event_data=None, temperature=None):
        """
        Log a general event with timestamp.
        
        Args:
            event_type: Type of event (e.g., 'trial_start', 'pain_detected', 'medoc_trigger')
            event_data: Dictionary of additional event data
            led_flash: Whether to trigger an LED flash for this event
        """
        if event_label is None:
            event_label = event_type
        timestamp = self.clock.getTime()
        time_from_session_start = timestamp - self.session_start_time if self.session_start_time else 0
        event_code = config.events[event_type]
        
        event_entry = {
            'time_stamp': round(time_from_session_start, 6),
            'event_label': event_type,
            '_participant_id': self.participant_id,
            '_session_type': self.session_type,
            '_event_code': event_code,
            '_event_message': event_label,
            '_temperature': temperature,
        }
        
        # Add event data if provided
        if event_data:
            event_entry.update(event_data)
        
        self.event_log.append(event_entry)
        print_log(f"Event {event_type} logged")
        self.eeg.send_trigger(event_code)
        self.el.send_message(event_label)
    
    def medoc_event(self, medoc_response: MedocResponse):
        """
        Log a Medoc event with timestamp.
        Args:
            medoc_response: MedocResponse object
        """
        medoc_event_entry = {
            'time_stamp': round(medoc_response.timestamp, 6),
            'command_id': medoc_response.command_id,
            'system_state': medoc_response.system_state,
            'test_state': medoc_response.test_state,
            'resp_code': medoc_response.resp_code,
            'test_time_s': medoc_response.test_time_s,
            'temperature_c': medoc_response.temperature_c,
            'covas': medoc_response.covas,
            'yes': medoc_response.yes,
            'no': medoc_response.no,
            '_participant_id': self.participant_id,
            '_session_type': self.session_type, 
        }
        self.medoc_event_log.append(medoc_event_entry)

    
    def finish_session(self):
        """Mark the end of a session and save all data."""
        self.event(config.session_end_msg)
        core.wait(0.5)
        for _ in range(config.led_num_flashes):
            self.event(config.led_on_msg)
            self.led.flash()
            self.event(config.led_off_msg)
            core.wait(config.led_interval_in_sec)
        print_log(f"Session finished: {self.session_type}")
        self.save_all()
        
        session_duration = self.clock.getTime() - self.session_start_time if self.session_start_time else 0
        print_log(f"Session finished: {self.session_type}")
        print_log(f"Total duration: {session_duration:.2f} seconds")
        print_log(f"Total trials logged: {len(self.trial_data)}")
        print_log(f"Total events logged: {len(self.event_log)}")
    
    def save_trials(self):
        """Save trial data to CSV file."""
        if not self.trial_data:
            cp.print_error("[LOG] - No trial data to save.")
            return
        
        filepath = os.path.join(self.output_dir, self.trials_filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = self.trial_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trial_data)
        
        print_log(f"Trial data saved to: {filepath}")
    
    def save_events(self):
        """Save event log to CSV file."""
        if not self.event_log:
            cp.print_error("[LOG] - No event data to save.")
            return
        
        filepath = os.path.join(self.output_dir, self.events_filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = self.event_log[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.event_log)
        
        print_log(f"Event data saved to: {filepath}")

    def save_medoc_events(self):
        """Save event log to CSV file."""
        if not self.medoc_event_log:
            cp.print_error("[LOG] - No medoc event data to save.")
            return
        
        filepath = os.path.join(self.output_dir, self.medoc_events_filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = self.medoc_event_log[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.medoc_event_log)
        
        print_log(f"Medoc event data saved to: {filepath}")
    
    def save_exp_info(self):
        """Save experiment info to both CSV and JSON files."""
        if not self.exp_info:
            cp.print_error("[LOG] - No experiment info to save.")
            return
        
        # Add timestamp to experiment info
        current_time = self.clock.getTime() if self.clock else 0
        timestamp = datetime.fromtimestamp(current_time).isoformat()
        
        # Save as CSV
        csv_filepath = os.path.join(self.output_dir, self.exp_info_filename)
        
        # Flatten the exp_info for CSV format
        flattened_info = {}
        for key, value in self.exp_info.items():
            # Convert tuples and lists to strings for CSV
            if isinstance(value, (tuple, list)):
                flattened_info[key] = str(value)
            else:
                flattened_info[key] = value
        
        # Add timestamp to flattened info
        flattened_info['timestamp'] = timestamp
        flattened_info['time_from_session_start'] = current_time - self.session_start_time if self.session_start_time else 0
        
        with open(csv_filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=flattened_info.keys())
            writer.writeheader()
            writer.writerow(flattened_info)
        
        print_log(f"Experiment info (CSV) saved to: {csv_filepath}")
        
        # Save as JSON (more readable for complex data)
        json_filename = self.exp_info_filename.replace('.csv', '.json')
        json_filepath = os.path.join(self.output_dir, json_filename)
        
        # Add timestamp to JSON data
        json_data = self.exp_info.copy()
        json_data['timestamp'] = timestamp
        json_data['time_from_session_start'] = current_time - self.session_start_time if self.session_start_time else 0
        
        with open(json_filepath, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=4, default=str)
        
        print_log(f"Experiment info (JSON) saved to: {json_filepath}")
    
    def save_all(self):
        """Save trial data, event log, and experiment info."""
        self.save_trials()
        self.save_events()
        self.save_medoc_events()
        self.save_exp_info()
    
    def _flatten_exp_info(self):
        """Flatten experiment info dictionary for CSV columns."""
        if not self.exp_info:
            return {}
        
        flattened = {}
        for key, value in self.exp_info.items():
            # Skip nested objects, keep only simple types
            if isinstance(value, (str, int, float, bool)):
                flattened[f'exp_{key}'] = value
        
        return flattened
    
    def get_trial_count(self):
        """Get the number of trials logged so far."""
        return len(self.trial_data)
    
    def get_event_count(self):
        """Get the number of events logged so far."""
        return len(self.event_log)
    
    def get_summary(self):
        """Get a summary of the session data."""
        summary = {
            'participant_id': self.participant_id,
            'session_number': self.session_number,
            'session_type': self.session_type,
            'total_trials': len(self.trial_data),
            'total_events': len(self.event_log),
        }
        
        # Add pain rating statistics if available
        ratings = [t['pain_rating'] for t in self.trial_data if t.get('pain_rating') is not None]
        if ratings:
            summary['mean_pain_rating'] = sum(ratings) / len(ratings)
            summary['min_pain_rating'] = min(ratings)
            summary['max_pain_rating'] = max(ratings)
        
        # Add reaction time statistics if available
        rts = [t['reaction_time'] for t in self.trial_data if t.get('reaction_time') is not None]
        if rts:
            summary['mean_reaction_time'] = sum(rts) / len(rts)
            summary['min_reaction_time'] = min(rts)
            summary['max_reaction_time'] = max(rts)
        
        return summary
    