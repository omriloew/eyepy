import csv
import os
import json
from datetime import datetime
import config
from psychopy import core

class DataRecordHandler:
    """
    Handles recording and logging of experimental data to CSV files.
    Tracks trials, events, pain ratings, and timestamps for experimental sessions.
    """
    
    def __init__(self,clock, led, el=None, eeg=None):
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
        self.exp_info = config.exp_info
        self.session_start_time = None
        self.trial_data = []
        self.event_log = []
        self.el = el
        self.eeg = eeg
        self.led = led
        self.clock = clock
        
        if not os.path.exists(config.datapath):
            os.makedirs(config.datapath)
        
        self.output_dir = os.path.join(config.datapath,self.session_type, self.participant_id)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            print(f"Output directory already exists: {self.output_dir}")
            participant_session_number = 1
            while os.path.exists(os.path.join(config.datapath, self.session_type, f"{self.participant_id}_{participant_session_number}")):
                participant_session_number += 1
            self.output_dir = os.path.join(config.datapath, self.session_type, f"{self.participant_id}_{participant_session_number}")
            os.makedirs(self.output_dir)
            print(f"Output directory created: {self.output_dir}")

        # Generate filenames
        self.trials_filename = self._generate_filename('trials')
        self.events_filename = self._generate_filename('events')
        self.exp_info_filename = self._generate_filename('exp_info')
        
        # Save experiment info immediately
        self.save_exp_info()
        
    def _generate_filename(self, data_type):
        """Generate a unique filename with timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{self.participant_id}_{self.session_type}_{data_type}_{timestamp}.csv"
    
    def start_session(self):
        """Mark the start of a session."""
        self.session_start_time = self.clock.getTime()
        self.event(config.session_start_msg, led_flash=False)
        core.wait(0.5)
        for _ in range(config.led_num_flashes):
            self.event(config.led_start_on_msg, led_flash=False)
            self.led.flash()
            self.event(config.led_start_off_msg, led_flash=False)
            core.wait(config.led_interval_in_sec)
        print(f"Session started: {self.session_type} for participant {self.participant_id}")

    
    def trial(self, trial_number, pain_rating=None, reaction_time=None, 
                  wait_time=None, trial_duration=None, **kwargs):
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
            'wait_time_before_trial': wait_time,
            'trial_duration': trial_duration,
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'time_from_session_start': round(time_from_session_start, 3),
            **kwargs
        }
        
        self.trial_data.append(trial_entry)
        print(f"Trial {trial_number} logged - Rating: {pain_rating}, RT: {reaction_time}")
    
    def event(self, event_type ,event_label=None, event_data=None, led_flash=True):
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
            'participant_id': self.participant_id,
            'session_type': self.session_type,
            'event_type': event_type,
            'event_code': event_code,
            'event_label': event_label,
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'time_from_session_start': round(time_from_session_start, 3),
            'led': led_flash,
        }
        
        # Add event data if provided
        if event_data:
            event_entry.update(event_data)
        
        self.event_log.append(event_entry)

        if self.eeg is not None:
            self.eeg.send_trigger(event_code)

        if self.el is not None:
            self.el.send_message(event_label + "_" +str(event_code))

        if led_flash and self.led is not None:
            self.led.flash()
            print(f"LED flash triggered for event: {event_type}")
    
    def mark(self, label, code):
        """
        Log a marker event (for EEG, Eyelink, etc.).
        Args:
            label: Marker label/description
            code: Numeric code for the marker
        """
        self.event(config.marker_msg, f"{label}", {'label': label, 'code': code})
    
    def finish_session(self):
        """Mark the end of a session and save all data."""
        self.event(config.session_end_msg, led_flash=False)
        core.wait(0.5)
        for _ in range(config.led_num_flashes):
            self.event(config.led_finish_on_msg, led_flash=False)
            self.led.flash()
            self.event(config.led_finish_off_msg, led_flash=False)
            core.wait(config.led_interval_in_sec)
        self.save_all()
        
        session_duration = self.clock.getTime() - self.session_start_time if self.session_start_time else 0
        print(f"Session finished: {self.session_type}")
        print(f"Total duration: {session_duration:.2f} seconds")
        print(f"Total trials logged: {len(self.trial_data)}")
        print(f"Total events logged: {len(self.event_log)}")
    
    def save_trials(self):
        """Save trial data to CSV file."""
        if not self.trial_data:
            print("No trial data to save.")
            return
        
        filepath = os.path.join(self.output_dir, self.trials_filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = self.trial_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trial_data)
        
        print(f"Trial data saved to: {filepath}")
    
    def save_events(self):
        """Save event log to CSV file."""
        if not self.event_log:
            print("No event data to save.")
            return
        
        filepath = os.path.join(self.output_dir, self.events_filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = self.event_log[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.event_log)
        
        print(f"Event data saved to: {filepath}")
    
    def save_exp_info(self):
        """Save experiment info to both CSV and JSON files."""
        if not self.exp_info:
            print("No experiment info to save.")
            return
        
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
        
        with open(csv_filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=flattened_info.keys())
            writer.writeheader()
            writer.writerow(flattened_info)
        
        print(f"Experiment info (CSV) saved to: {csv_filepath}")
        
        # Save as JSON (more readable for complex data)
        json_filename = self.exp_info_filename.replace('.csv', '.json')
        json_filepath = os.path.join(self.output_dir, json_filename)
        
        with open(json_filepath, 'w') as jsonfile:
            json.dump(self.exp_info, jsonfile, indent=4, default=str)
        
        print(f"Experiment info (JSON) saved to: {json_filepath}")
    
    def save_all(self):
        """Save trial data, event log, and experiment info."""
        self.save_trials()
        self.save_events()
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
