import mne
import pandas as pd

def edf_to_df(file_path: str) -> pd.DataFrame:
    """
    טוענת קובץ EDF, ממירה ל-DataFrame ומדפיסה את הראש.
    תומכת גם בקבצי Eyelink EDF וגם בקבצי EDF סטנדרטיים.
    מחזירה את ה-DataFrame.
    """
    # נסה לזהות אם זה קובץ Eyelink EDF
    try:
        with open(file_path, 'rb') as f:
            header = f.read(100).decode('latin-1', errors='ignore')
            is_eyelink = 'SR_RESEARCH' in header or 'EYELINK' in header.upper()
    except:
        is_eyelink = False
    
    if is_eyelink:
        # קובץ Eyelink EDF - שימוש ב-eyelinkio
        try:
            from eyelinkio import read_edf
            edf = read_edf(file_path)
            df_dict = edf.to_pandas()
            
            # Get samples DataFrame
            df = df_dict['samples'].copy()
            
            # Add time_stamp column from edf['times']
            df['time_stamp'] = edf['times']
            
            # Initialize event_label column (empty for samples)
            df['event_label'] = ''
            
            # Get messages/events and merge them into samples
            if 'discrete' in df_dict and 'messages' in df_dict['discrete']:
                messages = df_dict['discrete']['messages'].copy()

                #   - Filters OUT: !CAL, VALIDATE, RECCFG, ELCLCFG, GAZE_COORDS, !MODE
                #   - Keeps: Trigger_*, SYNCTIME, Exp start, Sentence_*, and other custom messages
                filter_pattern = '!CAL|VALIDATE|RECCFG|ELCLCFG|GAZE_COORDS|!MODE'
                event_messages = messages[~messages['msg'].str.contains(filter_pattern, na=False, regex=True)]
                
                # For each message, find the closest sample and add the event label
                for idx, row in event_messages.iterrows():
                    msg_time = row['stime']
                    msg_text = str(row['msg']).strip()
                    
                    # Find the closest sample index
                    closest_idx = (df['time_stamp'] - msg_time).abs().idxmin()
                    
                    # Add event label (append if there's already one)
                    if df.loc[closest_idx, 'event_label']:
                        df.loc[closest_idx, 'event_label'] += f"; {msg_text}"
                    else:
                        df.loc[closest_idx, 'event_label'] = msg_text
            
            print("Eyelink EDF file loaded successfully")
            print(f"Eye recorded: {edf.info.get('eye', 'Unknown')}")
            print(f"Sampling rate: {edf.info['sfreq']} Hz")
            print(f"Number of samples: {len(df)}")
            if 'discrete' in df_dict and 'messages' in df_dict['discrete']:
                total_messages = len(df_dict['discrete']['messages'])
                event_messages = len(df_dict['discrete']['messages'][~df_dict['discrete']['messages']['msg'].str.contains('!CAL|VALIDATE|RECCFG|ELCLCFG|GAZE_COORDS|!MODE', na=False, regex=True)])
                print(f"Total messages: {total_messages} (events: {event_messages})")
            print("\nSamples with events:")
            samples_with_events = df[df['event_label'] != '']
            if len(samples_with_events) > 0:
                print(samples_with_events.head(10))
            else:
                print("No events found in samples")
            return df
        except ImportError:
            raise ImportError(
                "eyelinkio package is required to read Eyelink EDF files.\n"
                "Install it with: pip install eyelinkio"
            )
        except Exception as e:
            raise ValueError(f"Failed to read Eyelink EDF file: {e}")
    else:
        # קובץ EDF סטנדרטי - שימוש ב-mne
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            df = raw.to_data_frame()
            print(df.head())
            return df
        except Exception as e:
            raise ValueError(f"Failed to read EDF file: {e}\nFile might be corrupted or in an unsupported format.")

def main():
    df = edf_to_df('rawFilesToTestWith/1011b4c2.edf')
    print(df.head())

if __name__ == "__main__":
    main()