import os
import re
import struct
import subprocess
import pandas as pd
import numpy as np
import mne
import xml.etree.ElementTree as ET
from datetime import datetime
from mffpy.reader import Reader

import config as config

def _extract_pns_channel_names(mff_dir: str) :
    """
    Best-effort extraction of PNS/Physio channel names from pnsSet.xml.
    Works across slightly different EGI/PhysioBox XML variants by grabbing common tags.
    """
    pns_xml = os.path.join(mff_dir, "pnsSet.xml")
    if not os.path.exists(pns_xml):
        return None

    text = open(pns_xml, "r", encoding="utf-8", errors="ignore").read()

    # Common tags/attributes that hold channel labels
    # We intentionally do multiple patterns and then de-duplicate while keeping order.
    patterns = [
        r"<name>\s*([^<]+?)\s*</name>",
        r"<label>\s*([^<]+?)\s*</label>",
        r'(?i)name\s*=\s*"([^"]+)"',
        r'(?i)label\s*=\s*"([^"]+)"',
        r"<displayName>\s*([^<]+?)\s*</displayName>",
    ]

    found = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            s = m.group(1).strip()
            # filter obvious non-channel garbage
            if not s or len(s) > 80:
                continue
            found.append(s)

    # de-duplicate preserving order
    seen = set()
    uniq = []
    for s in found:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    return uniq or None


def mff_pns_to_df(mff_path: str) :
    r = Reader(mff_path)

    samples = r.get_physical_samples()
    if "PNSData" not in samples:
        raise ValueError(f"No PNSData in get_physical_samples(). Keys: {list(samples.keys())}")

    data, t0 = samples["PNSData"]  # data: (n_channels, n_samples)

    if data.ndim != 2:
        raise ValueError(f"Unexpected PNSData shape: {data.shape}")

    n_channels, n_samples = data.shape
    sfreq = r.sampling_rates.get("PNSData", None)

    # channel names from pnsSet.xml (best effort)
    ch_names = _extract_pns_channel_names(mff_path)

    # If extraction returned too many/too few, fall back to generic names
    if not ch_names or len(ch_names) != n_channels:
        ch_names = [f"PNS_{i+1}" for i in range(n_channels)]

    df = pd.DataFrame(data.T, columns=ch_names)

    # add time column (seconds) if we know sfreq
    if sfreq:
        df.insert(0, "time_sec", (np.arange(n_samples) / float(sfreq)) + float(t0))

    return df, float(sfreq) if sfreq else None


def export_emg_leg_csv(mff_path: str, out_csv: str) -> pd.DataFrame:
    df, _ = mff_pns_to_df(mff_path)

    # Filter columns likely to be EMG LEG (adjust keywords if needed)
    keywords = ("emg", "leg")
    cols = [c for c in df.columns if any(k in str(c).lower() for k in keywords)]

    # If we didn't find, export everything (still useful) but tell you explicitly
    if not cols:
        # export full PNS
        df.to_csv(out_csv, index=False)
        print(f"[WARN] No EMG/LEG columns matched. Exported ALL PNSData columns to: {out_csv}")
        print("Columns:", list(df.columns))
        return df

    df_emg = df[["time_sec"] + cols] if "time_sec" in df.columns else df[cols]
    df_emg.to_csv(out_csv, index=False)
    print(f"Exported EMG/LEG columns to: {out_csv}")
    print("Exported columns:", ["time_sec"] + cols if "time_sec" in df.columns else cols)
    return df_emg

EDF2ASC_EXE = r"C:\Program Files (x86)\SR Research\EyeLink\EDF_Access_API\Example\edf2asc.exe"

FILTER_PATTERN = r"!CAL|VALIDATE|RECCFG|ELCLCFG|GAZE_COORDS|!MODE"


def _is_eyelink_edf(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(300).decode("latin-1", errors="ignore")
        return ("SR_RESEARCH" in header) or ("EYELINK" in header.upper())
    except Exception:
        return False


def _edf_to_asc(edf_path: str, edf2asc_exe: str = EDF2ASC_EXE) -> str:
    """
    Convert EyeLink EDF -> ASC using SR Research edf2asc.
    Creates <edf_path_without_ext>.asc next to the EDF.
    """
    asc_path = os.path.splitext(edf_path)[0] + ".asc"
    if os.path.exists(asc_path):
        return asc_path

    cmd = [edf2asc_exe, "-y", edf_path]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if not os.path.exists(asc_path):
        raise FileNotFoundError(f"ASC was not created: {asc_path}")

    return asc_path


def _asc_to_df(asc_path: str, filter_pattern: str = FILTER_PATTERN) -> pd.DataFrame:
    """
    Parse EyeLink ASC:
    - samples: lines starting with a timestamp (digits)
    - messages: lines starting with 'MSG <time> <text...>'
    Returns samples DF with time_stamp + event_label merged to nearest sample.
    """
    samples = []
    messages = []

    msg_re = re.compile(r"^MSG\s+(\d+)\s+(.*)$")

    with open(asc_path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = msg_re.match(line)
            if m:
                t = int(m.group(1))
                txt = m.group(2).strip()
                messages.append((t, txt))
                continue

            if line[0].isdigit():
                samples.append(line.split())

    if not samples:
        raise ValueError(f"No samples found in ASC: {asc_path}")

    # Build DataFrame with generic columns c0..cN (format can vary by recording config)
    max_len = max(len(r) for r in samples)
    cols = [f"c{i}" for i in range(max_len)]
    samples = [r + [""] * (max_len - len(r)) for r in samples]

    df = pd.DataFrame(samples, columns=cols)
    df["time_stamp"] = df["c0"].astype(int)
    df["event_label"] = ""

    def _to_num(col):
        return pd.to_numeric(col.replace(".", pd.NA), errors="coerce")

    has_left = "c3" in df.columns
    has_right = "c6" in df.columns and _to_num(df["c6"]).notna().sum() > 0

    if has_left and has_right:
        # binocular
        df["ps_left"] = _to_num(df["c3"])
        df["ps_right"] = _to_num(df["c6"])
        df["ps"] = np.nanmean(
            np.vstack([
                df["ps_left"].astype(float),
                df["ps_right"].astype(float)
            ]),
            axis=0
        )
    else:
        # monocular
        df["ps"] = _to_num(df["c3"])

    if messages:
        msg_df = pd.DataFrame(messages, columns=["stime", "msg"])
        msg_df = msg_df[~msg_df["msg"].str.contains(filter_pattern, na=False, regex=True)]
        msg_df = msg_df.sort_values("stime").reset_index(drop=True)

        # Efficient nearest merge
        df_sorted = df.sort_values("time_stamp").reset_index(drop=False)  # keep original index
        msg_df["stime"] = msg_df["stime"].astype("int64")
        df_sorted["time_stamp"] = df_sorted["time_stamp"].astype("int64")
        merged = pd.merge_asof(
            msg_df,
            df_sorted,
            left_on="stime",
            right_on="time_stamp",
            direction="nearest"
        )

        for _, row in merged.iterrows():
            i = int(row["index"])
            txt = str(row["msg"]).strip()
            if df.at[i, "event_label"]:
                df.at[i, "event_label"] += f"; {txt}"
            else:
                df.at[i, "event_label"] = txt

    # Remove the first row (keep your behavior)
    df = df.iloc[1:].reset_index(drop=True)
    return df


def _add_mff_events_to_df(df: pd.DataFrame, mff_path: str):
    """
    Parse MFF Events_*.xml and info.xml, map DIN codes to config labels,
    apply PNS shift from log, and assign event_label to nearest time_stamp.
    """
    info_path = os.path.join(mff_path, "info.xml")
    recording_start_time = None
    if os.path.exists(info_path):
        try:
            tree = ET.parse(info_path)
            root = tree.getroot()
            record_time_elem = root.find(".//{http://www.egi.com/info_mff}recordTime")
            if record_time_elem is not None and record_time_elem.text:
                time_str = record_time_elem.text
                if time_str.endswith("+0200"):
                    time_str = time_str.replace("+0200", "+02:00")
                elif time_str.endswith("-0200"):
                    time_str = time_str.replace("-0200", "-02:00")
                recording_start_time = datetime.fromisoformat(time_str)
        except Exception as e:
            print(f"Warning: Could not parse recording start time: {e}")

    pns_shift_seconds = 0.0
    log_files = [f for f in os.listdir(mff_path) if f.startswith("log_") and f.endswith(".txt")]
    if log_files:
        log_path = os.path.join(mff_path, log_files[0])
        try:
            with open(log_path, "r") as log_f:
                for line in log_f:
                    if "PNS shift:" in line:
                        if "ms" in line:
                            parts = line.split("PNS shift:")
                            if len(parts) > 1:
                                ms_str = parts[1].split("ms")[0].strip()
                                try:
                                    pns_shift_seconds = float(ms_str) / 1000.0
                                    break
                                except ValueError:
                                    pass
                        if "samples" in line and "=" in line:
                            # e.g. "PNS shift: 20 samples = 80 ms"
                            m = re.search(r"=\s*([\d.]+)\s*ms", line)
                            if m:
                                try:
                                    pns_shift_seconds = float(m.group(1)) / 1000.0
                                    break
                                except ValueError:
                                    pass
                            m = re.search(r"(\d+)\s*samples", line)
                            if m:
                                try:
                                    pns_shift_seconds = float(m.group(1)) / 250.0
                                    break
                                except ValueError:
                                    pass
        except Exception as e:
            print(f"Warning: Could not read PNS shift from log: {e}")

    code_to_label = {v: k for k, v in config.events.items()}
    event_files = [f for f in os.listdir(mff_path) if f.startswith("Events_") and f.endswith(".xml")]
    events = []

    for event_file in event_files:
        event_path = os.path.join(mff_path, event_file)
        try:
            tree = ET.parse(event_path)
            root = tree.getroot()
            for event_elem in root.findall(".//{http://www.egi.com/event_mff}event"):
                begin_time_elem = event_elem.find("{http://www.egi.com/event_mff}beginTime")
                code_elem = event_elem.find("{http://www.egi.com/event_mff}code")
                label_elem = event_elem.find("{http://www.egi.com/event_mff}label")

                if begin_time_elem is None or not begin_time_elem.text:
                    continue
                try:
                    event_time_str = begin_time_elem.text
                    if event_time_str.endswith("+0200"):
                        event_time_str = event_time_str.replace("+0200", "+02:00")
                    elif event_time_str.endswith("-0200"):
                        event_time_str = event_time_str.replace("-0200", "-02:00")
                    event_time = datetime.fromisoformat(event_time_str)

                    if recording_start_time:
                        time_diff = (event_time - recording_start_time).total_seconds()
                    else:
                        time_diff = event_time.timestamp()
                    time_diff = time_diff - pns_shift_seconds

                    code_str = None
                    if code_elem is not None and code_elem.text:
                        code_str = code_elem.text.strip()
                    if (not code_str or not code_str.upper().startswith("DI")) and label_elem is not None and label_elem.text:
                        lbl = label_elem.text.strip()
                        if lbl.upper().startswith("DI"):
                            code_str = lbl
                    if not code_str or not code_str.upper().startswith("DI"):
                        if label_elem is not None and label_elem.text:
                            events.append((time_diff, label_elem.text.strip()))
                        continue

                    num_match = re.search(r"\d+", code_str)
                    if num_match:
                        code_num = int(num_match.group())
                        if code_num in code_to_label:
                            event_label = code_to_label[code_num]
                        else:
                            event_label = label_elem.text.strip() if (label_elem is not None and label_elem.text) else code_str
                    else:
                        event_label = label_elem.text.strip() if (label_elem is not None and label_elem.text) else code_str
                    if event_label:
                        events.append((time_diff, event_label))
                except Exception as e:
                    print(f"Warning: Could not parse event: {e}")
        except Exception as e:
            print(f"Warning: Could not parse event file {event_file}: {e}")

    events.sort(key=lambda x: x[0])
    for event_time, event_label in events:
        closest_idx = (df["time_stamp"] - event_time).abs().idxmin()
        if df.loc[closest_idx, "event_label"]:
            df.loc[closest_idx, "event_label"] += f"; {event_label}"
        else:
            df.loc[closest_idx, "event_label"] = event_label
    if events:
        print(f"Loaded {len(events)} events from MFF file")


def _read_pns_mff(mff_path: str) -> pd.DataFrame:
    """Read PNS MFF from pnsSet.xml + signal1.bin; add time_stamp, event_label; fill events via _add_mff_events_to_df."""
    ch_names = _extract_pns_channel_names(mff_path)
    if not ch_names:
        raise ValueError(f"No channel names from pnsSet.xml in {mff_path}")

    signal_path = os.path.join(mff_path, "signal1.bin")
    if not os.path.exists(signal_path):
        raise ValueError(f"signal1.bin not found in {mff_path}")

    sampling_rate = 250.0
    log_files = [f for f in os.listdir(mff_path) if f.startswith("log_") and f.endswith(".txt")]
    if log_files:
        log_path = os.path.join(mff_path, log_files[0])
        try:
            with open(log_path, "r") as log_f:
                for line in log_f:
                    if "Sampling Rate:" in line or "sampling" in line.lower():
                        m = re.search(r"(\d+)\s*s/s", line, re.I) or re.search(r"(\d+)\s*Hz", line, re.I)
                        if m:
                            sampling_rate = float(m.group(1))
                            break
        except Exception:
            pass

    file_size = os.path.getsize(signal_path)
    with open(signal_path, "rb") as f:
        version = struct.unpack("<I", f.read(4))[0]
        header_size = struct.unpack("<I", f.read(4))[0]
        data_size = struct.unpack("<I", f.read(4))[0]
        num_channels_file = struct.unpack("<I", f.read(4))[0]
        sampling_rate_raw = struct.unpack("<f", f.read(4))[0]
        if header_size > 20:
            f.read(header_size - 20)

        remaining_bytes = file_size - header_size
        num_blocks = (remaining_bytes // data_size) if data_size > 0 else 1
        extra_bytes = remaining_bytes % data_size

        all_blocks_data = []
        for _ in range(num_blocks):
            all_blocks_data.append(np.frombuffer(f.read(data_size), dtype=np.float32))
        if extra_bytes > 0:
            all_blocks_data.append(np.frombuffer(f.read(extra_bytes), dtype=np.float32))

        data = np.concatenate(all_blocks_data)
    num_samples = len(data) // num_channels_file
    data = data[: num_samples * num_channels_file].reshape(num_samples, num_channels_file)

    if len(ch_names) != num_channels_file:
        ch_names = ch_names[:num_channels_file] if len(ch_names) >= num_channels_file else ch_names + [f"PNS_{i+1}" for i in range(len(ch_names), num_channels_file)]

    df = pd.DataFrame(data, columns=ch_names)
    df.insert(0, "time_stamp", np.arange(num_samples) / float(sampling_rate))
    df["event_label"] = ""
    _add_mff_events_to_df(df, mff_path)
    df = df.iloc[1:].reset_index(drop=True)
    return df


def edf_to_df(file_path: str) -> pd.DataFrame:
    """
    Supports Eyelink EDF, standard EDF, and MFF (EEG or PNS).
    - EyeLink EDF: edf2asc -> parse ASC -> DataFrame
    - MFF: try MNE EEG first; on failure use custom PNS reader
    - Standard EDF: mne
    """
    if _is_eyelink_edf(file_path):
        asc_path = _edf_to_asc(file_path)
        df = _asc_to_df(asc_path)

        print("Eyelink EDF loaded via edf2asc successfully")
        print(f"ASC: {asc_path}")
        print(f"Number of samples: {len(df)}")
        print(f"Samples with events: {(df['event_label'] != '').sum()}")
        samples_with_events = df[df["event_label"] != ""]
        if len(samples_with_events) > 0:
            print(samples_with_events.head(10))
        else:
            print("No events found in samples")
        return df

    # MFF (directory or .mff path): try EEG first, then PNS
    is_mff = os.path.isdir(file_path) or (isinstance(file_path, str) and file_path.lower().endswith(".mff"))
    if is_mff:
        try:
            raw = mne.io.read_raw_egi(
                file_path,
                preload=True,
                verbose=False,
                events_as_annotations=False,
                channel_naming="E%d",  # E1, E2, ... E256
            )
            df = raw.to_data_frame()
            if "time" in df.columns and "time_stamp" not in df.columns:
                df = df.rename(columns={"time": "time_stamp"})
            if "event_label" not in df.columns:
                df["event_label"] = ""
            _add_mff_events_to_df(df, file_path)
            df = df.iloc[1:].reset_index(drop=True)
            print("EEG MFF file loaded successfully")
            return df
        except Exception as e:
            print(f"MNE could not read as EEG MFF ({e}), trying PNS reader...")
            df = _read_pns_mff(file_path)
            return df

    # Standard EDF via MNE
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    df = raw.to_data_frame()
    df = df.iloc[1:].reset_index(drop=True)
    return df

def eeg_to_df(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".mff":
        # EGI MFF — channel_naming='E%d' gives E1, E2, ... E256
        raw = mne.io.read_raw_egi(
            file_path, preload=True, verbose=False, channel_naming="E%d"
        )
        df = raw.to_data_frame()  # includes 'time' column
        return df

    if ext == ".edf":
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        df = raw.to_data_frame()
        return df

    raise ValueError(f"Unsupported EEG format: {ext} (expected .edf or .mff)")



def main():
    df = edf_to_df(r"log_files/omer/10/threshold/test.edf")
    print(df.head())


if __name__ == "__main__":
    main()
