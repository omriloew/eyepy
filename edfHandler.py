import os
import re
import subprocess
import pandas as pd
import numpy as np
import mne
from mffpy.reader import Reader

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


def edf_to_df(file_path: str) -> pd.DataFrame:
    """
    Supports both Eyelink EDF files and standard EDF files.
    - EyeLink EDF: edf2asc -> parse ASC -> DataFrame
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

    # Standard EDF via MNE
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    df = raw.to_data_frame()
    df = df.iloc[1:].reset_index(drop=True)
    return df

def eeg_to_df(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".mff":
        # EGI MFF
        raw = mne.io.read_raw_egi(file_path, preload=True, verbose=False)
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
