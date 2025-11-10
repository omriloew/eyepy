
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

EVENT_COL = "event_label"
TIME_COL  = "time_stamp"
FRAME_COL = "frame_index"

class SyncError(Exception):
    pass

def _get_numeric_data_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Get numeric columns, excluding specified columns and discrete columns (starting with _)."""
    exclude_set = set(exclude)
    return [c for c in df.columns if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c]) and not c.startswith('_')]

def _get_discrete_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Get discrete columns that start with _ and should not use window pooling."""
    exclude_set = set(exclude)
    return [c for c in df.columns if c not in exclude_set and c.startswith('_')]


def _fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y = a*x + b using least squares."""
    if x.size == 2:
        dx = x[1] - x[0]
        if dx == 0:
            raise SyncError("Identical device times - cannot fit affine transformation.")
        a = (y[1] - y[0]) / dx
        b = y[0] - a * x[0]
        return float(a), float(b)
    
    # Least squares
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def _extract_event_series(df: pd.DataFrame) -> List[Tuple[float, str]]:
    events = []
    for _, row in df.iterrows():
        event_label = str(row[EVENT_COL]).strip()
        if event_label and not pd.isna(event_label) and event_label != '' and event_label != 'nan':
            events.append((float(row[TIME_COL]), event_label))
    events = sorted(events, key=lambda x: x[0])
    print("--------------------------------")
    print(events)
    return events

def calculate_affine_transformations(df_device: pd.DataFrame, E_events: List[Tuple[float, str]]) -> Tuple[float, float]:
    """
    Calculate affine transformation from device time to E time.
    """
    device_events = _extract_event_series(df_device)
    if len(device_events) < 2:
        raise SyncError(f"Device has less than 2 events.")
    x = [device_event[0] for device_event in device_events]
    y = []
    curr_E_idx = 0
    for device_event in device_events:
        while device_event[1] != E_events[curr_E_idx][1]:
            curr_E_idx += 1
            if curr_E_idx >= len(E_events):
                raise SyncError(f"Device event {device_event[1]} not found in E events.")
        y.append(E_events[curr_E_idx][0])
        curr_E_idx += 1
    a, b = _fit_affine(np.array(x), np.array(y))
    return a, b

def get_affine_mapping(reference_name: str, devices: Dict[str, pd.DataFrame], df_E: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Synchronize multiple device dataframes to a reference timeline.
    """
    E_events = _extract_event_series(df_E)
    affine_transformations = {}
    for device_name, device_df in devices.items():
        a, b = calculate_affine_transformations(device_df, E_events)
        affine_transformations[device_name] = (a, b)

    a_ref_to_E,b_ref_to_E = affine_transformations[reference_name]
    for device_name, device_df in devices.items():
        if device_name == reference_name:
            a_E_to_ref = 1/a_ref_to_E
            b_E_to_ref = -b_ref_to_E / a_ref_to_E
            affine_transformations[device_name] = (a_E_to_ref, b_E_to_ref)
            continue
        a_dev_to_E, b_dev_to_E = affine_transformations[device_name]
        a_dev_to_ref = a_dev_to_E / a_ref_to_E
        b_dev_to_ref = (b_dev_to_E - b_ref_to_E) / a_ref_to_E
        affine_transformations[device_name] = (a_dev_to_ref, b_dev_to_ref)
    return affine_transformations

def _extract_vals(df: pd.DataFrame, col: str):
    vals = df[col].dropna().to_numpy()
    times = df[TIME_COL][df[col].notna()].to_numpy(dtype=float)
    return zip(times, vals)

def _pool_discrete_col(reference_df: pd.DataFrame, device_df: pd.DataFrame, col: str, window_ms: float = 5.0) -> pd.DataFrame:
    device_vals_and_times = _extract_vals(device_df, col)
    reference_times = reference_df[TIME_COL].to_numpy(dtype=float)  
    window_sec = window_ms / 1000.0
    for time, val in device_vals_and_times:
        nearest_reference_time = np.argmin(np.abs(reference_times - time))
        if np.abs(reference_times[nearest_reference_time] - time) <= window_sec:
            reference_df.loc[nearest_reference_time, col] = val
        else:
            reference_df.loc[nearest_reference_time, col] = np.nan
    return reference_df

def _pool_numeric_col(reference_df: pd.DataFrame, device_df: pd.DataFrame, col: str, window_ms: float = 5.0) -> pd.DataFrame:
    device_times = device_df[TIME_COL].to_numpy(dtype=float)
    reference_times = reference_df[TIME_COL].to_numpy(dtype=float)
    window_sec = window_ms / 1000.0
    for reference_time in reference_times:
        times_in_window = device_times[np.abs(device_times - reference_time) <= window_sec]
        vals_in_window = device_df.loc[device_df[TIME_COL].isin(times_in_window), col].to_numpy(dtype=float)
        if len(vals_in_window) > 0:
            reference_df.loc[reference_df[TIME_COL] == reference_time, col] = np.mean(vals_in_window)
    return reference_df

def _pool_events(reference_df: pd.DataFrame, device_df: pd.DataFrame) -> pd.DataFrame:
    device_events = np.array(_extract_event_series(device_df))
    reference_events = np.array(_extract_event_series(reference_df))
    device_times = device_events[:, 0].astype(float)
    reference_times = reference_events[:, 0].astype(float)
    for device_time, device_event in zip(device_times, device_events[:, 1]):
        nearest_reference_event_time = np.argmin(np.abs(reference_times - device_time))
        nearest_reference_event = reference_events[nearest_reference_event_time, 1]
        reference_df.loc[nearest_reference_event, EVENT_COL] = f"{nearest_reference_event}; {device_event}"
    return reference_df

def _pool_into_reference_df(reference_df: pd.DataFrame, device_df: pd.DataFrame, window_ms: float = 5.0) -> pd.DataFrame:
    discrete_cols = _get_discrete_cols(device_df, exclude=[TIME_COL, EVENT_COL])
    numeric_cols = _get_numeric_data_cols(device_df, exclude=[TIME_COL, EVENT_COL])
    for col in discrete_cols:
        _pool_discrete_col(reference_df, device_df, col, window_ms)
    for col in numeric_cols:
        _pool_numeric_col(reference_df, device_df, col, window_ms)
    _pool_discrete_col(reference_df, device_df, col="event_label", window_ms=window_ms)

    return reference_df

def synchronize_to_reference(reference_name: str, devices: Dict[str, pd.DataFrame], df_E: pd.DataFrame, window_ms: float = 5.0) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    affine_transformations = get_affine_mapping(reference_name=reference_name, devices=devices, df_E=df_E)
    
    for device_name, device_df in devices.items():
        if device_name == reference_name:
            continue
        a_dev_to_ref, b_dev_to_ref = affine_transformations[device_name]
        device_df[TIME_COL] = a_dev_to_ref * device_df[TIME_COL] + b_dev_to_ref
        _extract_event_series(device_df)

    out = devices[reference_name].copy()

    a_E_to_ref, b_E_to_ref = affine_transformations[reference_name]
    df_E[TIME_COL] = a_E_to_ref * df_E[TIME_COL] + b_E_to_ref

    for device_name, device_df in devices.items():
        if device_name == reference_name:
            continue
        _pool_into_reference_df(out, device_df, window_ms)

    _pool_into_reference_df(out, df_E, window_ms)
    _pool_discrete_col(out, df_E, col="event_label", window_ms=window_ms)

    for device_name, device_df in devices.items():
        device_df.to_csv(f"synced_{device_name}.csv", index=False)
    df_E.to_csv("synced_E.csv", index=False)
    out.to_csv("synced_reference.csv", index=False)
    return out, affine_transformations


def main():
    df_E = pd.read_csv("rawResultsExample/E.csv")
    df_SWIR = pd.read_csv("rawResultsExample/IR.csv")
    df_VC = pd.read_csv("rawResultsExample/VC.csv")
    df_EL = pd.read_csv("rawResultsExample/EL.csv")
    df_MD = pd.read_csv("rawResultsExample/MD.csv")
    devices = {
        "VC": df_VC,
        "EL": df_EL,
        "MD": df_MD,
        "SWIR": df_SWIR,
    }
    
    synced_df, mappings = synchronize_to_reference(
        reference_name="SWIR",
        devices=devices,
        df_E=df_E,
        window_ms=5.0
    )
    
    synced_df.to_csv("synced_to_reference.csv", index=False)
    print("Mappings:", mappings)

if __name__ == "__main__":
    main()
