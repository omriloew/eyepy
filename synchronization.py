
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

EVENT_COL = "event_label"
TIME_COL  = "time_stamp"
FRAME_COL = "frame_index"  # קיים רק ב-IR (אם אין, אל חובה)

class SyncError(Exception):
    pass

def _get_numeric_data_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    exclude_set = set(exclude)
    return [c for c in df.columns if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])]

def _extract_event_series(df: pd.DataFrame) -> pd.DataFrame:
    """מחזיר רק אירועים (label+time), ממויין בזמן, בלי NaN בלייבל."""
    ev = df[[TIME_COL, EVENT_COL]].dropna(subset=[EVENT_COL]).copy()
    ev = ev.sort_values(TIME_COL).reset_index(drop=True)
    return ev

def _align_events_to_E(device_events: pd.DataFrame, E_events: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    מייצר יישור לפי הסדר הכרונולוגי הגלובלי של E:
    עוברים על אירועי E לפי זמן; לכל label סופרים את ההופעה ה-k;
    מהמכשיר לוקחים את ההופעה ה-k של אותו label (אם קיימת) ושומרים זוג זמנים.
    מאפשר חזרות כמו A B A D בלי לשבור מונוטוניות גלובלית.
    """
    if device_events.empty:
        raise SyncError("אין אירועים בטבלה — נדרש לפחות שני עוגנים.")

    # מיון מראש
    dev = device_events.sort_values(TIME_COL).reset_index(drop=True)
    Eev = E_events.sort_values(TIME_COL).reset_index(drop=True)

    # אינדקסים של ההופעה ה-k לכל label בכל צד
    # מכשיר: רשימת זמנים לכל label
    dev_by_label = {}
    for lab, g in dev.groupby(EVENT_COL):
        dev_by_label[lab] = g[TIME_COL].to_list()

    # מונה הופעות שכבר נצרכו מכל label
    counters = {lab: 0 for lab in dev_by_label.keys()}

    t_dev_list = []
    t_E_list   = []

    # סריקה כרונולוגית של E
    for _, row in Eev.iterrows():
        lab = row[EVENT_COL]
        tE  = float(row[TIME_COL])

        # אם label זה לא קיים במכשיר — מדלגים (המכשיר לא רשם את האירוע הזה)
        if lab not in dev_by_label:
            continue

        k = counters[lab]
        # אם למכשיר יש עוד הופעה עבור label זה — נצרף; אחרת דילול (המכשיר סיים את כל ההופעות שלו עבור label זה)
        if k < len(dev_by_label[lab]):
            tD = float(dev_by_label[lab][k])
            t_dev_list.append(tD)
            t_E_list.append(tE)
            counters[lab] += 1
        # אם לא — ממשיכים הלאה (אין שגיאה: המכשיר יכול להכיל תת־קבוצה של E)

    if len(t_dev_list) < 2:
        raise SyncError("נדרשים לפחות שני עוגנים משותפים (במצטבר) ליצירת התאמה.")

    t_dev = np.array(t_dev_list, dtype=float)
    t_E   = np.array(t_E_list,   dtype=float)

    # בדיקת מונוטוניות גלובלית אחרי היישור (אמורה להחזיק מעצם הסריקה לפי E)
    if not (np.all(np.diff(t_dev) >= 0) and np.all(np.diff(t_E) >= 0)):
        raise SyncError("זוהתה חריגה במונוטוניות העוגנים לאחר יישור כרונולוגי.")

    return t_dev, t_E


def _fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    מתאים y ≈ a*x + b. אם יש בדיוק 2 נקודות — פתרון סגור; אחרת LS.
    """
    if x.size == 2:
        dx = x[1] - x[0]
        if dx == 0:
            raise SyncError("עוגנים זהים בזמן המכשיר — אי אפשר להתאים אפינית.")
        a = (y[1] - y[0]) / dx
        b = y[0] - a * x[0]
        return float(a), float(b)
    # LS
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def _fit_device_to_E(df_device: pd.DataFrame, df_E: pd.DataFrame) -> Tuple[float, float]:
    dev_ev = _extract_event_series(df_device)
    E_ev   = _extract_event_series(df_E)
    x, y   = _align_events_to_E(dev_ev, E_ev)
    a, b   = _fit_affine(x, y)
    # אימות: שמירה על סדר אחרי המיפוי
    mapped = a * x + b
    if not np.all(np.diff(mapped) > 0):
        raise SyncError("התאמה אפינית גורמת לפגיעה במונוטוניות (a<=0?).")
    return a, b

def _compose_device_to_IR(ab_devE: Tuple[float,float], ab_irE: Tuple[float,float]) -> Tuple[float,float]:
    """
    אם t_E = a_dev * t_dev + b_dev  וגם  t_E = a_ir * t_ir + b_ir
    אז t_ir = (t_E - b_ir) / a_ir = (a_dev/a_ir) * t_dev + (b_dev - b_ir)/a_ir
    """
    a_dev, b_dev = ab_devE
    a_ir,  b_ir  = ab_irE
    if a_ir == 0:
        raise SyncError("a_ir==0 בלתי אפשרי.")
    a = a_dev / a_ir
    b = (b_dev - b_ir) / a_ir
    if a <= 0:
        raise SyncError("התוצאה גורמת לשיפוע לא חיובי (a<=0) במיפוי למרחב IR.")
    return float(a), float(b)

def _windowed_average_at_times(device_times_ir: np.ndarray,
                               device_vals: pd.DataFrame,
                               query_times_ir: np.ndarray,
                               window_ms: float) -> pd.DataFrame:
    """
    ממוצע בכל חלון [t - w, t + w] סביב כל זמן IR.
    מימוש וקטורי: מצטברים + שני merge_asof כדי להביא סכום/ספירה בקצוות.
    """
    # לבטיחות: מיון עולה
    order = np.argsort(device_times_ir)
    dt = device_times_ir[order]
    vals = device_vals.iloc[order].copy()

    # נכין מצטברים לעמודות נומריות
    num_cols = vals.columns.tolist()
    # מחליפים אינפים/NaN כראוי לספירה
    mask = ~vals[num_cols].isna()

    csum = vals[num_cols].fillna(0).cumsum()
    ccnt = mask.astype(np.int64).cumsum()

    # יוצרים DF עזר עם זמן+מצטברים
    helper = pd.DataFrame({TIME_COL: dt})
    for c in num_cols:
        helper[f"sum__{c}"] = csum[c].to_numpy()
        helper[f"cnt__{c}"] = ccnt[c].to_numpy()

    # קצוות החלון לכל t_q
    left_q  = pd.DataFrame({TIME_COL: query_times_ir - window_ms})
    right_q = pd.DataFrame({TIME_COL: query_times_ir + window_ms})

    # asof: לוקחים את השורה האחרונה שהזמן שלה <= גבול
    left  = pd.merge_asof(left_q,  helper, on=TIME_COL, direction="backward")
    right = pd.merge_asof(right_q, helper, on=TIME_COL, direction="backward")

    out = {}
    for c in num_cols:
        s_right = right[f"sum__{c}"]
        s_left  = left[f"sum__{c}"].fillna(0)
        n_right = right[f"cnt__{c}"]
        n_left  = left[f"cnt__{c}"].fillna(0)

        num = s_right - s_left
        den = n_right - n_left
        with np.errstate(invalid='ignore', divide='ignore'):
            avg = num / den
        avg[den == 0] = np.nan
        out[c] = avg.to_numpy()

    return pd.DataFrame(out)

def synchronize_to_IR(
    df_IR: pd.DataFrame,
    devices: Dict[str, pd.DataFrame],  # {"VC": df_vc, "EEG": df_eeg, "MD": df_md, ... (אפשר לכלול גם "IR": df_IR אך אין צורך)}
    df_E: pd.DataFrame,
    window_ms: float = 5.0
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float,float]]]:
    """
    מחזיר:
      - DataFrame מסונכרן לציר ה-IR (שומר את עמודות IR המקוריות + עמודות ממוצעות מכל מכשיר עם קידומת)
      - dict עם פרמטרי המיפוי למעקב: {"IR->E": (a,b), "VC->E": (a,b), ..., "VC->IR": (a,b), ...}
    """
    # בדיקות בסיס
    for name, df in [("IR", df_IR), ("E", df_E), *devices.items()]:
        if TIME_COL not in df.columns or EVENT_COL not in df.columns:
            raise SyncError(f"{name}: חסרות עמודות חובה '{TIME_COL}' או '{EVENT_COL}'.")

    # התאמות אפיניות ל-E
    a_IR_E, b_IR_E = _fit_device_to_E(df_IR, df_E)
    mappings = {"IR->E": (a_IR_E, b_IR_E)}

    device_to_IR_map: Dict[str, Tuple[float,float]] = {}

    # ציר הייחוס של IR
    ir_times = df_IR[TIME_COL].to_numpy(dtype=float)

    # נתחיל את פלט עם עותק של IR (לא פוגעים במקור)
    out = df_IR.copy()

    # עבור כל מכשיר: התאמת dev->E, הרכבת dev->IR, ממוצע חלון והוספה ל-out
    for dev_name, df_dev in devices.items():
        a_dev_E, b_dev_E = _fit_device_to_E(df_dev, df_E)
        mappings[f"{dev_name}->E"] = (a_dev_E, b_dev_E)

        a_dev_IR, b_dev_IR = _compose_device_to_IR((a_dev_E, b_dev_E), (a_IR_E, b_IR_E))
        device_to_IR_map[dev_name] = (a_dev_IR, b_dev_IR)
        mappings[f"{dev_name}->IR"] = (a_dev_IR, b_dev_IR)

        # ממפים את כל הזמנים של המכשיר לציר IR
        t_dev = df_dev[TIME_COL].to_numpy(dtype=float)
        t_ir_mapped = a_dev_IR * t_dev + b_dev_IR

        # עמודות דאטה נומריות
        cols = _get_numeric_data_cols(df_dev, exclude=[TIME_COL, EVENT_COL])
        if not cols:
            # אין עמודות דאטה — מדלגים
            continue

        dev_vals = df_dev[cols]
        # ממוצע חלון יעיל סביב כל זמן IR
        avg_df = _windowed_average_at_times(t_ir_mapped, dev_vals, ir_times, window_ms)

        # קידומת שם המכשיר
        avg_df.columns = [f"{dev_name}__{c}" for c in avg_df.columns]

        # מיזוג לפי אינדקס (אותו סדר כמו df_IR)
        out = pd.concat([out.reset_index(drop=True), avg_df.reset_index(drop=True)], axis=1)

    return out, mappings

df_E   = pd.read_csv("rawResultsExample/E.csv")
df_IR  = pd.read_csv("rawResultsExample/IR.csv")
df_VC  = pd.read_csv("rawResultsExample/VC.csv")
df_EL = pd.read_csv("rawResultsExample/EL.csv")
df_MD  = pd.read_csv("rawResultsExample/MD.csv")
devices = {
    "VC": df_VC,
    "EL": df_EL,
    "MD": df_MD,
    # אפשר להוסיף טבלאות נוספות באותה צורה
}

synced_df, maps = synchronize_to_IR(
    df_IR=df_IR,
    devices=devices,
    df_E=df_E,
    window_ms=5.0  # אפשר לשנות: 2.0/10.0 לפי רזולוציה ונויז
)

# לשמור לקובץ:
synced_df.to_csv("synced_to_IR.csv", index=False)
print("Mappings:", maps)