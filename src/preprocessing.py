import pandas as pd
import numpy as np
from pathlib import Path

# Columns that are identifiers or labels, NOT behavioural measurements.
ID_COLUMNS = ["SessionID"]
LABEL_COLUMNS = ["UserType", "BotType", "CoordinationGroupID"]
METADATA_COLUMNS = ["SessionSource", "BotID", "EventType", "Page", "Value", "Detail"]

# Numeric behavioural measurement columns.
BEHAVIOURAL_COLUMNS = [
    "Timestamp", "ElapsedMs", "IdleMs",
    "X", "Y", "DX", "DY", "Speed", "Acceleration",
    "ClickIntervalMs", "ClickCount",
    "ScrollDelta", "ScrollSpeed", "ScrollCount",
    "InterKeyIntervalMs", "TypingCount",
    "NavigationCount", "SessionDurationMs",
    "TotalMouseSamples", "TotalClicks", "TotalScrolls",
    "TotalTypingEvents", "TotalNavigations",
]

# Columns that should be present for specific event types (structural presence).
MOUSE_COLUMNS = ["X", "Y", "DX", "DY", "Speed", "Acceleration"]
KEYBOARD_COLUMNS = ["InterKeyIntervalMs", "TypingCount", "KeyCategory"]
CLICK_COLUMNS = ["ClickIntervalMs", "ClickCount"]
SCROLL_COLUMNS = ["ScrollDelta", "ScrollSpeed", "ScrollCount"]

# Non-negative measurement columns (negative = physically impossible).
NON_NEGATIVE_COLUMNS = ["ElapsedMs", "IdleMs", "Speed", "Acceleration",
                        "ClickCount", "ScrollCount", "TypingCount",
                        "NavigationCount", "SessionDurationMs",
                        "TotalMouseSamples", "TotalClicks", "TotalScrolls",
                        "TotalTypingEvents", "TotalNavigations"]


def load_data(path):
    """Load the original Excel dataset read-only."""
    path = Path(path)
    df = pd.read_excel(path, sheet_name="Events")
    return df


def split_labels_metadata_behaviour(df):
    """Split a dataframe into ids, labels, metadata, and behavioural measurements.

    Returns a dict of dataframes so labels/metadata are never mixed into
    behavioural feature computation.
    """
    df = df.copy()
    ids = df[ID_COLUMNS] if all(c in df.columns for c in ID_COLUMNS) else pd.DataFrame()
    labels = df[[c for c in LABEL_COLUMNS if c in df.columns]]
    metadata = df[[c for c in METADATA_COLUMNS if c in df.columns]]
    behavioural = df[[c for c in BEHAVIOURAL_COLUMNS if c in df.columns]]
    return {"ids": ids, "labels": labels, "metadata": metadata,
            "behavioural": behavioural}


def add_datetime(df, ts_col="Timestamp", out_col="Timestamp_dt"):
    """Convert epoch-millisecond timestamp to datetime (read-only, returns copy)."""
    df = df.copy()
    df[out_col] = pd.to_datetime(df[ts_col], unit="ms", errors="coerce")
    return df


def sort_chronologically(df, id_col="SessionID", ts_col="Timestamp"):
    """Sort events within each session by timestamp."""
    df = df.copy()
    df = df.sort_values([id_col, ts_col], kind="mergesort").reset_index(drop=True)
    return df


def validate_numeric_columns(df, columns=None):
    """Coerce behavioural columns to numeric, reporting non-convertible values."""
    if columns is None:
        columns = BEHAVIOURAL_COLUMNS
    df = df.copy()
    report = {}
    for col in columns:
        if col not in df.columns:
            continue
        before_null = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_null = df[col].isna().sum()
        newly_null = int(after_null - before_null)
        report[col] = {"newly_null_from_conversion": newly_null}
    return df, report


def find_invalid_measurements(df):
    """Identify rows with physically impossible measurements."""
    issues = []
    for col in NON_NEGATIVE_COLUMNS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        neg_mask = s < 0
        if neg_mask.any():
            issues.append({
                "column": col,
                "n_negative": int(neg_mask.sum()),
                "session_ids": sorted(df.loc[neg_mask, "SessionID"].unique().tolist())
                    if "SessionID" in df.columns else [],
            })
    return issues


def handle_missing_by_event_type(df, event_col="EventType"):
    """Flag structural vs unexpected missing values per event type.

    Structural nulls (expected for a given event type) are left as NaN and NOT
    replaced with zero. Unexpected nulls (e.g., missing X/Y on a mouse event)
    are flagged for review.
    """
    df = df.copy()
    flags = []

    def norm_et(v):
        if pd.isna(v):
            return ""
        return str(v).lower()

    et = df[event_col].apply(norm_et) if event_col in df.columns else pd.Series([""] * len(df))

    def has_any_null(row, cols):
        return row[cols].isna().any()

    mouse = et.str.contains("mouse", na=False)
    keyboard = et.str.contains("key", na=False) | et.str.contains("type", na=False)
    click = et.str.contains("click", na=False)
    scroll = et.str.contains("scroll", na=False)

    # Mouse events should have spatial data; null X/Y on mouse = unexpected.
    if "X" in df.columns and "Y" in df.columns:
        bad_mouse = mouse & (df["X"].isna() | df["Y"].isna())
        if bad_mouse.any():
            flags.append({
                "issue": "mouse_event_null_coordinates",
                "n_records": int(bad_mouse.sum()),
                "session_ids": sorted(df.loc[bad_mouse, "SessionID"].unique().tolist())
                    if "SessionID" in df.columns else [],
                "handling": "flagged for review; excluded from spatial features",
            })

    return flags


def preprocess(df, exclude_invalid=True):
    """Run the full reproducible preprocessing pipeline.

    Returns a dict with cleaned dataframes, excluded records, and a report.
    Original dataframe is never modified.
    """
    df = df.copy()

    # 1. Convert timestamp to datetime
    df = add_datetime(df, "Timestamp", "Timestamp_dt")

    # 2. Sort chronologically within session
    df = sort_chronologically(df, "SessionID", "Timestamp")

    # 3. Validate numeric behavioural columns
    df, num_report = validate_numeric_columns(df, BEHAVIOURAL_COLUMNS)

    # 4. Identify invalid measurements
    invalid_issues = find_invalid_measurements(df)

    # 5. Handle missing per event type
    missing_flags = handle_missing_by_event_type(df)

    # Build exclusion mask
    exclude_mask = pd.Series(False, index=df.index)
    excluded_reasons = []

    if "Timestamp_dt" in df.columns:
        bad_ts = df["Timestamp_dt"].isna()
        if bad_ts.any():
            exclude_mask |= bad_ts
            excluded_reasons.append(("invalid_timestamp", int(bad_ts.sum())))

    # Negative speed: physically impossible
    if "Speed" in df.columns:
        speed = pd.to_numeric(df["Speed"], errors="coerce")
        neg_speed = speed < 0
        if neg_speed.any():
            exclude_mask |= neg_speed
            excluded_reasons.append(("negative_speed", int(neg_speed.sum())))

    # Mouse events with null coordinates
    mouse = df["EventType"].astype(str).str.contains("mouse", case=False, na=False) \
        if "EventType" in df.columns else pd.Series(False, index=df.index)
    if "X" in df.columns and "Y" in df.columns:
        null_xy_mouse = mouse & (df["X"].isna() | df["Y"].isna())
        if null_xy_mouse.any():
            exclude_mask |= null_xy_mouse
            excluded_reasons.append(("mouse_null_coordinates", int(null_xy_mouse.sum())))

    excluded_df = df[exclude_mask].copy() if exclude_mask.any() else df.iloc[0:0].copy()
    cleaned_df = df[~exclude_mask].copy() if exclude_mask.any() else df.copy()

    # Add exclusion reason column to excluded records
    if len(excluded_df) > 0:
        reason_list = []
        for idx in excluded_df.index:
            reasons = []
            for reason_name, _ in excluded_reasons:
                reasons.append(reason_name)
            reason_list.append("; ".join(reasons))
        excluded_df["exclusion_reason"] = reason_list

    report = {
        "n_original": int(len(df)),
        "n_excluded": int(len(excluded_df)),
        "n_cleaned": int(len(cleaned_df)),
        "excluded_reasons": excluded_reasons,
        "numeric_validation": num_report,
        "invalid_measurements": invalid_issues,
        "missing_flags": missing_flags,
    }

    return {
        "cleaned": cleaned_df,
        "excluded": excluded_df,
        "report": report,
    }


def save_splits(cleaned_df, output_dir):
    """Save cleaned dataset split into full, behavioural, and labels/metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full cleaned dataset
    cleaned_df.to_csv(output_dir / "cleaned_events.csv", index=False)

    parts = split_labels_metadata_behaviour(cleaned_df)
    parts["ids"].to_csv(output_dir / "cleaned_events_ids.csv", index=False)
    parts["labels"].to_csv(output_dir / "cleaned_events_labels.csv", index=False)
    parts["metadata"].to_csv(output_dir / "cleaned_events_metadata.csv", index=False)
    parts["behavioural"].to_csv(output_dir / "cleaned_events_behavioural.csv", index=False)

    return parts
