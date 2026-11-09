"""Session-level behavioural feature extraction for BotPrint.

Transforms cleaned event-level data into one row per session.
Identifiers/labels (SessionID, UserType, BotType, etc.) are preserved
separately; only behavioural measurements are aggregated into features.

Rules:
- Group by SessionID.
- Sort events chronologically (Timestamp) before sequential features.
- Use valid (non-null) observations only for each feature.
- Do not replace unavailable features with zero unless zero is the
  actual measured value (e.g., 0 clicks when click events exist).
- Return NaN for features that cannot be computed (insufficient data).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Columns to preserve as session-level labels/metadata (not features)
LABEL_COLS = ["UserType", "BotType"]
OPTIONAL_LABEL_COLS = ["CoordinationGroupID", "SessionSource", "BotID"]
ID_COL = "SessionID"

# Event type matchers (lowercased)
MOUSE_TYPES = {"mousemove", "mouse_move", "mouse"}
KEY_TYPES = {"keypress", "keydown", "keyup", "key"}
CLICK_TYPES = {"click", "mouse_click", "mousedown", "mouseup"}
SCROLL_TYPES = {"scroll", "wheel", "mousewheel"}
NAV_TYPES = {"navigation", "navigate", "pageview", "page_view"}


def _et(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def _is_type(series: pd.Series, types: set) -> pd.Series:
    return _et(series).isin(types)


def _safe_mean(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float(v.mean()) if len(v) else np.nan


def _safe_std(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float(v.std(ddof=1)) if len(v) >= 2 else (0.0 if len(v) == 1 else np.nan)


def _safe_sum(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float(v.sum()) if len(v) else np.nan


def _safe_count_valid(x: pd.Series) -> int:
    return int(pd.to_numeric(x, errors="coerce").notna().sum())


def _movement_distance(g: pd.DataFrame) -> float:
    """Total movement distance from consecutive mouse positions (Euclidean)."""
    m = g[_is_type(g["EventType"], MOUSE_TYPES)].copy()
    if "X" not in m.columns or "Y" not in m.columns:
        return np.nan
    m = m.dropna(subset=["X", "Y"]).sort_values("Timestamp")
    if len(m) < 2:
        return 0.0 if len(m) == 1 else np.nan
    dx = m["X"].diff()
    dy = m["Y"].diff()
    dist = np.sqrt(dx**2 + dy**2).dropna()
    return float(dist.sum()) if len(dist) else np.nan


def _direction_changes(g: pd.DataFrame) -> float:
    """Count direction changes in mouse movement using DX/DY sign changes.

    A direction change is counted when the sign of DX or DY changes
    between consecutive mousemove events (ignoring zero-only transitions
    where both are zero). Requires >= 2 valid movement vectors.
    """
    m = g[_is_type(g["EventType"], MOUSE_TYPES)].copy()
    if "DX" not in m.columns or "DY" not in m.columns:
        return np.nan
    m = m.dropna(subset=["DX", "DY"], how="all").sort_values("Timestamp")
    if len(m) < 2:
        return np.nan
    # Use angle of movement vector; count significant direction changes
    dx = pd.to_numeric(m["DX"], errors="coerce").fillna(0.0)
    dy = pd.to_numeric(m["DY"], errors="coerce").fillna(0.0)
    angles = np.arctan2(dy.values, dx.values)
    # Ignore steps with no movement
    moving = (np.abs(dx.values) + np.abs(dy.values)) > 0
    angles = angles[moving]
    if len(angles) < 2:
        return 0.0 if len(angles) == 1 else np.nan
    # Direction change: absolute angular difference > threshold (30 degrees)
    diffs = np.abs(np.diff(angles))
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    changes = int((diffs > (np.pi / 6)).sum())
    return float(changes)


def _session_duration_s(g: pd.DataFrame) -> float:
    """Session duration in seconds from SessionDurationMs or ElapsedMs."""
    for col in ["SessionDurationMs", "ElapsedMs"]:
        if col in g.columns:
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(v):
                return float(v.max()) / 1000.0
    # Fallback: timestamp range
    if "Timestamp" in g.columns:
        t = pd.to_numeric(g["Timestamp"], errors="coerce").dropna()
        if len(t) >= 2:
            return float(t.max() - t.min()) / 1000.0
    return np.nan


def _session_total_events(g: pd.DataFrame) -> int:
    return int(len(g))


def _navigation_count(g: pd.DataFrame) -> float:
    # Prefer explicit navigation events; fallback to NavigationCount field
    nav_events = _is_type(g["EventType"], NAV_TYPES).sum()
    if nav_events > 0:
        return float(nav_events)
    if "NavigationCount" in g.columns:
        v = pd.to_numeric(g["NavigationCount"], errors="coerce").dropna()
        if len(v):
            return float(v.max())
    # No navigation events observed -> 0 is the actual measured count
    return 0.0


def extract_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract session-level behavioural features from cleaned event data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned event-level dataset (one row per event).

    Returns
    -------
    pd.DataFrame
        One row per SessionID with labels and behavioural features.
    """
    required = {ID_COL, "EventType", "Timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data = data.sort_values([ID_COL, "Timestamp"], kind="mergesort").reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for session_id, g in data.groupby(ID_COL, sort=False):
        row: Dict[str, Any] = {ID_COL: session_id}

        # Preserve labels (one per session; take first non-null)
        for col in LABEL_COLS + OPTIONAL_LABEL_COLS:
            if col in g.columns:
                vals = g[col].dropna()
                row[col] = vals.iloc[0] if len(vals) else np.nan

        # ---- Session behaviour ----
        duration_s = _session_duration_s(g)
        total_events = _session_total_events(g)
        row["session_duration_s"] = duration_s
        row["session_total_events"] = total_events
        row["session_event_frequency"] = (
            float(total_events) / duration_s
            if pd.notna(duration_s) and duration_s > 0 else np.nan
        )
        row["session_mean_idle_ms"] = _safe_mean(g["IdleMs"]) if "IdleMs" in g.columns else np.nan
        row["session_navigation_count"] = _navigation_count(g)

        # ---- Mouse behaviour ----
        mouse = g[_is_type(g["EventType"], MOUSE_TYPES)]
        n_mouse = int(len(mouse))
        row["mouse_event_count"] = n_mouse
        row["mouse_has_data"] = int(n_mouse > 0)
        row["mouse_mean_speed"] = _safe_mean(mouse["Speed"]) if "Speed" in mouse.columns else np.nan
        row["mouse_std_speed"] = _safe_std(mouse["Speed"]) if "Speed" in mouse.columns else np.nan
        row["mouse_mean_acceleration"] = _safe_mean(mouse["Acceleration"]) if "Acceleration" in mouse.columns else np.nan
        row["mouse_std_acceleration"] = _safe_std(mouse["Acceleration"]) if "Acceleration" in mouse.columns else np.nan
        row["mouse_total_distance"] = _movement_distance(g)
        row["mouse_movement_frequency"] = (
            float(n_mouse) / duration_s
            if pd.notna(duration_s) and duration_s > 0 else np.nan
        ) if n_mouse > 0 else (0.0 if n_mouse == 0 else np.nan)
        # movement frequency: 0 is valid when there are no mouse events but session exists?
        # No - if no mouse events, frequency of mouse movement is not meaningfully 0 as a measured rate
        # in the same way; better to report NaN and rely on has_data indicator.
        row["mouse_movement_frequency"] = (
            float(n_mouse) / duration_s
            if n_mouse > 0 and pd.notna(duration_s) and duration_s > 0 else np.nan
        )
        row["mouse_direction_changes"] = _direction_changes(g)

        # ---- Keyboard behaviour ----
        key = g[_is_type(g["EventType"], KEY_TYPES)]
        n_key = int(len(key))
        row["keyboard_event_count"] = n_key
        row["keyboard_has_data"] = int(n_key > 0)
        row["keyboard_mean_interkey_ms"] = _safe_mean(key["InterKeyIntervalMs"]) if "InterKeyIntervalMs" in key.columns else np.nan
        row["keyboard_std_interkey_ms"] = _safe_std(key["InterKeyIntervalMs"]) if "InterKeyIntervalMs" in key.columns else np.nan
        row["keyboard_typing_frequency"] = (
            float(n_key) / duration_s
            if n_key > 0 and pd.notna(duration_s) and duration_s > 0 else np.nan
        )
        row["keyboard_typing_count"] = float(n_key)  # 0 is the actual measured count

        # ---- Click behaviour ----
        click = g[_is_type(g["EventType"], CLICK_TYPES)]
        n_click = int(len(click))
        row["click_event_count"] = n_click
        row["click_has_data"] = int(n_click > 0)
        row["click_mean_interval_ms"] = _safe_mean(click["ClickIntervalMs"]) if "ClickIntervalMs" in click.columns else np.nan
        row["click_std_interval_ms"] = _safe_std(click["ClickIntervalMs"]) if "ClickIntervalMs" in click.columns else np.nan
        row["click_frequency"] = (
            float(n_click) / duration_s
            if n_click > 0 and pd.notna(duration_s) and duration_s > 0 else np.nan
        )
        row["click_count"] = float(n_click)  # 0 is the actual measured count

        # ---- Scroll behaviour ----
        scroll = g[_is_type(g["EventType"], SCROLL_TYPES)]
        n_scroll = int(len(scroll))
        row["scroll_event_count"] = n_scroll
        row["scroll_has_data"] = int(n_scroll > 0)
        row["scroll_mean_speed"] = _safe_mean(scroll["ScrollSpeed"]) if "ScrollSpeed" in scroll.columns else np.nan
        row["scroll_std_speed"] = _safe_std(scroll["ScrollSpeed"]) if "ScrollSpeed" in scroll.columns else np.nan
        row["scroll_mean_delta"] = _safe_mean(scroll["ScrollDelta"]) if "ScrollDelta" in scroll.columns else np.nan
        row["scroll_count"] = float(n_scroll)  # 0 is the actual measured count

        rows.append(row)

    out = pd.DataFrame(rows)
    validate_session_features(out, data)
    return out


def validate_session_features(features: pd.DataFrame, source: pd.DataFrame) -> None:
    """Raise AssertionError if validation fails."""
    assert ID_COL in features.columns, "SessionID missing from features"
    # One row per session
    dup = features[ID_COL].duplicated().sum()
    assert dup == 0, f"Duplicate SessionID rows in features: {dup}"
    n_src = source[ID_COL].nunique()
    n_feat = features[ID_COL].nunique()
    assert n_feat == n_src, (
        f"Session count mismatch: source has {n_src}, features have {n_feat}"
    )
    # All source sessions present
    missing = set(source[ID_COL].unique()) - set(features[ID_COL].unique())
    assert not missing, f"Missing sessions in features: {sorted(missing)[:5]}"


def get_feature_columns(features: pd.DataFrame) -> List[str]:
    """Return behavioural feature column names (excludes IDs/labels/indicators metadata)."""
    exclude = {ID_COL} | set(LABEL_COLS) | set(OPTIONAL_LABEL_COLS)
    # Keep has_data indicators as metadata, not behavioural magnitudes
    exclude |= {c for c in features.columns if c.endswith("_has_data")}
    exclude |= {c for c in features.columns if c.endswith("_event_count")}
    return [c for c in features.columns if c not in exclude]


def build_feature_dictionary() -> pd.DataFrame:
    """Return a feature dictionary documenting each extracted feature."""
    rows = [
        # Mouse
        dict(feature_name="mouse_mean_speed", description="Mean mouse movement speed", formula="mean(Speed) over mousemove events with valid Speed",
             source_columns="Speed, EventType, SessionID, Timestamp", applicable_event_types="mousemove", missing_value_handling="NaN if no mousemove events with valid Speed"),
        dict(feature_name="mouse_std_speed", description="Standard deviation of mouse movement speed", formula="std(Speed, ddof=1) over mousemove events with valid Speed; 0 if exactly 1 observation",
             source_columns="Speed, EventType, SessionID, Timestamp", applicable_event_types="mousemove", missing_value_handling="NaN if no valid Speed observations"),
        dict(feature_name="mouse_mean_acceleration", description="Mean mouse acceleration", formula="mean(Acceleration) over mousemove events with valid Acceleration",
             source_columns="Acceleration, EventType, SessionID, Timestamp", applicable_event_types="mousemove", missing_value_handling="NaN if no valid observations"),
        dict(feature_name="mouse_std_acceleration", description="Standard deviation of mouse acceleration", formula="std(Acceleration, ddof=1); 0 if exactly 1 observation",
             source_columns="Acceleration, EventType, SessionID, Timestamp", applicable_event_types="mousemove", missing_value_handling="NaN if no valid observations"),
        dict(feature_name="mouse_total_distance", description="Total mouse movement distance (Euclidean path length)", formula="sum(sqrt((X_t-X_{t-1})^2 + (Y_t-Y_{t-1})^2)) over chronologically sorted mousemove events with valid X,Y",
             source_columns="X, Y, EventType, SessionID, Timestamp", applicable_event_types="mousemove", missing_value_handling="NaN if <2 valid positions and <1 position; 0.0 if exactly 1 position"),
        dict(feature_name="mouse_movement_frequency", description="Mouse movement events per second of session", formula="count(mousemove events) / session_duration_s",
             source_columns="EventType, SessionDurationMs/ElapsedMs, Timestamp, SessionID", applicable_event_types="mousemove + session duration", missing_value_handling="NaN if no mousemove events or duration unavailable/zero"),
        dict(feature_name="mouse_direction_changes", description="Number of significant direction changes in mouse trajectory", formula="count of consecutive angle differences > 30 degrees (pi/6) from atan2(DY,DX) over moving steps, sorted by Timestamp",
             source_columns="DX, DY, EventType, SessionID, Timestamp", applicable_event_types="mousemove", missing_value_handling="NaN if <2 valid movement vectors"),
        dict(feature_name="mouse_event_count", description="Count of mousemove events in session (availability indicator)", formula="count(EventType in mouse types)",
             source_columns="EventType, SessionID", applicable_event_types="mousemove", missing_value_handling="Always 0 or positive integer (measured count)"),
        dict(feature_name="mouse_has_data", description="Indicator: session has mouse data", formula="1 if mouse_event_count>0 else 0",
             source_columns="EventType, SessionID", applicable_event_types="mousemove", missing_value_handling="Always 0 or 1"),
        # Keyboard
        dict(feature_name="keyboard_mean_interkey_ms", description="Mean inter-key interval", formula="mean(InterKeyIntervalMs) over keyboard events with valid values",
             source_columns="InterKeyIntervalMs, EventType, SessionID, Timestamp", applicable_event_types="keypress/keydown/keyup", missing_value_handling="NaN if no valid observations"),
        dict(feature_name="keyboard_std_interkey_ms", description="Standard deviation of inter-key interval", formula="std(InterKeyIntervalMs, ddof=1); 0 if exactly 1 observation",
             source_columns="InterKeyIntervalMs, EventType, SessionID, Timestamp", applicable_event_types="keypress/keydown/keyup", missing_value_handling="NaN if no valid observations"),
        dict(feature_name="keyboard_typing_frequency", description="Typing events per second of session", formula="count(keyboard events) / session_duration_s",
             source_columns="EventType, SessionDurationMs/ElapsedMs, Timestamp, SessionID", applicable_event_types="keypress/keydown/keyup + session duration", missing_value_handling="NaN if no keyboard events or duration unavailable/zero"),
        dict(feature_name="keyboard_typing_count", description="Total typing event count", formula="count(keyboard events)",
             source_columns="EventType, SessionID", applicable_event_types="keypress/keydown/keyup", missing_value_handling="NaN if no keyboard events (unavailable); use keyboard_has_data/keyboard_event_count for zero-measured context"),
        dict(feature_name="keyboard_event_count", description="Count of keyboard events (availability indicator)", formula="count(EventType in key types)",
             source_columns="EventType, SessionID", applicable_event_types="keypress/keydown/keyup", missing_value_handling="Always 0 or positive integer"),
        dict(feature_name="keyboard_has_data", description="Indicator: session has keyboard data", formula="1 if keyboard_event_count>0 else 0",
             source_columns="EventType, SessionID", applicable_event_types="keypress/keydown/keyup", missing_value_handling="Always 0 or 1"),
        # Click
        dict(feature_name="click_mean_interval_ms", description="Mean click interval", formula="mean(ClickIntervalMs) over click events with valid values",
             source_columns="ClickIntervalMs, EventType, SessionID, Timestamp", applicable_event_types="click/mouse_click", missing_value_handling="NaN if no valid observations"),
        dict(feature_name="click_std_interval_ms", description="Standard deviation of click interval", formula="std(ClickIntervalMs, ddof=1); 0 if exactly 1 observation",
             source_columns="ClickIntervalMs, EventType, SessionID, Timestamp", applicable_event_types="click/mouse_click", missing_value_handling="NaN if no valid observations"),
        dict(feature_name="click_frequency", description="Clicks per second of session", formula="count(click events) / session_duration_s",
             source_columns="EventType, SessionDurationMs/ElapsedMs, Timestamp, SessionID", applicable_event_types="click/mouse_click + session duration", missing_value_handling="NaN if no click events or duration unavailable/zero"),
        dict(feature_name="click_count", description="Total click event count", formula="count(click events)",
             source_columns="EventType, SessionID", applicable_event_types="click/mouse_click", missing_value_handling="NaN if no click events"),
        dict(feature_name="click_event_count", description="Count of click events (availability indicator)", formula="count(EventType in click types)",
             source_columns="EventType, SessionID", applicable_event_types="click/mouse_click", missing_value_handling="Always 0 or positive integer"),
        dict(feature_name="click_has_data", description="Indicator: session has click data", formula="1 if click_event_count>0 else 0",
             source_columns="EventType, SessionID", applicable_event_types="click/mouse_click", missing_value_handling="Always 0 or 1"),
        # Scroll
        dict(feature_name="scroll_mean_speed", description="Mean scroll speed", formula="mean(ScrollSpeed) over scroll events with valid values",
             source_columns="ScrollSpeed, EventType, SessionID, Timestamp", applicable_event_types="scroll/wheel", missing_value_handling="NaN if no valid observations"),
        dict(feature_name="scroll_std_speed", description="Standard deviation of scroll speed", formula="std(ScrollSpeed, ddof=1); 0 if exactly 1 observation",
             source_columns="ScrollSpeed, EventType, SessionID, Timestamp", applicable_event_types="scroll/wheel", missing_value_handling="NaN if no valid observations"),
        dict(feature_name="scroll_mean_delta", description="Mean scroll delta", formula="mean(ScrollDelta) over scroll events with valid values",
             source_columns="ScrollDelta, EventType, SessionID, Timestamp", applicable_event_types="scroll/wheel", missing_value_handling="NaN if no valid observations"),
        dict(feature_name="scroll_count", description="Total scroll event count", formula="count(scroll events)",
             source_columns="EventType, SessionID", applicable_event_types="scroll/wheel", missing_value_handling="NaN if no scroll events"),
        dict(feature_name="scroll_event_count", description="Count of scroll events (availability indicator)", formula="count(EventType in scroll types)",
             source_columns="EventType, SessionID", applicable_event_types="scroll/wheel", missing_value_handling="Always 0 or positive integer"),
        dict(feature_name="scroll_has_data", description="Indicator: session has scroll data", formula="1 if scroll_event_count>0 else 0",
             source_columns="EventType, SessionID", applicable_event_types="scroll/wheel", missing_value_handling="Always 0 or 1"),
        # Session
        dict(feature_name="session_duration_s", description="Session duration in seconds", formula="max(SessionDurationMs)/1000 if available else max(ElapsedMs)/1000 else (max(Timestamp)-min(Timestamp))/1000",
             source_columns="SessionDurationMs, ElapsedMs, Timestamp, SessionID", applicable_event_types="all session events", missing_value_handling="NaN if duration cannot be determined"),
        dict(feature_name="session_mean_idle_ms", description="Mean idle time in milliseconds", formula="mean(IdleMs) over all session events with valid IdleMs",
             source_columns="IdleMs, SessionID", applicable_event_types="all session events", missing_value_handling="NaN if no valid IdleMs observations"),
        dict(feature_name="session_total_events", description="Total number of events in session", formula="count(all events for SessionID)",
             source_columns="EventType, SessionID", applicable_event_types="all session events", missing_value_handling="Always a measured count (>=1)"),
        dict(feature_name="session_event_frequency", description="Events per second over session duration", formula="session_total_events / session_duration_s",
             source_columns="EventType, SessionDurationMs/ElapsedMs, Timestamp, SessionID", applicable_event_types="all session events", missing_value_handling="NaN if duration unavailable or zero"),
        dict(feature_name="session_navigation_count", description="Number of navigation events/actions in session", formula="count(navigation events) if any else max(NavigationCount) if available else 0 (measured absence)",
             source_columns="EventType, NavigationCount, SessionID", applicable_event_types="navigation + all", missing_value_handling="0 if no navigation observed (actual measured value)"),
    ]
    return pd.DataFrame(rows)


def save_session_features(
    df: pd.DataFrame,
    output_path: str | Path,
    dict_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Extract features and save session_features.csv (and optional dictionary)."""
    features = extract_session_features(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    if dict_path is not None:
        d = build_feature_dictionary()
        dp = Path(dict_path)
        dp.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(dp, index=False)
    return features
