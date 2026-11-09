"""Behavioural fingerprint generation for BotPrint.

Transforms session-level behavioural features into numerical fingerprint
vectors. This module is exploratory: scaling parameters are estimated
from the available sessions and the fitted artifacts are saved for
reproducibility.

IMPORTANT:
- Fingerprints are behavioural representations for this research.
  They are NOT claimed to uniquely identify users.
- SessionID, BotID, CoordinationGroupID, UserType, BotType are
  identifiers/labels and are NEVER included in fingerprint vectors.
- For future model experiments, do NOT fit scalers/imputers on
  evaluation data; use training-only fitting and apply to evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

ID_COLS = ["SessionID"]
LABEL_COLS = ["UserType", "BotType"]
OPTIONAL_LABEL_COLS = ["CoordinationGroupID", "SessionSource", "BotID"]
EXCLUDED_FROM_FINGERPRINT = set(ID_COLS + LABEL_COLS + OPTIONAL_LABEL_COLS)

# Suffixes that indicate metadata/availability, not behavioural magnitude
INDICATOR_SUFFIXES = ("_has_data", "_event_count")


@dataclass
class FingerprintConfig:
    feature_columns: List[str]
    indicator_columns: List[str]
    excluded_columns: List[str]
    imputation_strategy: str
    selected_scaler: str
    scaling_fit_note: str
    n_sessions: int
    n_features: int


def _is_indicator(col: str) -> bool:
    return col.endswith(INDICATOR_SUFFIXES)


def select_fingerprint_features(
    df: pd.DataFrame,
    min_non_missing_ratio: float = 0.25,
    include_indicators: bool = False,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Select meaningful behavioural features for fingerprinting.

    Returns (feature_columns, indicator_columns, report).
    """
    excluded = [c for c in df.columns if c in EXCLUDED_FROM_FINGERPRINT]
    candidates = [c for c in df.columns if c not in EXCLUDED_FROM_FINGERPRINT]

    indicators = [c for c in candidates if _is_indicator(c)]
    numeric = []
    for c in candidates:
        if _is_indicator(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)

    missingness = {}
    kept = []
    dropped = []
    for c in numeric:
        ratio = float(df[c].notna().mean())
        missingness[c] = {
            "non_missing_ratio": round(ratio, 4),
            "n_missing": int(df[c].isna().sum()),
            "n_non_missing": int(df[c].notna().sum()),
        }
        # Keep if at least some sessions have values; extremely sparse can still
        # be informative but we document. Require minimum coverage.
        if ratio >= min_non_missing_ratio:
            kept.append(c)
        else:
            dropped.append(c)

    # session_total_events/navigation_count are measured counts (behavioural)
    # but not the same as availability indicators; keep if present.
    final = kept
    if include_indicators:
        final = final + [c for c in indicators if c not in final]

    report = {
        "n_rows": int(len(df)),
        "excluded_columns": excluded,
        "indicator_columns": indicators,
        "dropped_low_coverage": dropped,
        "missingness": missingness,
        "selected_features": final,
    }
    return final, indicators, report


def build_raw_matrix(
    df: pd.DataFrame, feature_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (raw_matrix, ids, labels) with aligned indices."""
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")
    raw = df[feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    ids = df[[c for c in ID_COLS if c in df.columns]].copy()
    labels = df[[c for c in LABEL_COLS + OPTIONAL_LABEL_COLS if c in df.columns]].copy()
    return raw, ids, labels


def compare_scalers(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit StandardScaler and RobustScaler on imputed data and compare.

    Imputation uses median (robust default). Scaling is fit on the
    provided X (all available sessions) for this exploratory dataset.
    """
    feature_names = list(X.columns)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=feature_names, index=X.index)

    scalers = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
    }
    results = {}
    comparison_rows = []
    for name, scaler in scalers.items():
        Xs = scaler.fit_transform(X_imp)
        Xs_df = pd.DataFrame(Xs, columns=feature_names, index=X.index)
        results[name] = {
            "scaler": scaler,
            "imputer": imputer,
            "matrix": Xs_df,
        }
        # Compare distribution properties
        comparison_rows.append({
            "scaler": name,
            "mean_of_means": float(np.nanmean(Xs_df.mean(axis=0))),
            "mean_of_stds": float(np.nanmean(Xs_df.std(axis=0, ddof=1))),
            "median_of_medians": float(np.nanmedian(Xs_df.median(axis=0))),
            "max_abs_value": float(np.nanmax(np.abs(Xs_df.values))),
            "note": "Fit on all available sessions (exploratory; do not refit on eval in future).",
        })

    comparison = pd.DataFrame(comparison_rows)
    return comparison, results


def select_scaler(comparison: pd.DataFrame) -> str:
    """Select scaler based on robustness to outliers.

    BotPrint behavioural data can contain outliers (e.g., extreme
    durations/speeds). RobustScaler uses median/IQR and is less
    sensitive to outliers than StandardScaler (mean/std), making it
    a reasonable default for exploratory fingerprinting. The comparison
    table is saved so the choice is transparent.
    """
    # Prefer RobustScaler for behavioural data with potential outliers;
    # if distributions are already well-behaved both are acceptable.
    if "RobustScaler" in comparison["scaler"].values:
        return "RobustScaler"
    return str(comparison["scaler"].iloc[0])


def generate_fingerprints(
    df: pd.DataFrame,
    min_non_missing_ratio: float = 0.25,
) -> Dict[str, Any]:
    """Generate fingerprint dataset and artifacts.

    Returns a dict with raw/processed matrices, ids, labels, config,
    comparison, fitted scaler, and imputer.
    """
    feature_cols, indicator_cols, selection_report = select_fingerprint_features(
        df, min_non_missing_ratio=min_non_missing_ratio, include_indicators=False
    )
    raw, ids, labels = build_raw_matrix(df, feature_cols)

    comparison, scaler_results = compare_scalers(raw)
    chosen = select_scaler(comparison)
    chosen_res = scaler_results[chosen]
    processed = chosen_res["matrix"]
    imputed = pd.DataFrame(
        chosen_res["imputer"].transform(raw), columns=feature_cols, index=raw.index
    )

    config = FingerprintConfig(
        feature_columns=feature_cols,
        indicator_columns=indicator_cols,
        excluded_columns=selection_report["excluded_columns"],
        imputation_strategy="median",
        selected_scaler=chosen,
        scaling_fit_note=(
            "Scaling/imputation parameters estimated from all available sessions "
            "in this exploratory fingerprint dataset. For future supervised "
            "evaluation, fit preprocessing ONLY on training data and apply to "
            "evaluation data to avoid leakage."
        ),
        n_sessions=int(len(df)),
        n_features=int(len(feature_cols)),
    )

    # Fingerprint dataset: id + labels + fingerprint vector
    fingerprint_df = pd.concat(
        [ids.reset_index(drop=True), labels.reset_index(drop=True),
         processed.reset_index(drop=True)], axis=1
    )

    # Summary
    summary = pd.DataFrame({
        "metric": [
            "n_sessions", "n_fingerprint_features", "selected_scaler",
            "imputation_strategy", "n_excluded_id_label_cols",
            "n_indicator_cols_available", "fingerprint_claim",
        ],
        "value": [
            config.n_sessions, config.n_features, config.selected_scaler,
            config.imputation_strategy, len(config.excluded_columns),
            len(indicator_cols),
            "Behavioural representations for research; not unique user identification.",
        ],
    })

    return {
        "raw_matrix": raw,
        "imputed_matrix": imputed,
        "processed_matrix": processed,
        "fingerprint_df": fingerprint_df,
        "ids": ids,
        "labels": labels,
        "config": config,
        "comparison": comparison,
        "selection_report": selection_report,
        "summary": summary,
        "scaler": chosen_res["scaler"],
        "imputer": chosen_res["imputer"],
    }


def save_fingerprint_artifacts(
    artifacts: Dict[str, Any], processed_dir: str | Path, analysis_dir: str | Path
) -> None:
    """Save matrices, fingerprint dataset, config, scaler, and reports."""
    processed_dir = Path(processed_dir)
    analysis_dir = Path(analysis_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    artifacts["raw_matrix"].to_csv(processed_dir / "fingerprint_raw_matrix.csv", index=False)
    artifacts["imputed_matrix"].to_csv(processed_dir / "fingerprint_imputed_matrix.csv", index=False)
    artifacts["processed_matrix"].to_csv(processed_dir / "fingerprint_processed_matrix.csv", index=False)
    artifacts["fingerprint_df"].to_csv(processed_dir / "fingerprint_dataset.csv", index=False)
    artifacts["comparison"].to_csv(analysis_dir / "fingerprint_scaler_comparison.csv", index=False)
    artifacts["summary"].to_csv(analysis_dir / "fingerprint_summary.csv", index=False)

    # Selection report
    sel = artifacts["selection_report"].copy()
    miss_rows = []
    for feat, info in sel.get("missingness", {}).items():
        miss_rows.append({"feature": feat, **info})
    pd.DataFrame(miss_rows).to_csv(analysis_dir / "fingerprint_feature_missingness.csv", index=False)

    config = artifacts["config"]
    config_path = processed_dir / "fingerprint_config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    joblib.dump(
        {
            "scaler": artifacts["scaler"],
            "imputer": artifacts["imputer"],
            "config": asdict(config),
        },
        processed_dir / "fingerprint_scaler.joblib",
    )
