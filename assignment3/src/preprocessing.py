"""Data loading and preprocessing helpers for the Smart Factory Alert Agent.

This module is intentionally deterministic and free of external services so that
the agent can be unit-tested without network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: List[str] = ["timestamp", "temp", "pressure", "vibration", "label"]
SENSOR_COLUMNS: List[str] = ["temp", "pressure", "vibration"]

# Normal operating ranges as defined in the assignment brief.
NORMAL_RANGES: Dict[str, Tuple[float, float]] = {
    "temp": (45.0, 50.0),
    "pressure": (1.00, 1.05),
    "vibration": (0.02, 0.04),
}


@dataclass
class CleaningSummary:
    rows: int
    missing_before: Dict[str, int] = field(default_factory=dict)
    missing_after: Dict[str, int] = field(default_factory=dict)
    imputed: Dict[str, int] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the sensor CSV and validate its schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sensor data file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Keep a stable column order.
    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Invalid or missing timestamp values detected.")
    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningSummary]:
    """Handle missing values and add standardized (z-score) sensor columns.

    Missing sensor values are imputed with the column median. Standardisation
    uses population std (ddof=0) and guards against zero variance.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    summary = CleaningSummary(rows=len(df), columns=list(df.columns))
    for col in SENSOR_COLUMNS:
        summary.missing_before[col] = int(df[col].isna().sum())

    cleaned = df.copy()
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

    for col in SENSOR_COLUMNS:
        median = cleaned[col].median()
        if pd.isna(median):
            median = 0.0
        mask = cleaned[col].isna()
        summary.imputed[col] = int(mask.sum())
        cleaned[col] = cleaned[col].fillna(median)

    for col in SENSOR_COLUMNS:
        std = cleaned[col].std(ddof=0)
        if std == 0 or pd.isna(std):
            cleaned[f"{col}_z"] = 0.0
        else:
            cleaned[f"{col}_z"] = (cleaned[col] - cleaned[col].mean()) / std

    for col in SENSOR_COLUMNS:
        summary.missing_after[col] = int(cleaned[col].isna().sum())

    return cleaned, summary


def cleaning_summary_to_text(summary: CleaningSummary) -> str:
    lines = [
        f"Rows: {summary.rows}",
        "Missing values before cleaning:",
    ]
    for col in SENSOR_COLUMNS:
        lines.append(f"  - {col}: {summary.missing_before.get(col, 0)}")
    lines.append("Imputed values:")
    for col in SENSOR_COLUMNS:
        lines.append(f"  - {col}: {summary.imputed.get(col, 0)}")
    lines.append("Missing values after cleaning:")
    for col in SENSOR_COLUMNS:
        lines.append(f"  - {col}: {summary.missing_after.get(col, 0)}")
    return "\n".join(lines)


__all__ = [
    "REQUIRED_COLUMNS",
    "SENSOR_COLUMNS",
    "NORMAL_RANGES",
    "CleaningSummary",
    "load_data",
    "clean_data",
    "cleaning_summary_to_text",
]
