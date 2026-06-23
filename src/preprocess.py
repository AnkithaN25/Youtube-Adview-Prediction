"""
Data loading and preprocessing for YouTube AdView prediction.

Handles:
- Category encoding (A–H → 1–8)
- Dropping rows with sentinel 'F' values
- Numeric type conversion
- ISO 8601 duration parsing (PT#H#M#S → total seconds)
- Label encoding of vidid, published, duration
- Outlier filtering on adview (training only)
- MinMaxScaler normalisation

Usage
-----
    from src.preprocess import load_and_prepare

    X_train, X_test, y_train, y_test, scaler = load_and_prepare("data/train.csv")
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.config import (
    ADVIEW_CAP,
    CATEGORY_MAP,
    NUMERIC_COLS,
    RANDOM_STATE,
    SCALER_PATH,
    TEST_SIZE,
)


# ── Duration parsing ──────────────────────────────────────────────────────────

def _parse_iso_duration(x: str) -> str:
    """
    Convert ISO 8601 duration string (e.g. 'PT1H23M45S') to 'HH:MM:SS'.

    Handles missing hours, minutes, or seconds gracefully.
    """
    y = x[2:]           # strip leading 'PT'
    h = m = s = mm = ""
    for ch in y:
        if ch not in ("H", "M", "S"):
            mm += ch
        elif ch == "H":
            h, mm = mm, ""
        elif ch == "M":
            m, mm = mm, ""
        else:
            s, mm = mm, ""
    return f"{h or '00'}:{m or '00'}:{s or '00'}"


def _hms_to_seconds(time_str: str) -> int:
    """Convert 'HH:MM:SS' string to total seconds."""
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _duration_to_seconds(series: pd.Series) -> pd.Series:
    """Apply ISO duration → seconds conversion to a Series."""
    return series.apply(_parse_iso_duration).apply(_hms_to_seconds)


# ── Main preprocessing ────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Clean and encode a raw train or test DataFrame.

    Steps:
        1. Map category letters to integers.
        2. Drop rows where views/likes/dislikes/comment == 'F' (corrupt data).
        3. Convert numeric columns to float.
        4. Label-encode vidid, published.
        5. Parse ISO 8601 duration → seconds.
        6. (Train only) convert adview to numeric and cap at ADVIEW_CAP.

    Args:
        df:       Raw DataFrame loaded from CSV.
        is_train: Set True for training data (also processes adview column).

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Category encoding
    df["category"] = df["category"].map(CATEGORY_MAP)

    # Drop rows with sentinel 'F' values in engagement columns
    for col in ["views", "likes", "dislikes", "comment"]:
        df = df[df[col] != "F"]

    # Convert engagement columns to numeric
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col])

    if is_train:
        df["adview"] = pd.to_numeric(df["adview"])
        df = df[df["adview"] < ADVIEW_CAP]

    # Label-encode string identifiers
    for col in ("vidid", "published"):
        df[col] = LabelEncoder().fit_transform(df[col])

    # Parse duration to seconds
    raw_duration = df["duration"].copy()
    df["duration"] = _duration_to_seconds(raw_duration)

    return df.reset_index(drop=True)


def load_and_prepare(
    csv_path: str,
    is_train: bool = True,
    save_scaler: bool = False,
    scaler: MinMaxScaler | None = None,
):
    """
    Load CSV, clean, split (train only), and scale features.

    Args:
        csv_path:    Path to train.csv or test.csv.
        is_train:    True → split into train/val and fit scaler.
                     False → apply provided scaler to full dataset.
        save_scaler: If True, persist the fitted scaler to SCALER_PATH.
        scaler:      Pre-fitted scaler to apply (required when is_train=False).

    Returns (train mode):
        X_train, X_test, y_train, y_test, fitted_scaler

    Returns (test/inference mode):
        X_scaled, y (or None if adview absent), scaler
    """
    df = clean(pd.read_csv(csv_path), is_train=is_train)
    print(f"Loaded {csv_path}: {df.shape[0]:,} rows after cleaning")

    if is_train:
        # Target is adview (column index 1 after cleaning)
        Y = pd.DataFrame(df["adview"].values, columns=["target"])
        X = df.drop(columns=["adview"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)       # transform only — no refit

        if save_scaler:
            import os
            os.makedirs("models", exist_ok=True)
            joblib.dump(scaler, SCALER_PATH)
            print(f"Scaler saved → {SCALER_PATH}")

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    else:
        if scaler is None:
            scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(df)
        return X_scaled, None, scaler
