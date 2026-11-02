# ai4ct/benchmarks/data.py
"""
Data loading and preprocessing.

Loads the Excel radiotherapy plan complexity dataset, splits into
train/test, and applies RobustScaler normalization.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from typing import Tuple
from pathlib import Path

from .config import SEED, FEATURE_COLS, LABEL_COL, TEST_SIZE


def load_data(
    file_path: str,
    feature_cols: Tuple[int, int] = FEATURE_COLS,
    label_col: int = LABEL_COL,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Excel data, split and normalize.

    Parameters
    ----------
    file_path : str
        Path to the .xlsx file.
    feature_cols : (start, end)
        Excel column range for features (0-indexed, end exclusive).
    label_col : int
        Excel column index for the label.
    test_size : float
        Fraction of data for test split.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Normalized train/test splits.
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(
            f"Feature table not found: {file_path}. Restore the historical table "
            "under data/processed or pass --data with an existing workbook."
        )
    df = pd.read_excel(file_path)
    X = df.iloc[:, feature_cols[0]:feature_cols[1]].values.astype(np.float64)
    y = df.iloc[:, label_col].values.astype(np.int64)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test
