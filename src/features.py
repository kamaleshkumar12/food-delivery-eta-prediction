
"""Feature validation helpers for the Food Delivery ETA project."""

from collections.abc import Sequence

import pandas as pd


def validate_features(
    data: pd.DataFrame,
    expected_features: Sequence[str],
) -> pd.DataFrame:
    """Validate model input columns and preserve training feature order."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if not data.columns.is_unique:
        raise ValueError("Input contains duplicate column names.")

    expected = list(expected_features)
    missing = [name for name in expected if name not in data.columns]

    if missing:
        raise ValueError(
            "Missing required model features: " + ", ".join(missing)
        )

    result = data.loc[:, expected].copy()

    if result.isnull().any().any():
        raise ValueError(
            "Input contains missing values. Apply the same "
            "missing-value handling used during model training."
        )

    return result
