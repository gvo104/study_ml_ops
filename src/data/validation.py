import pandas as pd

from src.config import NUMERICAL_COLUMNS, TARGET_COLUMN, TEXT_COLUMN


class DatasetValidationError(ValueError):
    """Raised when the dataset schema does not match expectations."""


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate that required columns exist in the training dataset."""
    required = {TEXT_COLUMN, TARGET_COLUMN, *NUMERICAL_COLUMNS}
    missing = required - set(df.columns)

    if missing:
        raise DatasetValidationError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise DatasetValidationError("Dataset is empty after loading.")

    if df[TARGET_COLUMN].isna().any():
        raise DatasetValidationError(
            f"Target column '{TARGET_COLUMN}' contains missing values."
        )
