from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, NUMERICAL_COLUMNS, TARGET_COLUMN, TEXT_COLUMN
from src.data.validation import validate_dataset


def load_dataset(data_path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load and validate the processed training CSV."""
    df = pd.read_csv(data_path, index_col=0)
    required = [TEXT_COLUMN, TARGET_COLUMN, *NUMERICAL_COLUMNS]
    df = df.dropna(subset=required).reset_index(drop=True)

    validate_dataset(df)

    return df
