import pytest
import pandas as pd

from src.config import NUMERICAL_COLUMNS, TARGET_COLUMN, TEXT_COLUMN
from src.data.validation import DatasetValidationError, validate_dataset
from src.features.build_features import FeatureBuilder
from src.features.numerical import extract_numerical_features
from src.features.preprocess import clean_text, preprocess_single


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            TEXT_COLUMN: ["feel sad", "happy day"],
            TARGET_COLUMN: ["Depression", "Normal"],
            NUMERICAL_COLUMNS[0]: [10, 20],
            NUMERICAL_COLUMNS[1]: [1, 2],
        }
    )


def test_clean_text_removes_url():
    result = clean_text("Check http://example.com now")
    assert "http" not in result


def test_extract_numerical_features():
    features = extract_numerical_features("Hello! How are you?")
    assert features.shape == (1, 2)
    assert features[0, 0] == len("Hello! How are you?")
    assert features[0, 1] == 2


def test_validate_dataset_ok(sample_df):
    validate_dataset(sample_df)


def test_validate_dataset_missing_column(sample_df):
    broken = sample_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(DatasetValidationError):
        validate_dataset(broken)


def test_feature_builder_shapes():
    texts = [
        "feel very sad and hopeless today",
        "happy wonderful day outside",
        "anxious about everything around me",
    ]
    numerical = [[10, 1], [20, 2], [30, 3]]
    builder = FeatureBuilder()
    x_train = builder.fit_transform(texts, numerical)
    x_test = builder.transform(["feeling much better now"], [[5, 1]])

    assert x_train.shape[0] == 3
    assert x_test.shape[0] == 1
    assert x_train.shape[1] == x_test.shape[1]
