import click
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config import (
    DATA_PATH,
    NUMERICAL_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TEXT_COLUMN,
)
from src.features.build_features import FeatureBuilder
from src.features.preprocess import preprocess_parallel
from src.models.artifacts import save_artifacts
from src.utils import get_logger


logger = get_logger(__name__)


def load_dataset(data_path=DATA_PATH):
    df = pd.read_csv(data_path, index_col=0)

    return df.dropna().reset_index(drop=True)


def train(data_path=DATA_PATH):
    logger.info("Loading dataset from %s...", data_path)
    df = load_dataset(data_path)

    logger.info("Preprocessing text...")
    df["processed_text"] = preprocess_parallel(df[TEXT_COLUMN])

    x_text = df["processed_text"]
    x_num = df[NUMERICAL_COLUMNS].values
    y = df[TARGET_COLUMN]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    (
        x_train_text,
        x_test_text,
        y_train,
        y_test,
        x_train_num,
        x_test_num,
    ) = train_test_split(
        x_text,
        y_encoded,
        x_num,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    logger.info("Building features...")
    feature_builder = FeatureBuilder()
    x_train = feature_builder.fit_transform(x_train_text, x_train_num)
    x_test = feature_builder.transform(x_test_text, x_test_num)

    logger.info("Oversampling...")
    sampler = RandomOverSampler(random_state=RANDOM_STATE)
    x_train, y_train = sampler.fit_resample(x_train, y_train)

    logger.info("Training model...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_train)

    logger.info("Evaluating...")
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", accuracy)
    print(classification_report(y_test, predictions, target_names=encoder.classes_))

    logger.info("Saving artifacts...")
    save_artifacts(model, feature_builder, encoder, accuracy)

    logger.info("Training completed.")

    return accuracy


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default=str(DATA_PATH),
    show_default=True,
    help="Path to the processed training CSV.",
)
def main(data_path):
    train(data_path)


if __name__ == "__main__":
    main()
