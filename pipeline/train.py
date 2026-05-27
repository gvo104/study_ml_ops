from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from imblearn.over_sampling import RandomOverSampler

from pipeline.load_data import load_dataset
from pipeline.preprocess import preprocess_parallel
from pipeline.feature_engineering import FeatureBuilder
from pipeline.save_artifacts import save_artifacts
from pipeline.utils import get_logger

logger = get_logger(__name__)


def main():

    logger.info("Loading dataset...")

    df = load_dataset()

    logger.info("Preprocessing text...")

    df["processed_text"] = preprocess_parallel(
        df["tokens_stemmed"]
    )

    X_text = df["processed_text"]

    X_num = df[
        [
            "num_of_characters",
            "num_of_sentences"
        ]
    ].values

    y = df["status"]

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    (
        X_train_text,
        X_test_text,
        y_train,
        y_test,
        X_train_num,
        X_test_num
    ) = train_test_split(
        X_text,
        y_encoded,
        X_num,
        test_size=0.2,
        random_state=101
    )

    logger.info("Building features...")

    feature_builder = FeatureBuilder()

    X_train = feature_builder.fit_transform(
        X_train_text,
        X_train_num
    )

    X_test = feature_builder.transform(
        X_test_text,
        X_test_num
    )

    logger.info("Oversampling...")

    ros = RandomOverSampler(random_state=101)

    X_train, y_train = ros.fit_resample(
        X_train,
        y_train
    )

    logger.info("Training model...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=101,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    logger.info("Evaluating...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nAccuracy:", accuracy)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_
        )
    )

    logger.info("Saving artifacts...")

    save_artifacts(
        model,
        feature_builder,
        encoder,
        accuracy
    )

    logger.info("Training completed.")


if __name__ == "__main__":
    main()