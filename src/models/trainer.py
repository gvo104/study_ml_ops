from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DATA_PATH,
    FIGURES_DIR,
    NUMERICAL_COLUMNS,
    RAW_TEXT_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
)
from src.config_loader import (
    ExperimentConfig,
    config_to_mlflow_params,
    load_config,
)
from src.data.loaders import load_dataset
from src.features.build_features import FeatureBuilder
from src.features.numerical import extract_numerical_features
from src.features.preprocess import preprocess_parallel
from src.models.factory import MODEL_REGISTRY
from src.models.registry import TrainResult, save_artifacts
from src.models.tracking import log_training_results, start_training_run
from src.utils import get_logger
from src.visualization.visualize import (
    plot_confusion_matrix,
    save_classification_report,
)


logger = get_logger(__name__)


def _prepare_text_column(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.Series:
    if config.training.repreprocess_text:
        logger.info("Re-preprocessing text column...")
        return pd.Series(
            preprocess_parallel(df[TEXT_COLUMN]),
            index=df.index,
        )

    return df[TEXT_COLUMN]


def _prepare_numerical_features(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> np.ndarray:
    if config.training.compute_numerical_from_raw:
        if RAW_TEXT_COLUMN not in df.columns:
            raise ValueError(
                f"Column '{RAW_TEXT_COLUMN}' required when "
                "compute_numerical_from_raw=true"
            )
        return np.array(
            [
                extract_numerical_features(str(text))[0]
                for text in df[RAW_TEXT_COLUMN]
            ]
        )

    return df[NUMERICAL_COLUMNS].values


def _resolve_run_name(
    config: ExperimentConfig,
    config_path: Path | str | None,
) -> str:
    if config.experiment.run_name:
        return config.experiment.run_name
    if config_path is not None:
        return Path(config_path).stem
    return config.model.name


def train(
    data_path: Path | str = DATA_PATH,
    config: ExperimentConfig | None = None,
    config_path: Path | str | None = None,
) -> TrainResult:
    config = config or load_config()
    run_name = _resolve_run_name(config, config_path)
    eval_dir = FIGURES_DIR / "experiments" / run_name

    logger.info("Loading dataset from %s...", data_path)
    df = load_dataset(data_path)

    x_text = _prepare_text_column(df, config)
    x_num = _prepare_numerical_features(df, config)
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
        test_size=config.training.test_size,
        random_state=config.training.random_state,
    )

    logger.info("Building features...")
    feature_builder = FeatureBuilder(config.features)
    x_train = feature_builder.fit_transform(x_train_text, x_train_num)
    x_test = feature_builder.transform(x_test_text, x_test_num)

    if config.training.oversample:
        logger.info("Oversampling...")
        sampler = RandomOverSampler(random_state=config.training.random_state)
        x_train, y_train = sampler.fit_resample(x_train, y_train)

    build_model = MODEL_REGISTRY.get(config.model.name)
    if build_model is None:
        available_models = list(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model: {config.model.name}. "
            f"Available: {available_models}"
        )

    params = config_to_mlflow_params(config)
    tags = {}
    if config.experiment.description:
        tags["description"] = config.experiment.description

    model_dir = None

    with start_training_run(params, run_name=run_name, tags=tags or None):
        logger.info("Training model: %s (%s)...", config.model.name, run_name)
        model = build_model(config)
        model.fit(x_train, y_train)

        logger.info("Evaluating...")
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro")
        weighted_f1 = f1_score(y_test, predictions, average="weighted")

        report = classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
        )

        print(f"\n=== {run_name} ===")
        print("Accuracy:", accuracy)
        print("Macro F1:", macro_f1)
        print(report)

        cm_path = plot_confusion_matrix(
            y_test,
            predictions,
            list(encoder.classes_),
            output_path=eval_dir / "confusion_matrix.png",
        )
        report_path = save_classification_report(
            report,
            eval_dir / "classification_report.txt",
        )

        if config.experiment.save_local_artifacts:
            model_dir = save_artifacts(
                model,
                feature_builder,
                encoder,
                accuracy,
                macro_f1,
            )
            plot_confusion_matrix(
                y_test,
                predictions,
                list(encoder.classes_),
            )

        log_training_results(
            model,
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            },
            model_dir,
            model_name=config.model.name,
            report_artifacts=[cm_path, report_path],
        )

    logger.info("Experiment '%s' completed.", run_name)

    return TrainResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        model_dir=model_dir or eval_dir,
    )
