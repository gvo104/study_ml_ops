import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from src.config import (
    MLFLOW_ENABLED,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    configure_mlflow_environment,
)

logger = logging.getLogger(__name__)


def is_mlflow_available() -> bool:
    if not MLFLOW_ENABLED:
        return False

    try:
        configure_mlflow_environment()
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.search_experiments(max_results=1)
        return True
    except Exception as exc:
        logger.warning(
            "MLflow unavailable (%s). Using local artifacts only.",
            exc,
        )
        return False


@contextmanager
def start_training_run(
    params: dict[str, Any],
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Generator[Any, None, None]:
    """Start an MLflow run if tracking server is reachable."""
    if not is_mlflow_available():
        yield None
        return

    import mlflow

    configure_mlflow_environment()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name or None) as run:
        mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        yield run


def _log_model(model, model_name: str) -> None:
    import mlflow

    if model_name == "xgboost":
        import mlflow.xgboost

        mlflow.xgboost.log_model(model, name="model")
    elif model_name == "lightgbm":
        import mlflow.lightgbm

        mlflow.lightgbm.log_model(model, name="model")
    else:
        mlflow.sklearn.log_model(model, name="model")


def log_training_results(
    model,
    metrics: dict[str, float],
    artifacts_dir: Path | None,
    model_name: str = "xgboost",
    report_artifacts: list[Path] | None = None,
) -> None:
    """Log metrics, model and artifacts to the active MLflow run.

    report_artifacts:
        paths to log under evaluation/ (e.g. confusion matrix).
    """
    if not is_mlflow_available():
        return

    import mlflow

    configure_mlflow_environment()

    try:
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        _log_model(model, model_name)

        for artifact_path in report_artifacts or []:
            path = Path(artifact_path)
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="evaluation")

        if artifacts_dir is not None and Path(artifacts_dir).exists():
            mlflow.log_artifacts(
                str(artifacts_dir),
                artifact_path="model_bundle",
            )

        logger.info("MLflow run logged successfully.")
    except Exception as exc:
        logger.warning(
            "MLflow logging failed (%s). Local training artifacts are intact.",
            exc,
        )
