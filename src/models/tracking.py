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


def mlflow_tracking_status() -> dict[str, Any]:
    return {
        "enabled": MLFLOW_ENABLED,
        "uri": MLFLOW_TRACKING_URI,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
    }


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


def list_mlflow_runs(max_results: int = 25) -> dict[str, Any]:
    """Return recent MLflow experiments and runs for the web UI."""
    status = mlflow_tracking_status()
    if not MLFLOW_ENABLED:
        return {
            **status,
            "connected": False,
            "error": "MLFLOW_ENABLED is false.",
            "experiments": [],
            "runs": [],
        }

    try:
        configure_mlflow_environment()
        from mlflow.entities import ViewType
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiments = client.search_experiments(
            view_type=ViewType.ACTIVE_ONLY,
            max_results=100,
        )
        experiment_payload = [
            {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "lifecycle_stage": experiment.lifecycle_stage,
                "artifact_location": experiment.artifact_location,
            }
            for experiment in experiments
        ]

        experiment_ids = [experiment.experiment_id for experiment in experiments]
        runs = []
        if experiment_ids:
            for run in client.search_runs(
                experiment_ids=experiment_ids,
                max_results=max_results,
                order_by=["attributes.start_time DESC"],
            ):
                runs.append(
                    {
                        "run_id": run.info.run_id,
                        "experiment_id": run.info.experiment_id,
                        "run_name": run.data.tags.get("mlflow.runName", run.info.run_id),
                        "status": run.info.status,
                        "start_time": run.info.start_time,
                        "end_time": run.info.end_time,
                        "artifact_uri": run.info.artifact_uri,
                        "metrics": dict(run.data.metrics),
                        "params": dict(run.data.params),
                        "tags": dict(run.data.tags),
                    }
                )

        return {
            **status,
            "connected": True,
            "error": None,
            "experiments": experiment_payload,
            "runs": runs,
        }
    except Exception as exc:
        logger.warning("Could not read MLflow runs (%s).", exc)
        return {
            **status,
            "connected": False,
            "error": str(exc),
            "experiments": [],
            "runs": [],
        }


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
