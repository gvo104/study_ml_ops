from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable

from src.api.bootstrap import ensure_dataset
from src.config import DATA_PATH, DEFAULT_CONFIG_PATH, MLFLOW_TRACKING_URI
from src.config_loader import load_config
from src.models.tracking import is_mlflow_available
from src.models.trainer import train


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PredictionRecord:
    id: int
    timestamp: str
    text_preview: str
    prediction: str
    confidence: float
    anomaly_flags: list[str]


@dataclass
class DriftNotification:
    level: str
    message: str
    created_at: str


@dataclass
class RetrainingStatus:
    state: str = "idle"
    started_at: str | None = None
    completed_at: str | None = None
    message: str = "Ready to start retraining."
    last_run_name: str | None = None
    last_accuracy: float | None = None
    last_macro_f1: float | None = None


@dataclass
class DashboardState:
    recent_predictions: deque[PredictionRecord] = field(
        default_factory=lambda: deque(maxlen=25)
    )
    recent_text_lengths: deque[int] = field(
        default_factory=lambda: deque(maxlen=25)
    )
    drift_notifications: deque[DriftNotification] = field(
        default_factory=lambda: deque(maxlen=5)
    )
    retraining: RetrainingStatus = field(default_factory=RetrainingStatus)
    prediction_counter: int = 0
    baseline_confidence: float | None = None
    baseline_text_length: float | None = None
    lock: Lock = field(default_factory=Lock)

    def record_prediction(self, text: str, result: dict[str, Any]) -> PredictionRecord:
        confidence = float(result["confidence"])
        text_length = len(text)
        flags = self._detect_anomalies(text, confidence)

        with self.lock:
            self.prediction_counter += 1
            record = PredictionRecord(
                id=self.prediction_counter,
                timestamp=_utc_now().isoformat(),
                text_preview=self._preview(text),
                prediction=result["prediction"],
                confidence=confidence,
                anomaly_flags=flags,
            )
            self.recent_predictions.appendleft(record)
            self.recent_text_lengths.appendleft(text_length)
            self._refresh_drift_notifications(text_length)
            return record

    def _detect_anomalies(self, text: str, confidence: float) -> list[str]:
        flags: list[str] = []

        if confidence < 0.55:
            flags.append("low_confidence")
        if len(text.strip()) < 20:
            flags.append("too_short")
        if len(text) > 1200:
            flags.append("too_long")

        return flags

    def _refresh_drift_notifications(self, text_length: int) -> None:
        window = list(self.recent_predictions)[:10]
        length_window = list(self.recent_text_lengths)[:10]
        if len(window) < 5:
            if self.baseline_confidence is None:
                self.baseline_confidence = sum(r.confidence for r in window) / len(window)
                self.baseline_text_length = sum(length_window) / len(length_window)
            return

        avg_confidence = sum(r.confidence for r in window) / len(window)
        anomaly_rate = sum(bool(r.anomaly_flags) for r in window) / len(window)
        avg_text_length = sum(length_window) / len(length_window)

        if self.baseline_confidence is None:
            self.baseline_confidence = avg_confidence
        if self.baseline_text_length is None:
            self.baseline_text_length = avg_text_length

        notifications: list[DriftNotification] = []
        confidence_delta = self.baseline_confidence - avg_confidence
        length_delta = abs(self.baseline_text_length - avg_text_length)

        if confidence_delta >= 0.18:
            notifications.append(
                DriftNotification(
                    level="high",
                    message=(
                        "Confidence drift detected: recent average confidence "
                        f"{avg_confidence:.2f} vs baseline {self.baseline_confidence:.2f}."
                    ),
                    created_at=_utc_now().isoformat(),
                )
            )
        elif anomaly_rate >= 0.35:
            notifications.append(
                DriftNotification(
                    level="medium",
                    message=(
                        "Anomaly rate increased in recent requests: "
                        f"{anomaly_rate:.0%} of the last {len(window)} predictions are flagged."
                    ),
                    created_at=_utc_now().isoformat(),
                )
            )

        if length_delta >= 120:
            notifications.append(
                DriftNotification(
                    level="medium",
                    message=(
                        "Input shape drift detected: recent text length pattern "
                        "differs noticeably from the baseline."
                    ),
                    created_at=_utc_now().isoformat(),
                )
            )

        self.drift_notifications.clear()
        for notification in notifications:
            self.drift_notifications.appendleft(notification)

    def dashboard_summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "recent_predictions": [
                    asdict(record) for record in self.recent_predictions
                ],
                "drift_notifications": [
                    asdict(notification)
                    for notification in self.drift_notifications
                ],
                "retraining": asdict(self.retraining),
            }

    def experiments_summary(
        self,
        predictor_metadata: dict[str, Any] | None,
        configs_dir: Path,
        startup_error: str | None = None,
    ) -> dict[str, Any]:
        configs = []
        for config_path in sorted(configs_dir.glob("*.yaml")):
            config = load_config(config_path)
            configs.append(
                {
                    "name": config.experiment.run_name or config_path.stem,
                    "config_path": str(config_path.relative_to(configs_dir.parent.parent)),
                    "model": config.model.name,
                    "description": config.experiment.description,
                    "tfidf_max_features": config.features.tfidf_max_features,
                    "svd_components": config.features.svd_components,
                    "oversample": config.training.oversample,
                }
            )

        with self.lock:
            return {
                "retraining": asdict(self.retraining),
                "drift_notifications": [
                    asdict(notification)
                    for notification in self.drift_notifications
                ],
                "active_model": predictor_metadata or {},
                "model_loaded": predictor_metadata is not None,
                "startup_error": startup_error,
                "experiment_configs": configs,
                "tracking": {
                    "enabled": is_mlflow_available(),
                    "uri": MLFLOW_TRACKING_URI,
                },
            }

    def trigger_retraining(
        self,
        reload_predictor: Callable[[], Any],
        config_path: Path = DEFAULT_CONFIG_PATH,
        data_path: Path = DATA_PATH,
        reason: str | None = None,
    ) -> bool:
        with self.lock:
            if self.retraining.state == "running":
                return False

            self.retraining = RetrainingStatus(
                state="running",
                started_at=_utc_now().isoformat(),
                message=reason or "Retraining is running in the background.",
            )

        thread = Thread(
            target=self._run_retraining_job,
            args=(reload_predictor, config_path, data_path),
            daemon=True,
        )
        thread.start()
        return True

    def _run_retraining_job(
        self,
        reload_predictor: Callable[[], Any],
        config_path: Path,
        data_path: Path,
    ) -> None:
        try:
            dataset_ready, dataset_message = ensure_dataset(data_path)
            if not dataset_ready:
                raise RuntimeError(dataset_message)

            config = load_config(config_path)
            run_name = config.experiment.run_name or Path(config_path).stem
            result = train(
                data_path=data_path,
                config=config,
                config_path=config_path,
            )
            reload_predictor()
            with self.lock:
                self.retraining = RetrainingStatus(
                    state="completed",
                    started_at=self.retraining.started_at,
                    completed_at=_utc_now().isoformat(),
                    message=(
                        "Retraining completed successfully and the model was "
                        f"reloaded. {dataset_message}"
                    ),
                    last_run_name=run_name,
                    last_accuracy=result.accuracy,
                    last_macro_f1=result.macro_f1,
                )
        except Exception as exc:
            with self.lock:
                self.retraining = RetrainingStatus(
                    state="failed",
                    started_at=self.retraining.started_at,
                    completed_at=_utc_now().isoformat(),
                    message=f"Retraining failed: {exc}",
                )

    @staticmethod
    def _preview(text: str, limit: int = 96) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 1]}..."
