from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily

from drift.monitoring.config import MonitoringSettings


STATUS_TO_CODE = {
    "ok": 0,
    "warning": 1,
    "critical": 2,
    "insufficient_data": 3,
    "unknown": -1,
}
METRIC_SPECS = {
    "drift_token_distribution_jsd": "Replay/latest token distribution Jensen-Shannon divergence.",
    "drift_model_prediction_distribution_jsd": "Replay/latest model prediction distribution drift.",
    "drift_model_confidence_mean": "Replay/latest mean model confidence.",
    "drift_target_distribution_jsd": "Replay/latest expert label distribution drift.",
    "drift_model_expert_disagreement_rate": "Replay/latest disagreement rate between model and expert.",
    "drift_model_expert_macro_f1": "Replay/latest macro F1 between model predictions and expert labels.",
    "drift_expert_confidence_mean": "Replay/latest mean expert confidence.",
    "drift_token_label_association_drift": "Replay/latest token-label association drift score.",
    "drift_window_status_code": "Replay/latest window status code: 0=ok, 1=warning, 2=critical.",
    "drift_window_index": "Replay/latest evaluated sliding window index.",
    "drift_insufficient_data_flag": "Whether replay/latest association metric lacked enough data.",
}
WINDOW_METRIC_SPECS = {
    "drift_window_token_distribution_jsd": (
        "token_distribution_jsd",
        "Window history token distribution Jensen-Shannon divergence.",
    ),
    "drift_window_model_prediction_distribution_jsd": (
        "model_prediction_distribution_jsd",
        "Window history model prediction distribution drift.",
    ),
    "drift_window_model_confidence_mean": (
        "model_confidence_mean",
        "Window history mean model confidence.",
    ),
    "drift_window_target_distribution_jsd": (
        "target_distribution_jsd",
        "Window history expert label distribution Jensen-Shannon divergence.",
    ),
    "drift_window_model_expert_disagreement_rate": (
        "model_expert_disagreement_rate",
        "Window history disagreement rate between model and expert.",
    ),
    "drift_window_model_expert_macro_f1": (
        "model_expert_macro_f1",
        "Window history macro F1 between model predictions and expert labels.",
    ),
    "drift_window_expert_confidence_mean": (
        "expert_confidence_mean",
        "Window history mean expert confidence.",
    ),
    "drift_window_token_label_association_drift": (
        "token_label_association_drift",
        "Window history token-label association drift score.",
    ),
}
FILE_PRESENT_METRIC = "drift_metrics_file_present"
WINDOW_FILE_PRESENT_METRIC = "drift_window_metrics_file_present"
LAST_UPDATE_METRIC = "drift_metrics_last_update_timestamp"
LABEL_NAMES = ("run_id", "mode", "phase", "status")
WINDOW_LABEL_NAMES = ("run_id", "mode", "phase", "status", "window_index")


@dataclass(frozen=True)
class MetricsSnapshot:
    payload: dict[str, Any] | None
    metrics_path: Path | None
    window_metrics_path: Path | None = None
    windows: list[dict[str, Any]] | None = None
    parse_errors: list[str] | None = None
    replay_enabled: bool = False
    replay_window: dict[str, Any] | None = None

    @property
    def file_present(self) -> bool:
        return self.payload is not None and self.metrics_path is not None

    @property
    def window_file_present(self) -> bool:
        return bool(self.windows) and self.window_metrics_path is not None

    @property
    def window_count(self) -> int:
        return len(self.windows or [])


def resolve_metrics_path(settings: MonitoringSettings) -> Path | None:
    if settings.explicit_metrics_path is not None:
        return settings.explicit_metrics_path

    if not settings.runs_root.exists():
        return None

    candidates = sorted(
        (
            child / "prom_metrics_latest.json"
            for child in settings.runs_root.iterdir()
            if child.is_dir() and (child / "prom_metrics_latest.json").exists()
        ),
        key=lambda path: path.parent.name,
    )
    return candidates[-1] if candidates else None


def resolve_window_metrics_path(
    settings: MonitoringSettings,
    metrics_path: Path | None,
) -> Path | None:
    if settings.explicit_window_metrics_path is not None:
        return settings.explicit_window_metrics_path
    if metrics_path is None:
        return None
    return metrics_path.parent / "window_metrics.jsonl"


def load_metrics_snapshot(settings: MonitoringSettings) -> MetricsSnapshot:
    parse_errors: list[str] = []
    metrics_path = resolve_metrics_path(settings)
    payload = _load_json(metrics_path, "latest metrics", parse_errors)
    window_metrics_path = resolve_window_metrics_path(settings, metrics_path)
    windows = _load_window_metrics(window_metrics_path, parse_errors)
    replay_window = select_replay_window(
        windows,
        settings,
        now=monotonic(),
    )
    return MetricsSnapshot(
        payload=payload,
        metrics_path=metrics_path,
        window_metrics_path=window_metrics_path,
        windows=windows,
        parse_errors=parse_errors,
        replay_enabled=settings.replay_enabled,
        replay_window=replay_window,
    )


def select_replay_window(
    windows: list[dict[str, Any]],
    settings: MonitoringSettings,
    now: float | None = None,
) -> dict[str, Any] | None:
    if not settings.replay_enabled or not windows:
        return None
    current_time = monotonic() if now is None else now
    elapsed = max(0.0, current_time - settings.replay_started_at)
    raw_index = int(elapsed // settings.replay_step_seconds)
    if settings.replay_loop:
        index = raw_index % len(windows)
    else:
        index = min(raw_index, len(windows) - 1)
    return windows[index]


def generate_metrics_text(snapshot: MetricsSnapshot) -> bytes:
    registry = CollectorRegistry()
    families = []
    families.extend(_build_file_state_metrics(snapshot))
    if snapshot.file_present:
        latest_payload = _effective_latest_payload(snapshot)
        families.extend(_build_latest_metrics(latest_payload))
        families.extend(_build_window_status_metrics(latest_payload, snapshot))
        families.extend(_build_window_history_metrics(snapshot))

    if families:
        registry.register(_StaticCollector(families))
    return generate_latest(registry)


def _load_json(
    path: Path | None,
    label: str,
    parse_errors: list[str],
) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        parse_errors.append(f"{label}: {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        parse_errors.append(f"{label}: {path}: expected JSON object")
        return None
    return payload


def _load_window_metrics(
    path: Path | None,
    parse_errors: list[str],
) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []

    windows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"window metrics: {path}:{line_number}: {exc}")
            continue
        if not isinstance(payload, dict):
            parse_errors.append(
                f"window metrics: {path}:{line_number}: expected JSON object"
            )
            continue
        windows.append(payload)
    return windows


def _effective_latest_payload(snapshot: MetricsSnapshot) -> dict[str, Any]:
    assert snapshot.payload is not None
    if snapshot.replay_enabled and snapshot.replay_window is not None:
        return _build_latest_payload_from_window(snapshot.payload, snapshot.replay_window)
    return snapshot.payload


def _build_latest_payload_from_window(
    base_payload: dict[str, Any],
    window: dict[str, Any],
) -> dict[str, Any]:
    status = str(window.get("overall_status", "unknown"))
    run_id = str(base_payload.get("run_id", "unknown"))
    mode = str(base_payload.get("drift_runner_mode", "unknown"))
    phase = str(window.get("phase", "unknown"))
    return {
        "run_id": run_id,
        "drift_runner_mode": mode,
        "drift_token_distribution_jsd": window.get("token_distribution_jsd"),
        "drift_model_prediction_distribution_jsd": window.get(
            "model_prediction_distribution_jsd"
        ),
        "drift_model_confidence_mean": window.get("model_confidence_mean"),
        "drift_target_distribution_jsd": window.get("target_distribution_jsd"),
        "drift_model_expert_disagreement_rate": window.get(
            "model_expert_disagreement_rate"
        ),
        "drift_model_expert_macro_f1": window.get("model_expert_macro_f1"),
        "drift_expert_confidence_mean": window.get("expert_confidence_mean"),
        "drift_token_label_association_drift": window.get(
            "token_label_association_drift"
        ),
        "drift_window_status_code": STATUS_TO_CODE.get(status, -1),
        "drift_window_phase": phase,
        "drift_window_index": window.get("window_index"),
        "drift_insufficient_data_flag": int(
            window.get("token_label_association_status") == "insufficient_data"
        ),
        "labels": {
            "run_id": run_id,
            "mode": mode,
            "phase": phase,
            "status": status,
        },
    }


def _build_file_state_metrics(snapshot: MetricsSnapshot) -> list[GaugeMetricFamily]:
    labels = _extract_labels(snapshot.payload or {})
    families = []
    file_present_metric = GaugeMetricFamily(
        FILE_PRESENT_METRIC,
        "Whether a drift latest metrics JSON file is available for export.",
        labels=list(labels.keys()),
    )
    file_present_metric.add_metric(
        list(labels.values()),
        1 if snapshot.file_present else 0,
    )
    families.append(file_present_metric)

    window_file_present_metric = GaugeMetricFamily(
        WINDOW_FILE_PRESENT_METRIC,
        "Whether a drift window metrics JSONL file is available for export.",
        labels=list(labels.keys()),
    )
    window_file_present_metric.add_metric(
        list(labels.values()),
        1 if snapshot.window_file_present else 0,
    )
    families.append(window_file_present_metric)

    if snapshot.metrics_path is not None and snapshot.metrics_path.exists():
        last_update_metric = GaugeMetricFamily(
            LAST_UPDATE_METRIC,
            "Unix timestamp of the latest drift metrics file modification.",
            labels=list(labels.keys()),
        )
        last_update_metric.add_metric(
            list(labels.values()),
            snapshot.metrics_path.stat().st_mtime,
        )
        families.append(last_update_metric)

    return families


def _build_latest_metrics(payload: dict[str, Any]) -> list[GaugeMetricFamily]:
    labels = _latest_window_labels(payload)
    families = []
    for metric_name, metric_help in METRIC_SPECS.items():
        if metric_name == "drift_window_status_code":
            continue
        value = payload.get(metric_name)
        if not isinstance(value, (int, float)):
            continue
        family = GaugeMetricFamily(
            metric_name,
            metric_help,
            labels=list(labels.keys()),
        )
        family.add_metric(list(labels.values()), float(value))
        families.append(family)
    return families


def _build_window_status_metrics(
    latest_payload: dict[str, Any],
    snapshot: MetricsSnapshot,
) -> list[GaugeMetricFamily]:
    family = GaugeMetricFamily(
        "drift_window_status_code",
        "Window status code: 0=ok, 1=warning, 2=critical, 3=insufficient_data.",
        labels=[*WINDOW_LABEL_NAMES, "source"],
    )

    latest_value = latest_payload.get("drift_window_status_code")
    if isinstance(latest_value, (int, float)):
        latest_labels = _latest_window_labels(latest_payload)
        family.add_metric([*latest_labels.values(), "latest"], float(latest_value))

    for window in snapshot.windows or []:
        labels = _window_labels(
            window,
            run_id=_run_id(snapshot.payload),
            mode=_mode(snapshot.payload),
        )
        status = str(window.get("overall_status", "unknown"))
        family.add_metric(
            [*labels.values(), "history"],
            float(STATUS_TO_CODE.get(status, -1)),
        )

    return [family]


def _build_window_history_metrics(
    snapshot: MetricsSnapshot,
) -> list[GaugeMetricFamily]:
    if not snapshot.windows:
        return []

    run_id = _run_id(snapshot.payload)
    mode = _mode(snapshot.payload)
    families = []

    window_info = GaugeMetricFamily(
        "drift_window_info",
        "Window history info sample carrying run, phase, status and window labels.",
        labels=list(WINDOW_LABEL_NAMES),
    )
    metric_status_family = GaugeMetricFamily(
        "drift_window_metric_status_code",
        "Per-window per-metric status code: 0=ok, 1=warning, 2=critical, 3=insufficient_data.",
        labels=[*WINDOW_LABEL_NAMES, "metric"],
    )
    metric_families = {
        metric_name: GaugeMetricFamily(
            metric_name,
            metric_help,
            labels=list(WINDOW_LABEL_NAMES),
        )
        for metric_name, (_, metric_help) in WINDOW_METRIC_SPECS.items()
    }

    for window in snapshot.windows:
        labels = _window_labels(window, run_id=run_id, mode=mode)
        label_values = list(labels.values())
        window_info.add_metric(label_values, 1)

        for metric_name, (field_name, _) in WINDOW_METRIC_SPECS.items():
            value = window.get(field_name)
            if isinstance(value, (int, float)):
                metric_families[metric_name].add_metric(label_values, float(value))

        metric_statuses = window.get("metric_statuses") or {}
        if isinstance(metric_statuses, dict):
            for metric, metric_status in metric_statuses.items():
                metric_status_family.add_metric(
                    [*label_values, str(metric)],
                    float(STATUS_TO_CODE.get(str(metric_status), -1)),
                )

    families.extend(metric_families.values())
    families.extend([metric_status_family, window_info])
    return families


def _extract_labels(payload: dict[str, Any]) -> dict[str, str]:
    raw_labels = payload.get("labels") or {}
    return {
        label_name: str(raw_labels.get(label_name, "unknown"))
        for label_name in LABEL_NAMES
    }


def _window_labels(
    window: dict[str, Any],
    run_id: str,
    mode: str,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "mode": mode,
        "phase": str(window.get("phase", "unknown")),
        "status": str(window.get("overall_status", "unknown")),
        "window_index": str(window.get("window_index", "unknown")),
    }


def _latest_window_labels(payload: dict[str, Any]) -> dict[str, str]:
    labels = _extract_labels(payload)
    return {
        "run_id": labels["run_id"],
        "mode": labels["mode"],
        "phase": labels["phase"],
        "status": labels["status"],
        "window_index": str(payload.get("drift_window_index", "unknown")),
    }


def _run_id(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "unknown"
    return str(payload.get("run_id", "unknown"))


def _mode(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "unknown"
    return str(payload.get("drift_runner_mode", "unknown"))


class _StaticCollector:
    def __init__(self, metrics: list[GaugeMetricFamily]):
        self._metrics = metrics

    def collect(self):
        yield from self._metrics
