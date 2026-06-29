from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily

from drift_v2.monitoring.config import MonitoringSettings


STATUS_TO_CODE = {
    "ok": 0,
    "warning": 1,
    "critical": 2,
    "insufficient_data": 3,
    "unknown": -1,
}
LATEST_METRIC_SPECS = {
    "drift_v2_visible_event_count": "Current visible event count in the clean drift_v2 run.",
    "drift_v2_total_event_count": "Total event count visible to the clean drift_v2 run.",
    "drift_v2_complete_window_count": "Current number of complete windows.",
    "drift_v2_token_distribution_jsd": "Latest complete window token distribution drift.",
    "drift_v2_model_prediction_distribution_jsd": "Latest complete window model prediction distribution drift.",
    "drift_v2_target_distribution_jsd": "Latest complete window expert label distribution drift.",
    "drift_v2_model_expert_disagreement_rate": "Latest complete window disagreement rate.",
    "drift_v2_model_expert_macro_f1": "Latest complete window macro F1.",
    "drift_v2_window_status_code": "Latest complete window status code.",
    "drift_v2_window_index": "Latest complete window index.",
}
WINDOW_METRIC_SPECS = {
    "drift_v2_window_token_distribution_jsd": (
        "token_distribution_jsd",
        "Per-window token distribution drift.",
    ),
    "drift_v2_window_model_prediction_distribution_jsd": (
        "model_prediction_distribution_jsd",
        "Per-window model prediction distribution drift.",
    ),
    "drift_v2_window_target_distribution_jsd": (
        "target_distribution_jsd",
        "Per-window expert label distribution drift.",
    ),
    "drift_v2_window_model_expert_disagreement_rate": (
        "model_expert_disagreement_rate",
        "Per-window disagreement rate.",
    ),
    "drift_v2_window_model_expert_macro_f1": (
        "model_expert_macro_f1",
        "Per-window macro F1.",
    ),
}
WINDOW_LABEL_NAMES = ("run_id", "mode", "status", "pipeline", "window_index")
RUN_LABEL_NAMES = ("run_id", "mode", "status", "pipeline")


@dataclass(frozen=True)
class MetricsSnapshot:
    active: bool
    latest_payload: dict[str, Any] | None
    metrics_path: Path | None
    window_metrics_path: Path | None
    recent_events_path: Path | None
    windows: list[dict[str, Any]]
    recent_events: list[dict[str, Any]]
    parse_errors: list[str]

    @property
    def file_present(self) -> bool:
        return self.active and self.latest_payload is not None and self.metrics_path is not None

    @property
    def window_count(self) -> int:
        return len(self.windows)


def resolve_metrics_path(settings: MonitoringSettings) -> Path | None:
    paths = resolve_metrics_paths(settings)
    return paths[-1] if paths else None


def resolve_metrics_paths(settings: MonitoringSettings) -> list[Path]:
    if settings.explicit_metrics_path is not None:
        return [settings.explicit_metrics_path]
    if not settings.runs_root.exists():
        return []
    candidates = sorted(
        (
            child / "prom_metrics_latest.json"
            for child in settings.runs_root.iterdir()
            if child.is_dir()
            and (child / "prom_metrics_latest.json").exists()
            and (
                settings.explicit_run_id is None
                or child.name.startswith(f"{settings.explicit_run_id}_")
            )
        ),
        key=lambda path: path.parent.name,
    )
    return candidates


def resolve_window_metrics_path(
    settings: MonitoringSettings,
    metrics_path: Path | None,
) -> Path | None:
    if settings.explicit_window_metrics_path is not None:
        return settings.explicit_window_metrics_path
    if metrics_path is None:
        return None
    return metrics_path.parent / "window_metrics.jsonl"


def resolve_recent_events_path(
    settings: MonitoringSettings,
    metrics_path: Path | None,
) -> Path | None:
    if settings.explicit_recent_events_path is not None:
        return settings.explicit_recent_events_path
    if metrics_path is None:
        return None
    return metrics_path.parent / "recent_events.json"


def load_metrics_snapshot(settings: MonitoringSettings) -> MetricsSnapshot:
    paths = resolve_metrics_paths(settings)
    if not paths:
        return MetricsSnapshot(
            active=False,
            latest_payload=None,
            metrics_path=None,
            window_metrics_path=None,
            recent_events_path=None,
            windows=[],
            recent_events=[],
            parse_errors=[],
        )
    return _load_snapshot(paths[-1], settings)


def load_metrics_snapshots(settings: MonitoringSettings) -> list[MetricsSnapshot]:
    return [_load_snapshot(path, settings) for path in resolve_metrics_paths(settings)]


def _load_snapshot(
    metrics_path: Path,
    settings: MonitoringSettings,
) -> MetricsSnapshot:
    parse_errors: list[str] = []
    active = _is_active_run(metrics_path)
    latest_payload = _load_json_object(metrics_path, "latest metrics", parse_errors)
    if not active:
        return MetricsSnapshot(
            active=False,
            latest_payload=_inactive_payload(metrics_path, latest_payload),
            metrics_path=metrics_path,
            window_metrics_path=None,
            recent_events_path=None,
            windows=[],
            recent_events=[],
            parse_errors=parse_errors,
        )
    window_metrics_path = resolve_window_metrics_path(settings, metrics_path)
    recent_events_path = resolve_recent_events_path(settings, metrics_path)
    windows = _load_window_metrics(window_metrics_path, parse_errors)
    recent_events = _load_recent_events(recent_events_path, parse_errors)
    return MetricsSnapshot(
        active=True,
        latest_payload=latest_payload,
        metrics_path=metrics_path,
        window_metrics_path=window_metrics_path,
        recent_events_path=recent_events_path,
        windows=windows,
        recent_events=recent_events,
        parse_errors=parse_errors,
    )


def generate_metrics_text(snapshot_or_snapshots: MetricsSnapshot | list[MetricsSnapshot]) -> bytes:
    snapshots = (
        snapshot_or_snapshots
        if isinstance(snapshot_or_snapshots, list)
        else [snapshot_or_snapshots]
    )
    registry = CollectorRegistry()
    metrics = []
    metrics.extend(_build_file_metrics(snapshots))
    metrics.extend(_build_latest_metrics(snapshots))
    metrics.extend(_build_window_history_metrics(snapshots))
    if metrics:
        registry.register(_StaticCollector(metrics))
    return generate_latest(registry)


def _build_file_metrics(snapshots: list[MetricsSnapshot]) -> list[GaugeMetricFamily]:
    file_present = GaugeMetricFamily(
        "drift_v2_metrics_file_present",
        "Whether clean drift_v2 metrics are available.",
        labels=list(RUN_LABEL_NAMES),
    )
    freshness = GaugeMetricFamily(
        "drift_v2_metrics_last_update_timestamp",
        "Unix timestamp of latest clean drift_v2 metrics update.",
        labels=list(RUN_LABEL_NAMES),
    )
    has_freshness = False
    for snapshot in snapshots:
        labels = _extract_latest_labels(snapshot.latest_payload or {})
        file_present.add_metric(list(labels.values()), 1 if snapshot.file_present else 0)
        if snapshot.metrics_path is not None and snapshot.metrics_path.exists():
            has_freshness = True
            freshness.add_metric(
                list(labels.values()),
                snapshot.metrics_path.stat().st_mtime,
            )
    metrics = [file_present]
    if has_freshness:
        metrics.append(freshness)
    return metrics


def _build_latest_metrics(snapshots: list[MetricsSnapshot]) -> list[GaugeMetricFamily]:
    metrics = []
    for metric_name, help_text in LATEST_METRIC_SPECS.items():
        family = GaugeMetricFamily(metric_name, help_text, labels=list(WINDOW_LABEL_NAMES))
        for snapshot in snapshots:
            if not snapshot.file_present:
                continue
            payload = snapshot.latest_payload or {}
            value = payload.get(metric_name)
            if not isinstance(value, (int, float)):
                continue
            labels = _latest_labels(payload)
            family.add_metric(list(labels.values()), float(value))
        if family.samples:
            metrics.append(family)
    return metrics


def _build_window_history_metrics(snapshots: list[MetricsSnapshot]) -> list[GaugeMetricFamily]:
    window_info = GaugeMetricFamily(
        "drift_v2_window_info",
        "Window info sample for clean drift_v2 pipeline.",
        labels=list(WINDOW_LABEL_NAMES),
    )
    status_family = GaugeMetricFamily(
        "drift_v2_window_status_code",
        "Clean drift_v2 window status code.",
        labels=[*WINDOW_LABEL_NAMES, "source"],
    )
    metric_status_family = GaugeMetricFamily(
        "drift_v2_window_metric_status_code",
        "Per-window per-metric clean drift_v2 status code.",
        labels=[*WINDOW_LABEL_NAMES, "phase", "metric"],
    )
    metric_families = {
        metric_name: GaugeMetricFamily(metric_name, help_text, labels=list(WINDOW_LABEL_NAMES))
        for metric_name, (_, help_text) in WINDOW_METRIC_SPECS.items()
    }
    for snapshot in snapshots:
        if not snapshot.file_present or not snapshot.windows:
            continue
        latest_payload = snapshot.latest_payload or {}
        latest_status = latest_payload.get("labels", {}).get("status", "unknown")
        latest_index = latest_payload.get("drift_v2_window_index")
        latest_pipeline = _pipeline(latest_payload)
        if isinstance(latest_index, (int, float)):
            labels = {
                "run_id": _run_id(latest_payload),
                "mode": _mode(latest_payload),
                "status": str(latest_status),
                "pipeline": latest_pipeline,
                "window_index": str(latest_index),
            }
            status_family.add_metric(
                [*labels.values(), "latest"],
                float(STATUS_TO_CODE.get(str(latest_status), -1)),
            )

        for window in snapshot.windows:
            labels = {
                "run_id": _run_id(latest_payload),
                "mode": _mode(latest_payload),
                "status": str(window.get("overall_status", "unknown")),
                "pipeline": latest_pipeline,
                "window_index": str(window.get("window_index", "unknown")),
            }
            label_values = list(labels.values())
            window_info.add_metric(label_values, 1)
            status_family.add_metric(
                [*label_values, "history"],
                float(STATUS_TO_CODE.get(labels["status"], -1)),
            )
            for metric_name, (field_name, _) in WINDOW_METRIC_SPECS.items():
                value = window.get(field_name)
                if isinstance(value, (int, float)):
                    metric_families[metric_name].add_metric(label_values, float(value))
            metric_statuses = window.get("metric_statuses") or {}
            if isinstance(metric_statuses, dict):
                phase = str(window.get("phase", "unknown"))
                for metric_name, status in metric_statuses.items():
                    metric_status_family.add_metric(
                        [*label_values, phase, str(metric_name)],
                        float(STATUS_TO_CODE.get(str(status), -1)),
                    )
    metrics = []
    for family in metric_families.values():
        if family.samples:
            metrics.append(family)
    if status_family.samples:
        metrics.append(status_family)
    if metric_status_family.samples:
        metrics.append(metric_status_family)
    if window_info.samples:
        metrics.append(window_info)
    return metrics


def _load_json_object(
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


def _is_active_run(metrics_path: Path | None) -> bool:
    if metrics_path is None:
        return False
    return (metrics_path.parent / ".active").exists()


def _inactive_payload(
    metrics_path: Path | None,
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if payload is not None:
        labels = payload.get("labels") or {}
        return {
            "run_id": _run_id(payload),
            "drift_v2_runner_mode": _mode(payload),
            "drift_v2_pipeline_mode": _pipeline(payload),
            "labels": {
                "run_id": str(labels.get("run_id", _run_id(payload))),
                "mode": str(labels.get("mode", _mode(payload))),
                "status": "inactive",
                "pipeline": str(labels.get("pipeline", _pipeline(payload))),
            },
        }
    if metrics_path is None:
        return None
    directory_name = metrics_path.parent.name
    if "_" not in directory_name:
        return {
            "run_id": "unknown",
            "drift_v2_runner_mode": "unknown",
            "drift_v2_pipeline_mode": "unknown",
            "labels": {
                "run_id": "unknown",
                "mode": "unknown",
                "status": "inactive",
                "pipeline": "unknown",
            },
        }
    run_id, mode = directory_name.rsplit("_", 1)
    return {
        "run_id": run_id,
        "drift_v2_runner_mode": mode,
        "drift_v2_pipeline_mode": "unknown",
        "labels": {
            "run_id": run_id,
            "mode": mode,
            "status": "inactive",
            "pipeline": "unknown",
        },
    }


def _load_window_metrics(path: Path | None, parse_errors: list[str]) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    windows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"window metrics: {path}:{line_number}: {exc}")
            continue
        if isinstance(payload, dict):
            windows.append(payload)
    return windows


def _load_recent_events(path: Path | None, parse_errors: list[str]) -> list[dict[str, Any]]:
    payload = _load_json_object(path, "recent events", parse_errors)
    if payload is None:
        return []
    events = payload.get("events")
    if not isinstance(events, list):
        parse_errors.append(f"recent events: {path}: expected 'events' list")
        return []
    return [event for event in events if isinstance(event, dict)]


def _extract_latest_labels(payload: dict[str, Any]) -> dict[str, str]:
    raw = payload.get("labels") or {}
    return {name: str(raw.get(name, "unknown")) for name in RUN_LABEL_NAMES}


def _latest_labels(payload: dict[str, Any]) -> dict[str, str]:
    labels = _extract_latest_labels(payload)
    return {
        "run_id": labels["run_id"],
        "mode": labels["mode"],
        "status": labels["status"],
        "pipeline": labels["pipeline"],
        "window_index": str(payload.get("drift_v2_window_index", "unknown")),
    }


def _run_id(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "unknown"
    return str(payload.get("run_id", "unknown"))


def _mode(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "unknown"
    return str(payload.get("drift_v2_runner_mode", "unknown"))


def _pipeline(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "unknown"
    labels = payload.get("labels") or {}
    if "pipeline" in labels:
        return str(labels["pipeline"])
    return str(payload.get("drift_v2_pipeline_mode", "unknown"))


class _StaticCollector:
    def __init__(self, metrics: list[GaugeMetricFamily]):
        self._metrics = metrics

    def collect(self):
        yield from self._metrics
