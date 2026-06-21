from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily

from drift.monitoring.config import MonitoringSettings


METRIC_SPECS = {
    "drift_token_distribution_jsd": "Latest token distribution Jensen-Shannon divergence.",
    "drift_target_distribution_jsd": "Latest target label distribution Jensen-Shannon divergence.",
    "drift_model_expert_disagreement_rate": "Latest disagreement rate between model and expert.",
    "drift_model_expert_macro_f1": "Latest macro F1 between model predictions and expert labels.",
    "drift_token_label_association_drift": "Latest token-label association drift score.",
    "drift_window_status_code": "Latest window status code: 0=ok, 1=warning, 2=critical.",
    "drift_window_index": "Latest evaluated sliding window index.",
    "drift_insufficient_data_flag": "Whether latest association metric lacked enough data.",
}
FILE_PRESENT_METRIC = "drift_metrics_file_present"
LAST_UPDATE_METRIC = "drift_metrics_last_update_timestamp"
LABEL_NAMES = ("run_id", "mode", "phase", "status")


@dataclass(frozen=True)
class MetricsSnapshot:
    payload: dict[str, Any] | None
    metrics_path: Path | None

    @property
    def file_present(self) -> bool:
        return self.payload is not None and self.metrics_path is not None


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


def load_metrics_snapshot(settings: MonitoringSettings) -> MetricsSnapshot:
    metrics_path = resolve_metrics_path(settings)
    if metrics_path is None or not metrics_path.exists():
        return MetricsSnapshot(payload=None, metrics_path=metrics_path)

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return MetricsSnapshot(payload=payload, metrics_path=metrics_path)


def generate_metrics_text(snapshot: MetricsSnapshot) -> bytes:
    registry = CollectorRegistry()
    _register_file_state_metrics(registry, snapshot)
    if snapshot.file_present:
        _register_payload_metrics(registry, snapshot)
    return generate_latest(registry)


def _register_file_state_metrics(
    registry: CollectorRegistry,
    snapshot: MetricsSnapshot,
) -> None:
    labels = _extract_labels(snapshot.payload or {})
    families = []

    file_present_metric = GaugeMetricFamily(
        FILE_PRESENT_METRIC,
        "Whether a drift metrics JSON file is available for export.",
        labels=list(labels.keys()),
    )
    file_present_metric.add_metric(
        list(labels.values()),
        1 if snapshot.file_present else 0,
    )
    families.append(file_present_metric)

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

    registry.register(_StaticCollector(families))


def _register_payload_metrics(
    registry: CollectorRegistry,
    snapshot: MetricsSnapshot,
) -> None:
    assert snapshot.payload is not None
    labels = _extract_labels(snapshot.payload)
    families = []
    for metric_name, metric_help in METRIC_SPECS.items():
        value = snapshot.payload.get(metric_name)
        if not isinstance(value, (int, float)):
            continue
        family = GaugeMetricFamily(
            metric_name,
            metric_help,
            labels=list(labels.keys()),
        )
        family.add_metric(list(labels.values()), float(value))
        families.append(family)

    if families:
        registry.register(_StaticCollector(families))


def _extract_labels(payload: dict[str, Any]) -> dict[str, str]:
    raw_labels = payload.get("labels") or {}
    return {
        label_name: str(raw_labels.get(label_name, "unknown"))
        for label_name in LABEL_NAMES
    }


class _StaticCollector:
    def __init__(self, metrics: list[GaugeMetricFamily]):
        self._metrics = metrics

    def collect(self):
        yield from self._metrics
