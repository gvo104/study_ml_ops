from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUNS_ROOT = Path("drift_v2/artifacts/runs")


@dataclass(frozen=True)
class MonitoringSettings:
    runs_root: Path = DEFAULT_RUNS_ROOT
    explicit_run_id: str | None = None
    explicit_metrics_path: Path | None = None
    explicit_window_metrics_path: Path | None = None
    explicit_recent_events_path: Path | None = None


def load_settings() -> MonitoringSettings:
    explicit_run_id = os.getenv("DRIFT_V2_MONITOR_RUN_ID")
    explicit_metrics_path = os.getenv("DRIFT_V2_PROM_METRICS_PATH")
    explicit_window_path = os.getenv("DRIFT_V2_WINDOW_METRICS_PATH")
    explicit_recent_events_path = os.getenv("DRIFT_V2_RECENT_EVENTS_PATH")
    return MonitoringSettings(
        explicit_run_id=explicit_run_id or None,
        explicit_metrics_path=(
            Path(explicit_metrics_path) if explicit_metrics_path else None
        ),
        explicit_window_metrics_path=(
            Path(explicit_window_path) if explicit_window_path else None
        ),
        explicit_recent_events_path=(
            Path(explicit_recent_events_path) if explicit_recent_events_path else None
        ),
    )
