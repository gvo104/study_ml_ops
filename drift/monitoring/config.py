from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUNS_ROOT = Path("drift/artifacts/runs")


@dataclass(frozen=True)
class MonitoringSettings:
    runs_root: Path = DEFAULT_RUNS_ROOT
    explicit_metrics_path: Path | None = None


def load_settings() -> MonitoringSettings:
    explicit_path = os.getenv("DRIFT_PROM_METRICS_PATH")
    return MonitoringSettings(
        explicit_metrics_path=Path(explicit_path) if explicit_path else None,
    )
