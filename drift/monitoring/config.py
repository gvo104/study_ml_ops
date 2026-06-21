from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import monotonic


DEFAULT_RUNS_ROOT = Path("drift/artifacts/runs")


@dataclass(frozen=True)
class MonitoringSettings:
    runs_root: Path = DEFAULT_RUNS_ROOT
    explicit_metrics_path: Path | None = None
    explicit_window_metrics_path: Path | None = None
    replay_enabled: bool = False
    replay_step_seconds: int = 15
    replay_loop: bool = True
    replay_started_at: float = 0.0


def load_settings() -> MonitoringSettings:
    explicit_path = os.getenv("DRIFT_PROM_METRICS_PATH")
    explicit_window_path = os.getenv("DRIFT_WINDOW_METRICS_PATH")
    return MonitoringSettings(
        explicit_metrics_path=Path(explicit_path) if explicit_path else None,
        explicit_window_metrics_path=(
            Path(explicit_window_path) if explicit_window_path else None
        ),
        replay_enabled=_env_bool("DRIFT_REPLAY_ENABLED", default=False),
        replay_step_seconds=_env_int("DRIFT_REPLAY_STEP_SECONDS", default=15),
        replay_loop=_env_bool("DRIFT_REPLAY_LOOP", default=True),
        replay_started_at=monotonic(),
    )


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default
