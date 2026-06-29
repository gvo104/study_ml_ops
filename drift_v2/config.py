from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drift_v2.runner.config import RunnerConfig, load_runner_config


DEFAULT_DB_PATH = Path("drift_v2/artifacts/demo/events.sqlite")
DEFAULT_OUTPUT_ROOT = Path("drift_v2/artifacts/runs")
DEFAULT_RUN_ID = "offline_demo_live_v2"
DEFAULT_ONLINE_RUN_ID = "online_demo_live_v2"
DEFAULT_EVENT_SPAN = 40
DEFAULT_WINDOW_SIZE = 20
DEFAULT_STEP_SIZE = 20
DEFAULT_ONLINE_SOURCE_MODE = "online_synthetic_v2"


@dataclass(frozen=True)
class DriftV2Config:
    runner: RunnerConfig
    db_path: Path
    output_root: Path
    run_id: str
    pipeline: str
    source_mode: str | None
    event_span: int
    window_size: int
    step_size: int


def load_drift_v2_config(
    config_path: Path | str = "drift_v2/configs/runner.yaml",
    mode: str = "debug",
    db_path: Path | str = DEFAULT_DB_PATH,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    pipeline: str = "offline",
    source_mode: str | None = None,
    event_span: int = DEFAULT_EVENT_SPAN,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> DriftV2Config:
    runner = load_runner_config(config_path, mode)
    return DriftV2Config(
        runner=runner,
        db_path=Path(db_path),
        output_root=Path(output_root),
        run_id=run_id,
        pipeline=pipeline,
        source_mode=source_mode,
        event_span=event_span,
        window_size=window_size,
        step_size=step_size,
    )
