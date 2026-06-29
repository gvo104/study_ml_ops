"""Clean drift demo pipeline built alongside legacy drift modules."""

from drift_v2.contracts import DemoEvent, PlaybackState, WindowBatch, WindowPlan
from drift_v2.playback import advance_playback_state, initial_playback_state
from drift_v2.repository import DemoEventRepository
from drift_v2.windows import build_prefix_windows, build_window_plan

__all__ = [
    "DemoEvent",
    "DemoEventRepository",
    "DriftV2Config",
    "PlaybackState",
    "WindowBatch",
    "WindowPlan",
    "advance_playback_state",
    "build_prefix_windows",
    "build_window_plan",
    "initial_playback_state",
    "load_drift_v2_config",
    "run_offline_replay_tick",
]


def __getattr__(name: str):
    if name in {"DriftV2Config", "load_drift_v2_config"}:
        from drift_v2.config import DriftV2Config, load_drift_v2_config

        return {
            "DriftV2Config": DriftV2Config,
            "load_drift_v2_config": load_drift_v2_config,
        }[name]
    if name == "run_offline_replay_tick":
        from drift_v2.pipeline import run_offline_replay_tick

        return run_offline_replay_tick
    raise AttributeError(f"module 'drift_v2' has no attribute {name!r}")
