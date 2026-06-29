from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoEvent:
    event_index: int
    event_id: str
    created_at: str
    source_mode: str
    raw_text: str
    tokens_stemmed: str
    num_of_characters: int
    num_of_sentences: int
    model_prediction: str
    model_confidence: float | None
    expert_label: str | None
    expert_confidence: float | None
    hidden_phase: str | None
    hidden_target_label: str | None


@dataclass(frozen=True)
class WindowBatch:
    window_index: int
    start_event_index: int
    end_event_index: int
    events: tuple[DemoEvent, ...]

    @property
    def size(self) -> int:
        return len(self.events)


@dataclass(frozen=True)
class WindowPlan:
    total_events: int
    visible_events: int
    window_size: int
    step_size: int
    total_complete_windows: int


@dataclass(frozen=True)
class PlaybackState:
    visible_events: int
    total_events: int
    batch_size: int
    cycles_completed: int


@dataclass(frozen=True)
class RecentEventPoint:
    event_index: int
    event_id: str
    created_at: str
    model_prediction: str
    model_confidence: float | None
    expert_label: str
    expert_confidence: float | None


@dataclass(frozen=True)
class WindowMetricRecord:
    window_index: int
    start_event_index: int
    end_event_index: int
    phase: str
    token_distribution_jsd: float | None
    model_prediction_distribution_jsd: float | None
    target_distribution_jsd: float | None
    model_expert_disagreement_rate: float | None
    model_expert_macro_f1: float | None
    overall_status: str
    metric_statuses: dict[str, str]


@dataclass(frozen=True)
class ArtifactBundle:
    run_id: str
    mode: str
    pipeline: str
    visible_event_count: int
    total_event_count: int
    complete_window_count: int
    threshold_calibration_status: str
    recent_events: list[dict[str, object]]
    window_metrics: list[dict[str, object]]
    latest_payload: dict[str, object]
    summary_payload: dict[str, object]
