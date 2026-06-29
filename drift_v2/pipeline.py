from __future__ import annotations

from dataclasses import asdict

from drift_v2.artifacts import write_artifact_bundle
from drift_v2.config import DriftV2Config
from drift_v2.contracts import ArtifactBundle
from drift_v2.metrics import (
    build_recent_event_points,
    compute_window_metric_records,
    latest_metrics_payload,
    summary_payload,
)
from drift_v2.online import generate_online_events, label_online_events
from drift_v2.repository import DemoEventRepository
from drift_v2.windows import build_prefix_windows, build_window_plan


def initialize_offline_replay_artifacts(
    config: DriftV2Config,
    total_events: int,
) -> ArtifactBundle:
    return initialize_artifacts(config=config, total_events=total_events, visible_events=0)


def initialize_artifacts(
    config: DriftV2Config,
    total_events: int,
    visible_events: int,
) -> ArtifactBundle:
    latest_payload = latest_metrics_payload(
        run_id=config.run_id,
        mode=config.runner.mode,
        pipeline=config.pipeline,
        visible_event_count=visible_events,
        total_event_count=total_events,
        window_records=[],
        threshold_calibration_status="insufficient_data",
    )
    summary = summary_payload(
        run_id=config.run_id,
        mode=config.runner.mode,
        pipeline=config.pipeline,
        visible_event_count=visible_events,
        total_event_count=total_events,
        window_records=[],
        threshold_calibration_status="insufficient_data",
    )
    bundle = ArtifactBundle(
        run_id=config.run_id,
        mode=config.runner.mode,
        pipeline=config.pipeline,
        visible_event_count=visible_events,
        total_event_count=total_events,
        complete_window_count=0,
        threshold_calibration_status="insufficient_data",
        recent_events=[],
        window_metrics=[],
        latest_payload=latest_payload,
        summary_payload=summary,
    )
    write_artifact_bundle(config.output_root, bundle)
    return bundle


def run_offline_replay_tick(
    config: DriftV2Config,
    visible_events: int,
) -> ArtifactBundle:
    repository = DemoEventRepository(config.db_path)
    events = repository.load_all_events()
    plan = build_window_plan(
        events=events,
        visible_events=visible_events,
        window_size=config.window_size,
        step_size=config.step_size,
    )
    windows = build_prefix_windows(
        events=events,
        visible_events=plan.visible_events,
        window_size=config.window_size,
        step_size=config.step_size,
    )
    window_records, calibration_status = compute_window_metric_records(windows, config)
    recent_events = [
        asdict(point)
        for point in build_recent_event_points(events[: plan.visible_events], config.event_span)
    ]
    latest_payload = latest_metrics_payload(
        run_id=config.run_id,
        mode=config.runner.mode,
        pipeline=config.pipeline,
        visible_event_count=plan.visible_events,
        total_event_count=plan.total_events,
        window_records=window_records,
        threshold_calibration_status=calibration_status,
    )
    summary = summary_payload(
        run_id=config.run_id,
        mode=config.runner.mode,
        pipeline=config.pipeline,
        visible_event_count=plan.visible_events,
        total_event_count=plan.total_events,
        window_records=window_records,
        threshold_calibration_status=calibration_status,
    )
    bundle = ArtifactBundle(
        run_id=config.run_id,
        mode=config.runner.mode,
        pipeline=config.pipeline,
        visible_event_count=plan.visible_events,
        total_event_count=plan.total_events,
        complete_window_count=plan.total_complete_windows,
        threshold_calibration_status=calibration_status,
        recent_events=recent_events,
        window_metrics=[asdict(record) for record in window_records],
        latest_payload=latest_payload,
        summary_payload=summary,
    )
    write_artifact_bundle(config.output_root, bundle)
    return bundle


def run_online_tick(
    config: DriftV2Config,
    synthetic_client,
    predict_client,
    hidden_phase: str,
    traffic_count: int,
    expert_batch_size: int,
    lookback_minutes: int,
) -> tuple[ArtifactBundle, dict[str, int], dict[str, int]]:
    if not config.source_mode:
        raise ValueError("online drift_v2 config requires source_mode")

    traffic = generate_online_events(
        db_path=config.db_path,
        synthetic_client=synthetic_client,
        predict_client=predict_client,
        source_mode=config.source_mode,
        hidden_phase=hidden_phase,
        count=traffic_count,
        allowed_labels=config.runner.expert_allowed_labels,
    )
    expert = {"selected": 0, "updated": 0, "failed": 0, "total_events": traffic["total_events"]}
    if expert_batch_size > 0:
        expert = label_online_events(
            db_path=config.db_path,
            synthetic_client=synthetic_client,
            source_mode=config.source_mode,
            allowed_labels=config.runner.expert_allowed_labels,
            batch_size=expert_batch_size,
            lookback_minutes=lookback_minutes,
        )

    repository = DemoEventRepository(config.db_path)
    events = repository.load_source_events(config.source_mode)
    bundle = _build_bundle_from_events(config=config, events=events, visible_events=len(events))
    return bundle, traffic, expert


def _build_bundle_from_events(
    config: DriftV2Config,
    events,
    visible_events: int,
) -> ArtifactBundle:
    plan = build_window_plan(
        events=events,
        visible_events=visible_events,
        window_size=config.window_size,
        step_size=config.step_size,
    )
    windows = build_prefix_windows(
        events=events,
        visible_events=plan.visible_events,
        window_size=config.window_size,
        step_size=config.step_size,
    )
    window_records, calibration_status = compute_window_metric_records(windows, config)
    recent_events = [
        asdict(point)
        for point in build_recent_event_points(events[: plan.visible_events], config.event_span)
    ]
    latest_payload = latest_metrics_payload(
        run_id=config.run_id,
        mode=config.runner.mode,
        pipeline=config.pipeline,
        visible_event_count=plan.visible_events,
        total_event_count=plan.total_events,
        window_records=window_records,
        threshold_calibration_status=calibration_status,
    )
    summary = summary_payload(
        run_id=config.run_id,
        mode=config.runner.mode,
        pipeline=config.pipeline,
        visible_event_count=plan.visible_events,
        total_event_count=plan.total_events,
        window_records=window_records,
        threshold_calibration_status=calibration_status,
    )
    bundle = ArtifactBundle(
        run_id=config.run_id,
        mode=config.runner.mode,
        pipeline=config.pipeline,
        visible_event_count=plan.visible_events,
        total_event_count=plan.total_events,
        complete_window_count=plan.total_complete_windows,
        threshold_calibration_status=calibration_status,
        recent_events=recent_events,
        window_metrics=[asdict(record) for record in window_records],
        latest_payload=latest_payload,
        summary_payload=summary,
    )
    write_artifact_bundle(config.output_root, bundle)
    return bundle
