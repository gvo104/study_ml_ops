from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from drift_v2.runner.metrics import (
    compute_disagreement_rate,
    compute_macro_f1,
    compute_model_prediction_distribution_jsd,
    compute_pre_label_window_metrics,
    compute_target_distribution_jsd,
)
from drift_v2.runner.reference import ensure_reference_snapshot
from drift_v2.config import DriftV2Config
from drift_v2.contracts import RecentEventPoint, WindowBatch, WindowMetricRecord


BASELINE_PHASES = {"normal", "a_baseline", "a", "baseline"}
CORE_METRIC_DIRECTIONS = {
    "token_distribution_jsd": "higher_is_worse",
    "model_prediction_distribution_jsd": "higher_is_worse",
    "target_distribution_jsd": "higher_is_worse",
    "model_expert_disagreement_rate": "higher_is_worse",
    "model_expert_macro_f1": "lower_is_worse",
}
WARNING_MARGIN = 0.05
CRITICAL_MARGIN = 0.15


def build_recent_event_points(
    events,
    limit: int,
) -> list[RecentEventPoint]:
    recent_events = events[-limit:]
    return [
        RecentEventPoint(
            event_index=event.event_index,
            event_id=event.event_id,
            created_at=event.created_at,
            model_prediction=event.model_prediction,
            model_confidence=event.model_confidence,
            expert_label=event.expert_label or "unlabeled",
            expert_confidence=event.expert_confidence,
        )
        for event in recent_events
    ]


def compute_window_metric_records(
    windows: list[WindowBatch],
    config: DriftV2Config,
) -> tuple[list[WindowMetricRecord], str]:
    _, reference_stats = ensure_reference_snapshot(config.runner)
    draft_records = [
        _compute_window_metrics(window, reference_stats, config)
        for window in windows
    ]
    baseline_records = [
        record for record in draft_records if record["phase"].lower() in BASELINE_PHASES
    ]
    thresholds = _compute_thresholds(baseline_records)
    calibration_status = "ok" if baseline_records else "insufficient_baseline"

    final_records: list[WindowMetricRecord] = []
    for record in draft_records:
        status_payload = _classify_status(record, thresholds, calibration_status)
        final_records.append(
            WindowMetricRecord(
                window_index=record["window_index"],
                start_event_index=record["start_event_index"],
                end_event_index=record["end_event_index"],
                phase=record["phase"],
                token_distribution_jsd=record["token_distribution_jsd"],
                model_prediction_distribution_jsd=record["model_prediction_distribution_jsd"],
                target_distribution_jsd=record["target_distribution_jsd"],
                model_expert_disagreement_rate=record["model_expert_disagreement_rate"],
                model_expert_macro_f1=record["model_expert_macro_f1"],
                overall_status=status_payload["overall_status"],
                metric_statuses=status_payload["metric_statuses"],
            )
        )
    return final_records, calibration_status


def latest_metrics_payload(
    run_id: str,
    mode: str,
    pipeline: str,
    visible_event_count: int,
    total_event_count: int,
    window_records: list[WindowMetricRecord],
    threshold_calibration_status: str,
) -> dict[str, Any]:
    latest = window_records[-1] if window_records else None
    if latest is None:
        return {
            "run_id": run_id,
            "drift_v2_runner_mode": mode,
            "drift_v2_pipeline_mode": pipeline,
            "drift_v2_visible_event_count": visible_event_count,
            "drift_v2_total_event_count": total_event_count,
            "drift_v2_complete_window_count": 0,
            "drift_v2_window_status_code": 3,
            "drift_v2_window_index": None,
            "drift_v2_threshold_calibration_status": threshold_calibration_status,
            "labels": {
                "run_id": run_id,
                "mode": mode,
                "status": "insufficient_data",
                "pipeline": pipeline,
            },
        }

    return {
        "run_id": run_id,
        "drift_v2_runner_mode": mode,
        "drift_v2_pipeline_mode": pipeline,
        "drift_v2_visible_event_count": visible_event_count,
        "drift_v2_total_event_count": total_event_count,
        "drift_v2_complete_window_count": len(window_records),
        "drift_v2_token_distribution_jsd": latest.token_distribution_jsd,
        "drift_v2_model_prediction_distribution_jsd": latest.model_prediction_distribution_jsd,
        "drift_v2_target_distribution_jsd": latest.target_distribution_jsd,
        "drift_v2_model_expert_disagreement_rate": latest.model_expert_disagreement_rate,
        "drift_v2_model_expert_macro_f1": latest.model_expert_macro_f1,
        "drift_v2_window_status_code": _status_code(latest.overall_status),
        "drift_v2_window_index": latest.window_index,
        "drift_v2_threshold_calibration_status": threshold_calibration_status,
        "labels": {
            "run_id": run_id,
            "mode": mode,
            "status": latest.overall_status,
            "pipeline": pipeline,
        },
    }


def summary_payload(
    run_id: str,
    mode: str,
    pipeline: str,
    visible_event_count: int,
    total_event_count: int,
    window_records: list[WindowMetricRecord],
    threshold_calibration_status: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "pipeline": pipeline,
        "visible_event_count": visible_event_count,
        "total_event_count": total_event_count,
        "complete_window_count": len(window_records),
        "threshold_calibration_status": threshold_calibration_status,
        "latest_window": asdict(window_records[-1]) if window_records else None,
    }


def _compute_window_metrics(
    window: WindowBatch,
    reference_stats: dict[str, Any],
    config: DriftV2Config,
) -> dict[str, Any]:
    all_df = pd.DataFrame(
        {
            "statement": event.raw_text,
            "tokens_stemmed": event.tokens_stemmed,
            "num_of_characters": event.num_of_characters,
            "num_of_sentences": event.num_of_sentences,
            "model_prediction": event.model_prediction,
            "model_confidence": event.model_confidence,
            "expert_label": event.expert_label,
            "expert_confidence": event.expert_confidence,
        }
        for event in window.events
    )
    labeled_df = all_df[all_df["expert_label"].notna()].copy()
    metrics = compute_pre_label_window_metrics(
        all_df,
        reference_stats=reference_stats,
        top_k_tokens=config.runner.top_k_tokens,
    )
    if labeled_df.empty:
        metrics.update(
            {
                "target_distribution_jsd": None,
                "model_expert_disagreement_rate": None,
                "model_expert_macro_f1": None,
            }
        )
    else:
        metrics.update(
            {
                "target_distribution_jsd": compute_target_distribution_jsd(
                    labeled_df,
                    reference_stats["class_distribution"],
                ),
                "model_expert_disagreement_rate": compute_disagreement_rate(labeled_df),
                "model_expert_macro_f1": compute_macro_f1(labeled_df),
            }
        )

    phase = _window_phase(window)
    return {
        "window_index": window.window_index,
        "start_event_index": window.start_event_index,
        "end_event_index": window.end_event_index,
        "phase": phase,
        "token_distribution_jsd": metrics["token_distribution_jsd"],
        "model_prediction_distribution_jsd": metrics["model_prediction_distribution_jsd"],
        "target_distribution_jsd": metrics["target_distribution_jsd"],
        "model_expert_disagreement_rate": metrics["model_expert_disagreement_rate"],
        "model_expert_macro_f1": metrics["model_expert_macro_f1"],
    }


def _window_phase(window: WindowBatch) -> str:
    phases = sorted({event.hidden_phase or "unknown" for event in window.events})
    if len(phases) == 1:
        return phases[0]
    return "mixed"


def _compute_thresholds(
    baseline_records: list[dict[str, Any]],
) -> dict[str, dict[str, float | str | None]]:
    thresholds: dict[str, dict[str, float | str | None]] = {}
    for metric_name, direction in CORE_METRIC_DIRECTIONS.items():
        values = [
            float(record[metric_name])
            for record in baseline_records
            if record.get(metric_name) is not None
        ]
        if not values:
            thresholds[metric_name] = {
                "warning": None,
                "critical": None,
                "direction": direction,
            }
            continue

        if direction == "lower_is_worse":
            baseline = min(values)
            warning = baseline - max(abs(baseline) * WARNING_MARGIN, 1e-6)
            critical = baseline - max(abs(baseline) * CRITICAL_MARGIN, 1e-6)
        else:
            baseline = max(values)
            warning = baseline + max(abs(baseline) * WARNING_MARGIN, 1e-6)
            critical = baseline + max(abs(baseline) * CRITICAL_MARGIN, 1e-6)
        thresholds[metric_name] = {
            "warning": float(warning),
            "critical": float(critical),
            "direction": direction,
        }
    return thresholds


def _classify_status(
    record: dict[str, Any],
    thresholds: dict[str, dict[str, float | str | None]],
    calibration_status: str,
) -> dict[str, Any]:
    metric_statuses: dict[str, str] = {}
    warning_hits = 0
    critical_hits = 0

    if calibration_status != "ok":
        for metric_name in CORE_METRIC_DIRECTIONS:
            metric_statuses[metric_name] = "unavailable"
        return {
            "metric_statuses": metric_statuses,
            "overall_status": "insufficient_data",
        }

    for metric_name, threshold in thresholds.items():
        value = record.get(metric_name)
        if value is None or threshold["warning"] is None:
            metric_statuses[metric_name] = "unavailable"
            continue
        if threshold["direction"] == "lower_is_worse":
            if value < threshold["critical"]:
                metric_statuses[metric_name] = "critical"
                critical_hits += 1
            elif value < threshold["warning"]:
                metric_statuses[metric_name] = "warning"
                warning_hits += 1
            else:
                metric_statuses[metric_name] = "ok"
            continue

        if value > threshold["critical"]:
            metric_statuses[metric_name] = "critical"
            critical_hits += 1
        elif value > threshold["warning"]:
            metric_statuses[metric_name] = "warning"
            warning_hits += 1
        else:
            metric_statuses[metric_name] = "ok"

    if critical_hits >= 1 or warning_hits >= 2:
        overall_status = "critical"
    elif warning_hits >= 1:
        overall_status = "warning"
    else:
        overall_status = "ok"
    return {"metric_statuses": metric_statuses, "overall_status": overall_status}


def _status_code(status: str) -> int:
    return {
        "ok": 0,
        "warning": 1,
        "critical": 2,
        "insufficient_data": 3,
    }.get(status, -1)
