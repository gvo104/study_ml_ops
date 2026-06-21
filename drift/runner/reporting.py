from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drift.runner.config import RunnerConfig


def create_run_output_dir(config: RunnerConfig) -> tuple[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = config.output_root / f"{run_id}_{config.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_id, output_dir


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown_report(
    path: Path,
    summary: dict[str, Any],
    window_metrics: list[dict[str, Any]],
    label_diagnostics: dict[str, Any],
) -> None:
    strongest_token_window = max(
        window_metrics,
        key=lambda item: item.get("token_distribution_jsd", 0.0),
        default=None,
    )
    worst_disagreement_window = max(
        window_metrics,
        key=lambda item: item.get("model_expert_disagreement_rate", 0.0),
        default=None,
    )
    strongest_association_window = max(
        [item for item in window_metrics if item.get("token_label_association_drift") is not None],
        key=lambda item: item.get("token_label_association_drift", 0.0),
        default=None,
    )
    worst_generated_labels = label_diagnostics.get("worst_generated_labels", [])
    worst_disagreement_labels = label_diagnostics.get(
        "worst_model_disagreement_labels", []
    )
    insufficient_metrics = label_diagnostics.get("insufficient_data_metrics", [])

    lines = [
        "# Drift Runner Report",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- mode: `{summary['mode']}`",
        f"- overall_windows: `{len(window_metrics)}`",
        f"- exploratory_thresholds: `{summary['exploratory_thresholds']}`",
        "",
    ]
    if strongest_token_window:
        lines.append(
            f"- strongest token drift window: `{strongest_token_window['window_index']}`"
        )
    if worst_disagreement_window:
        lines.append(
            f"- worst disagreement window: `{worst_disagreement_window['window_index']}`"
        )
    if strongest_association_window:
        lines.append(
            f"- strongest association drift window: `{strongest_association_window['window_index']}`"
        )
    if worst_generated_labels:
        lines.append(
            f"- worst generated labels: `{', '.join(worst_generated_labels)}`"
        )
    if worst_disagreement_labels:
        lines.append(
            f"- worst model disagreement labels: `{', '.join(worst_disagreement_labels)}`"
        )
    if insufficient_metrics:
        lines.append(
            f"- insufficient-data metrics: `{', '.join(insufficient_metrics)}`"
        )
    if summary["mode"] == "debug":
        lines.append(
            "- association drift in debug mode is exploratory and not a final monitoring decision"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_label_diagnostics(
    accepted_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
    window_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted_by_label: dict[str, int] = {}
    terminal_failures_by_label: dict[str, int] = {}
    reject_matrix: dict[str, dict[str, int]] = {}
    model_expert_confusion_matrix: dict[str, dict[str, int]] = {}
    disagreement_by_label: dict[str, dict[str, float | int]] = {}

    for record in accepted_records:
        label = record["target_label"]
        accepted_by_label[label] = accepted_by_label.get(label, 0) + 1

        expert_label = record["expert_label"]
        model_label = record["model_prediction"]
        confusion_row = model_expert_confusion_matrix.setdefault(expert_label, {})
        confusion_row[model_label] = confusion_row.get(model_label, 0) + 1

        stats = disagreement_by_label.setdefault(
            expert_label,
            {"accepted": 0, "disagreements": 0, "disagreement_rate": 0.0},
        )
        stats["accepted"] += 1
        if expert_label != model_label:
            stats["disagreements"] += 1

    for record in rejected_records:
        target_label = record["target_label"]
        if record["rejection_reason"] == "max_attempts_exceeded":
            terminal_failures_by_label[target_label] = (
                terminal_failures_by_label.get(target_label, 0) + 1
            )
            continue
        expert_label = record.get("expert_label") or "unknown"
        row = reject_matrix.setdefault(target_label, {})
        row[expert_label] = row.get(expert_label, 0) + 1

    accept_rate_by_label: dict[str, float] = {}
    all_labels = sorted(set(accepted_by_label) | set(terminal_failures_by_label))
    for label in all_labels:
        accepted_count = accepted_by_label.get(label, 0)
        failed_count = terminal_failures_by_label.get(label, 0)
        total = accepted_count + failed_count
        accept_rate_by_label[label] = (
            round(accepted_count / total, 4) if total else 0.0
        )

    total_requested = len(accepted_records) + sum(terminal_failures_by_label.values())
    accept_rate = (
        round(len(accepted_records) / total_requested, 4) if total_requested else 0.0
    )

    accepted_distribution = {
        label: round(count / len(accepted_records), 4)
        for label, count in sorted(accepted_by_label.items())
        if accepted_records
    }

    for stats in disagreement_by_label.values():
        accepted_count = int(stats["accepted"])
        disagreements = int(stats["disagreements"])
        stats["disagreement_rate"] = (
            round(disagreements / accepted_count, 4) if accepted_count else 0.0
        )

    insufficient_data_metrics = sorted(
        {
            metric_name
            for window in window_metrics
            for metric_name, status in window.get("metric_statuses", {}).items()
            if status == "insufficient_data"
        }
    )

    worst_generated_labels = [
        label
        for label, _ in sorted(
            accept_rate_by_label.items(),
            key=lambda item: (item[1], item[0]),
        )[:3]
    ]
    worst_model_disagreement_labels = [
        label
        for label, _ in sorted(
            (
                (label, float(stats["disagreement_rate"]))
                for label, stats in disagreement_by_label.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )[:3]
    ]

    return {
        "accept_rate": accept_rate,
        "accept_rate_by_label": accept_rate_by_label,
        "accepted_distribution": accepted_distribution,
        "reject_matrix": reject_matrix,
        "model_expert_confusion_matrix": model_expert_confusion_matrix,
        "model_expert_disagreement_by_label": disagreement_by_label,
        "terminal_failures_by_label": terminal_failures_by_label,
        "worst_generated_labels": worst_generated_labels,
        "worst_model_disagreement_labels": worst_model_disagreement_labels,
        "insufficient_data_metrics": insufficient_data_metrics,
    }
