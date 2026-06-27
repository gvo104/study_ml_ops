from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import pandas as pd

from drift.replay.worker import DEFAULT_DB_PATH
from drift.runner.config import RunnerConfig, load_runner_config
from drift.runner.metrics import (
    METRIC_DIRECTIONS,
    compute_disagreement_rate,
    compute_macro_f1,
    compute_mean_numeric,
    compute_pre_label_window_metrics,
    compute_target_distribution_jsd,
    compute_token_label_association_drift,
)
from drift.runner.prom_metrics import build_prom_metrics_latest
from drift.runner.reference import ensure_reference_snapshot
from drift.runner.reporting import append_jsonl, write_json
from drift.store import PredictionEventStore, StoredPredictionEvent


DEFAULT_OUTPUT_ROOT = Path("drift/artifacts/demo/runs")
BASELINE_PHASES = {"normal", "a_baseline", "a", "baseline"}
DEFAULT_DEMO_WINDOW_SIZE = 20
DEFAULT_DEMO_STEP_SIZE = 20
DEMO_THRESHOLD_WARNING_MARGIN = 0.05
DEMO_THRESHOLD_CRITICAL_MARGIN = 0.15


@click.group()
def main():
    """Demo drift metrics worker backed by SQLite prediction events."""


@main.command("compute")
@click.option(
    "--db-path",
    type=click.Path(exists=True),
    default=str(DEFAULT_DB_PATH),
    show_default=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default="drift/configs/runner.yaml",
    show_default=True,
)
@click.option("--mode", default="debug", show_default=True)
@click.option("--window-size", type=int, default=None)
@click.option("--step-size", type=int, default=None)
@click.option(
    "--output-root",
    type=click.Path(),
    default=str(DEFAULT_OUTPUT_ROOT),
    show_default=True,
)
def compute_cmd(
    db_path: str,
    config_path: str,
    mode: str,
    window_size: int | None,
    step_size: int | None,
    output_root: str,
):
    config = load_runner_config(config_path, mode)
    summary = compute_demo_metrics(
        db_path=Path(db_path),
        config=config,
        output_root=Path(output_root),
        window_size=window_size or DEFAULT_DEMO_WINDOW_SIZE,
        step_size=step_size or DEFAULT_DEMO_STEP_SIZE,
    )
    click.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@main.command("export-candidates")
@click.option(
    "--db-path",
    type=click.Path(exists=True),
    default=str(DEFAULT_DB_PATH),
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="drift/artifacts/demo/training_candidates.csv",
    show_default=True,
)
@click.option("--include-hidden", is_flag=True, default=False)
def export_candidates_cmd(db_path: str, output_path: str, include_hidden: bool):
    count = export_training_candidates(
        db_path=Path(db_path),
        output_path=Path(output_path),
        include_hidden=include_hidden,
    )
    click.echo(json.dumps({"exported_rows": count}, ensure_ascii=False, indent=2))


def compute_demo_metrics(
    db_path: Path,
    config: RunnerConfig,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    window_size: int | None = None,
    step_size: int | None = None,
) -> dict[str, Any]:
    window_size = window_size or DEFAULT_DEMO_WINDOW_SIZE
    step_size = step_size or DEFAULT_DEMO_STEP_SIZE

    store = PredictionEventStore(db_path)
    events = list(reversed(store.load_recent_events(limit=100_000)))
    _, reference_stats = ensure_reference_snapshot(config)
    run_id, output_dir = _create_run_output_dir(output_root, config.mode)

    windows = _build_event_windows(events, window_size=window_size, step_size=step_size)
    window_metrics = [
        _compute_event_window_metrics(
            index=index,
            window_events=window,
            reference_stats=reference_stats,
            config=config,
        )
        for index, window in enumerate(windows)
    ]

    baseline_windows = [
        window
        for window in window_metrics
        if _is_baseline_window(windows[window["window_index"]])
    ]
    thresholds = compute_demo_thresholds(
        baseline_windows=baseline_windows,
        exploratory_thresholds=config.profile.exploratory_thresholds,
    )
    threshold_calibration_status = (
        "ok" if baseline_windows else "insufficient_baseline"
    )

    final_windows: list[dict[str, Any]] = []
    metrics_path = output_dir / "window_metrics.jsonl"
    critical_event_ids: list[str] = []
    for index, window in enumerate(window_metrics):
        status_payload = classify_demo_window_status(window, thresholds)
        enriched = {**window, **status_payload}
        final_windows.append(enriched)
        append_jsonl(metrics_path, enriched)
        if enriched["overall_status"] == "critical":
            critical_event_ids.extend(
                event.event_id
                for event in windows[index]
                if event.expert_label is not None
            )

    marked_candidates = store.mark_training_candidates(sorted(set(critical_event_ids)))
    latest_window = final_windows[-1] if final_windows else None
    prom_metrics = build_prom_metrics_latest(
        run_id=run_id,
        mode=f"demo_{config.mode}",
        latest_window=latest_window,
    )
    summary = {
        "run_id": run_id,
        "mode": f"demo_{config.mode}",
        "db_path": str(db_path),
        "window_size": window_size,
        "step_size": step_size,
        "event_count": len(events),
        "labeled_event_count": sum(1 for event in events if event.expert_label is not None),
        "window_count": len(final_windows),
        "threshold_calibration_status": threshold_calibration_status,
        "baseline_window_count": len(baseline_windows),
        "thresholds": thresholds,
        "latest_window": latest_window,
        "training_candidates_marked": marked_candidates,
    }

    write_json(output_dir / "run_summary.json", summary)
    write_json(output_dir / "prom_metrics_latest.json", prom_metrics)
    return summary


def export_training_candidates(
    db_path: Path,
    output_path: Path,
    include_hidden: bool = False,
) -> int:
    store = PredictionEventStore(db_path)
    events = store.load_training_candidates()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "created_at",
        "source_mode",
        "raw_text",
        "tokens_stemmed",
        "num_of_characters",
        "num_of_sentences",
        "expert_label",
        "expert_confidence",
        "model_prediction",
        "model_confidence",
    ]
    if include_hidden:
        fieldnames.extend(["hidden_phase", "hidden_target_label"])

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            row = {
                "event_id": event.event_id,
                "created_at": event.created_at,
                "source_mode": event.source_mode,
                "raw_text": event.raw_text,
                "tokens_stemmed": event.tokens_stemmed,
                "num_of_characters": event.num_of_characters,
                "num_of_sentences": event.num_of_sentences,
                "expert_label": event.expert_label,
                "expert_confidence": event.expert_confidence,
                "model_prediction": event.model_prediction,
                "model_confidence": event.model_confidence,
            }
            if include_hidden:
                row["hidden_phase"] = event.hidden_phase
                row["hidden_target_label"] = event.hidden_target_label
            writer.writerow(row)
    return len(events)


def _compute_event_window_metrics(
    index: int,
    window_events: list[StoredPredictionEvent],
    reference_stats: dict[str, Any],
    config: RunnerConfig,
) -> dict[str, Any]:
    all_df = pd.DataFrame([event.to_metrics_record() for event in window_events])
    labeled_df = all_df[all_df["expert_label"].notna()].copy()

    metrics = compute_pre_label_window_metrics(
        all_df,
        reference_stats=reference_stats,
        top_k_tokens=config.top_k_tokens,
    )

    if labeled_df.empty:
        metrics.update(
            {
                "target_distribution_jsd": None,
                "expert_confidence_mean": None,
                "model_expert_disagreement_rate": None,
                "model_expert_macro_f1": None,
                "token_label_association_drift": None,
                "token_label_association_status": "insufficient_data",
                "token_label_association_valid_labels": [],
            }
        )
    else:
        metrics.update(
            {
                "target_distribution_jsd": compute_target_distribution_jsd(
                    labeled_df,
                    reference_stats["class_distribution"],
                ),
                "expert_confidence_mean": compute_mean_numeric(
                    labeled_df,
                    "expert_confidence",
                ),
                "model_expert_disagreement_rate": compute_disagreement_rate(labeled_df),
                "model_expert_macro_f1": compute_macro_f1(labeled_df),
            }
        )
        metrics.update(
            compute_token_label_association_drift(
                labeled_df,
                reference_stats["label_token_scores"],
                min_examples_per_class=(
                    config.profile.min_examples_per_class_for_association
                ),
                min_classes=config.profile.min_classes_for_association_metric,
            )
        )

    return {
        "window_index": index,
        "phase": _window_phase(window_events),
        **metrics,
    }



def compute_demo_thresholds(
    baseline_windows: list[dict[str, Any]],
    exploratory_thresholds: bool,
) -> dict[str, Any]:
    thresholds: dict[str, Any] = {
        "exploratory_thresholds": exploratory_thresholds,
        "calibration_status": "ok" if baseline_windows else "insufficient_baseline",
        "metrics": {},
    }
    for name, direction in METRIC_DIRECTIONS.items():
        values = [
            float(window[name])
            for window in baseline_windows
            if window.get(name) is not None
        ]
        if not values:
            thresholds["metrics"][name] = {
                "warning": None,
                "critical": None,
                "direction": direction,
            }
            continue

        if direction == "lower_is_worse":
            baseline = min(values)
            warning_margin = max(abs(baseline) * DEMO_THRESHOLD_WARNING_MARGIN, 1e-6)
            critical_margin = max(abs(baseline) * DEMO_THRESHOLD_CRITICAL_MARGIN, 1e-6)
            warning = baseline - warning_margin
            critical = baseline - critical_margin
        else:
            baseline = max(values)
            warning_margin = max(abs(baseline) * DEMO_THRESHOLD_WARNING_MARGIN, 1e-6)
            critical_margin = max(abs(baseline) * DEMO_THRESHOLD_CRITICAL_MARGIN, 1e-6)
            warning = baseline + warning_margin
            critical = baseline + critical_margin

        thresholds["metrics"][name] = {
            "warning": float(warning),
            "critical": float(critical),
            "direction": direction,
        }
    return thresholds


def classify_demo_window_status(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    statuses: dict[str, str] = {}
    warning_hits = 0
    critical_hits = 0

    if thresholds.get("calibration_status") != "ok":
        for metric_name in thresholds.get("metrics", {}):
            statuses[metric_name] = "unavailable"
        return {"metric_statuses": statuses, "overall_status": "insufficient_data"}

    for metric_name, threshold in thresholds["metrics"].items():
        value = metrics.get(metric_name)
        if metric_name == "token_label_association_drift" and metrics.get(
            "token_label_association_status"
        ) == "insufficient_data":
            statuses[metric_name] = "insufficient_data"
            continue
        if value is None or threshold["warning"] is None:
            statuses[metric_name] = "unavailable"
            continue

        if threshold["direction"] == "lower_is_worse":
            if value < threshold["critical"]:
                statuses[metric_name] = "critical"
                critical_hits += 1
            elif value < threshold["warning"]:
                statuses[metric_name] = "warning"
                warning_hits += 1
            else:
                statuses[metric_name] = "ok"
            continue

        if value > threshold["critical"]:
            statuses[metric_name] = "critical"
            critical_hits += 1
        elif value > threshold["warning"]:
            statuses[metric_name] = "warning"
            warning_hits += 1
        else:
            statuses[metric_name] = "ok"

    if critical_hits >= 1 or warning_hits >= 2:
        overall = "critical"
    elif warning_hits >= 1:
        overall = "warning"
    else:
        overall = "ok"
    return {"metric_statuses": statuses, "overall_status": overall}

def _build_event_windows(
    events: list[StoredPredictionEvent],
    window_size: int,
    step_size: int,
) -> list[list[StoredPredictionEvent]]:
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")
    return [
        events[start : start + window_size]
        for start in range(0, len(events) - window_size + 1, step_size)
    ]



def _window_phase(events: list[StoredPredictionEvent]) -> str:
    phases = sorted({event.hidden_phase or "unknown" for event in events})
    if len(phases) == 1:
        return phases[0]
    return "mixed"

def _is_baseline_window(events: list[StoredPredictionEvent]) -> bool:
    if not events:
        return False
    return all((event.hidden_phase or "").lower() in BASELINE_PHASES for event in events)


def _create_run_output_dir(output_root: Path, mode: str) -> tuple[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"{run_id}_demo_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_id, output_dir


if __name__ == "__main__":
    main()
