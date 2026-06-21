from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from drift.runner.clients import PredictApiClient, SyntheticApiClient
from drift.runner.config import RunnerConfig, load_runner_config
from drift.runner.metrics import (
    classify_window_status,
    compute_thresholds,
    compute_window_metrics,
)
from drift.runner.pipeline import build_sample_stream, build_windows
from drift.runner.prom_metrics import build_prom_metrics_latest
from drift.runner.reference import ensure_reference_snapshot
from drift.runner.reporting import (
    append_jsonl,
    create_run_output_dir,
    write_json,
    write_markdown_report,
)
from drift.runner.scenario import build_phase_sequence


@click.group()
def main():
    """Synthetic drift monitoring runner."""


@main.command("run")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default="drift/configs/runner.yaml",
    show_default=True,
)
@click.option(
    "--mode",
    type=click.Choice(["debug", "full"]),
    default="debug",
    show_default=True,
)
def run_cmd(config_path: str, mode: str):
    config = load_runner_config(config_path, mode)
    summary = run_drift_monitoring(config)
    click.echo(json.dumps(summary, ensure_ascii=False, indent=2))


def run_drift_monitoring(
    config: RunnerConfig,
    synthetic_client=None,
    predict_client=None,
) -> dict:
    synthetic_client = synthetic_client or SyntheticApiClient(
        config.synthetic_api_url,
        timeout_seconds=config.request_timeout_seconds,
    )
    predict_client = predict_client or PredictApiClient(
        config.predict_api_url,
        timeout_seconds=config.request_timeout_seconds,
    )

    reference_df, reference_stats = ensure_reference_snapshot(config)
    run_id, output_dir = create_run_output_dir(config)
    phase_sequence = build_phase_sequence(config.profile.phase_counts)

    stream = build_sample_stream(
        phase_sequence=phase_sequence,
        reference_distribution=reference_stats["class_distribution"],
        config=config,
        synthetic_client=synthetic_client,
        predict_client=predict_client,
    )

    accepted_path = output_dir / "accepted_samples.jsonl"
    rejected_path = output_dir / "rejected_samples.jsonl"
    for record in stream.accepted:
        append_jsonl(accepted_path, record)
    for record in stream.rejected:
        append_jsonl(rejected_path, record)

    windows = build_windows(
        accepted_samples=stream.accepted,
        window_size=config.profile.window_size,
        step_size=config.profile.step_size,
    )
    window_metrics = _calculate_window_metrics(
        windows=windows,
        reference_stats=reference_stats,
        config=config,
    )

    baseline_windows = [
        item for item in window_metrics if item["phase"] == "A_baseline"
    ]
    thresholds = compute_thresholds(
        baseline_windows=baseline_windows,
        exploratory_thresholds=config.profile.exploratory_thresholds,
    )

    final_window_metrics = []
    metrics_path = output_dir / "window_metrics.jsonl"
    for window in window_metrics:
        status_payload = classify_window_status(window, thresholds)
        enriched = {**window, **status_payload}
        final_window_metrics.append(enriched)
        append_jsonl(metrics_path, enriched)

    latest_window = final_window_metrics[-1] if final_window_metrics else None
    prom_metrics = build_prom_metrics_latest(
        run_id=run_id,
        mode=config.mode,
        latest_window=latest_window,
    )

    summary = {
        "run_id": run_id,
        "mode": config.mode,
        "window_size": config.profile.window_size,
        "step_size": config.profile.step_size,
        "phase_counts": config.profile.phase_counts,
        "exploratory_thresholds": config.profile.exploratory_thresholds,
        "accepted_samples": len(stream.accepted),
        "rejected_samples": len(stream.rejected),
        "reference_rows": len(reference_df),
        "window_count": len(final_window_metrics),
        "thresholds": thresholds,
        "latest_window": latest_window,
    }

    write_json(output_dir / "run_summary.json", summary)
    write_json(output_dir / "prom_metrics_latest.json", prom_metrics)
    write_markdown_report(
        output_dir / "run_report.md",
        summary=summary,
        window_metrics=final_window_metrics,
    )
    return summary


def _calculate_window_metrics(
    windows: list[pd.DataFrame],
    reference_stats: dict,
    config: RunnerConfig,
) -> list[dict]:
    results = []
    for index, window in enumerate(windows):
        metrics = compute_window_metrics(
            window,
            reference_stats=reference_stats,
            top_k_tokens=config.top_k_tokens,
            min_examples_per_class=config.min_examples_per_class_for_association,
            min_classes=config.min_classes_for_association_metric,
        )
        phase = _majority_phase(window)
        results.append(
            {
                "window_index": index,
                "phase": phase,
                **metrics,
            }
        )
    return results


def _majority_phase(window: pd.DataFrame) -> str:
    return str(window["phase"].value_counts().idxmax())


if __name__ == "__main__":
    main()
