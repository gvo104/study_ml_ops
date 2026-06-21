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
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
