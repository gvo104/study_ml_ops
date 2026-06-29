from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from drift_v2.contracts import ArtifactBundle


def run_output_dir(output_root: Path, run_id: str, mode: str) -> Path:
    output_dir = output_root / f"{run_id}_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_artifact_bundle(output_root: Path, bundle: ArtifactBundle) -> Path:
    output_dir = run_output_dir(output_root, bundle.run_id, bundle.mode)
    _write_json(output_dir / "recent_events.json", {"events": bundle.recent_events})
    _write_jsonl(output_dir / "window_metrics.jsonl", bundle.window_metrics)
    _write_json(output_dir / "prom_metrics_latest.json", bundle.latest_payload)
    _write_json(output_dir / "run_summary.json", bundle.summary_payload)
    return output_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")
