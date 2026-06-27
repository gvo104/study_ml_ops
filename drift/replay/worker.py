from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import click

from drift.runner.clients import PredictApiClient
from drift.runner.config import load_runner_config
from drift.runner.pipeline import build_preprocessed_record
from drift.store import PredictionEvent, PredictionEventStore


DEFAULT_REPLAY_DATASET = Path("drift/replay/messages.jsonl")
DEFAULT_DB_PATH = Path("drift/artifacts/demo/events.sqlite")


@dataclass(frozen=True)
class ReplayRecord:
    event_id: str
    text: str
    expert_label: str
    expert_confidence: float | None
    expert_reason: str | None
    hidden_phase: str | None
    hidden_target_label: str | None
    metadata: dict[str, Any]


def load_replay_records(path: Path | str) -> list[ReplayRecord]:
    records: list[ReplayRecord] = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(_parse_replay_record(payload, line_number))
    return records


def run_replay(
    dataset_path: Path | str,
    db_path: Path | str,
    predict_client,
    interval_seconds: float = 0.0,
    limit: int | None = None,
) -> dict[str, int]:
    store = PredictionEventStore(db_path)
    inserted = 0
    skipped = 0
    failed = 0

    records = load_replay_records(dataset_path)
    if limit is not None:
        records = records[:limit]

    for record in records:
        try:
            prediction = predict_client.predict(record.text)
            preprocessed = build_preprocessed_record(record.text)
            event = PredictionEvent(
                event_id=record.event_id,
                source_mode="replay",
                raw_text=record.text,
                tokens_stemmed=preprocessed["tokens_stemmed"],
                num_of_characters=preprocessed["num_of_characters"],
                num_of_sentences=preprocessed["num_of_sentences"],
                model_prediction=prediction["prediction"],
                model_confidence=prediction.get("confidence"),
                model_probabilities=prediction.get("probabilities", {}),
                expert_label=record.expert_label,
                expert_confidence=record.expert_confidence,
                expert_reason=record.expert_reason,
                hidden_phase=record.hidden_phase,
                hidden_target_label=record.hidden_target_label,
                metadata=record.metadata,
            )
            if store.insert_prediction_event(event):
                inserted += 1
            else:
                skipped += 1
        except Exception:
            failed += 1
            raise

        if interval_seconds > 0:
            time.sleep(interval_seconds)

    return {
        "inserted": inserted,
        "skipped": skipped,
        "failed": failed,
        "total_events": store.count_events(),
    }


@click.command("run")
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True),
    default=str(DEFAULT_REPLAY_DATASET),
    show_default=True,
)
@click.option(
    "--db-path",
    type=click.Path(),
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
@click.option("--interval-seconds", type=float, default=1.0, show_default=True)
@click.option("--limit", type=int, default=None)
def main(
    dataset_path: str,
    db_path: str,
    config_path: str,
    mode: str,
    interval_seconds: float,
    limit: int | None,
):
    config = load_runner_config(config_path, mode)
    result = run_replay(
        dataset_path=dataset_path,
        db_path=db_path,
        predict_client=PredictApiClient(
            config.predict_api_url,
            timeout_seconds=config.request_timeout_seconds,
        ),
        interval_seconds=interval_seconds,
        limit=limit,
    )
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


def _parse_replay_record(payload: dict[str, Any], line_number: int) -> ReplayRecord:
    required = ["event_id", "text", "expert_label"]
    missing = [field for field in required if not payload.get(field)]
    if missing:
        raise ValueError(f"Replay record {line_number} missing fields: {missing}")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Replay record {line_number} metadata must be an object")
    return ReplayRecord(
        event_id=str(payload["event_id"]),
        text=str(payload["text"]),
        expert_label=str(payload["expert_label"]),
        expert_confidence=_optional_float(payload.get("expert_confidence")),
        expert_reason=(
            str(payload["expert_reason"])
            if payload.get("expert_reason") is not None
            else None
        ),
        hidden_phase=(
            str(payload["hidden_phase"])
            if payload.get("hidden_phase") is not None
            else None
        ),
        hidden_target_label=(
            str(payload["hidden_target_label"])
            if payload.get("hidden_target_label") is not None
            else None
        ),
        metadata=metadata,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    main()
