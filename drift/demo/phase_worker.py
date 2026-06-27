from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from drift.replay.worker import DEFAULT_DB_PATH
from drift.runner.clients import PredictApiClient, SyntheticApiClient
from drift.runner.config import load_runner_config
from drift.runner.pipeline import build_preprocessed_record
from drift.store import PredictionEvent, PredictionEventStore
from drift.synthetic_api.schemas import ALLOWED_LABELS


DEFAULT_GENERATED_REPLAY_PATH = Path("drift/replay/messages.generated.jsonl")
DEFAULT_PHASE_COUNTS = {
    "A_baseline": 40,
    "B_lexical": 20,
    "C_label_distribution": 20,
    "D_association": 20,
    "A_recovery": 20,
}
PHASE_TO_API_PHASE = {
    "A_baseline": "A",
    "B_lexical": "B",
    "C_label_distribution": "C",
    "D_association": "D",
    "A_recovery": "A",
}
C_LABEL_SHIFT_DISTRIBUTION = [
    "Stress",
    "Stress",
    "Stress",
    "Stress",
    "Anxiety",
    "Anxiety",
    "Depression",
    "Normal",
]
D_ASSOCIATION_LABELS = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Stress",
]


@dataclass(frozen=True)
class PhaseGenerationResult:
    inserted: int
    skipped: int
    failed: int
    total_events: int
    replay_cache_path: str | None


@click.command()
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
@click.option("--interval-seconds", type=float, default=0.0, show_default=True)
@click.option(
    "--replay-output",
    type=click.Path(),
    default=str(DEFAULT_GENERATED_REPLAY_PATH),
    show_default=True,
)
@click.option("--reset-db/--no-reset-db", default=True, show_default=True)
def main(
    db_path: str,
    config_path: str,
    mode: str,
    interval_seconds: float,
    replay_output: str,
    reset_db: bool,
):
    config = load_runner_config(config_path, mode)
    result = generate_phase_demo_data(
        db_path=Path(db_path),
        synthetic_client=SyntheticApiClient(
            config.synthetic_api_url,
            timeout_seconds=config.request_timeout_seconds,
        ),
        predict_client=PredictApiClient(
            config.predict_api_url,
            timeout_seconds=config.request_timeout_seconds,
        ),
        allowed_labels=config.expert_allowed_labels,
        random_seed=config.random_seed,
        interval_seconds=interval_seconds,
        replay_output=Path(replay_output) if replay_output else None,
        reset_db=reset_db,
        progress_callback=_echo_progress,
    )
    click.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


def generate_phase_demo_data(
    db_path: Path,
    synthetic_client,
    predict_client,
    allowed_labels: list[str] | None = None,
    random_seed: int = 101,
    phase_counts: dict[str, int] | None = None,
    interval_seconds: float = 0.0,
    replay_output: Path | None = DEFAULT_GENERATED_REPLAY_PATH,
    reset_db: bool = True,
    progress_callback=None,
) -> PhaseGenerationResult:
    store = PredictionEventStore(db_path)
    if reset_db:
        store.clear_events()

    rng = random.Random(random_seed)
    allowed_labels = allowed_labels or ALLOWED_LABELS
    phase_counts = phase_counts or DEFAULT_PHASE_COUNTS
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    inserted = 0
    skipped = 0
    failed = 0
    replay_records: list[dict[str, Any]] = []
    total_planned = sum(phase_counts.values())

    sample_index = 0
    for hidden_phase, count in phase_counts.items():
        if progress_callback is not None:
            progress_callback(
                {
                    "type": "phase_start",
                    "hidden_phase": hidden_phase,
                    "phase_total": count,
                    "done": sample_index,
                    "total": total_planned,
                }
            )
        for phase_index in range(count):
            sample_index += 1
            target_label = choose_phase_target_label(
                hidden_phase=hidden_phase,
                index=phase_index,
                allowed_labels=allowed_labels,
                rng=rng,
            )
            try:
                api_phase = PHASE_TO_API_PHASE[hidden_phase]
                generator_payload = {
                    "role": "generator",
                    "phase": api_phase,
                    "target_label": target_label,
                    "constraints": {"length": "medium", "style": "neutral"},
                }
                generated = synthetic_client.generate(generator_payload)
                text = generated["text"]
                prediction = predict_client.predict(text)
                expert = synthetic_client.label(
                    {
                        "role": "expert",
                        "text": text,
                        "allowed_labels": allowed_labels,
                    }
                )
                preprocessed = build_preprocessed_record(text)
                event_id = f"phase_{run_id}_{sample_index:05d}"
                metadata = {
                    "target_label": target_label,
                    "expert_label": expert["label"],
                    "model_prediction": prediction["prediction"],
                    "generator_phase": api_phase,
                    "model_expert_disagreement": (
                        prediction["prediction"] != expert["label"]
                    ),
                    "target_expert_disagreement": target_label != expert["label"],
                }
                event = PredictionEvent(
                    event_id=event_id,
                    source_mode="phase_llm",
                    raw_text=text,
                    tokens_stemmed=preprocessed["tokens_stemmed"],
                    num_of_characters=preprocessed["num_of_characters"],
                    num_of_sentences=preprocessed["num_of_sentences"],
                    model_prediction=prediction["prediction"],
                    model_confidence=prediction.get("confidence"),
                    model_probabilities=prediction.get("probabilities", {}),
                    expert_label=expert["label"],
                    expert_confidence=expert.get("confidence"),
                    expert_reason=expert.get("reason"),
                    hidden_phase=hidden_phase,
                    hidden_target_label=target_label,
                    metadata=metadata,
                )
                if store.insert_prediction_event(event):
                    inserted += 1
                    replay_records.append(
                        {
                            "event_id": event_id,
                            "text": text,
                            "expert_label": expert["label"],
                            "expert_confidence": expert.get("confidence"),
                            "expert_reason": expert.get("reason"),
                            "hidden_phase": hidden_phase,
                            "hidden_target_label": target_label,
                            "metadata": metadata,
                        }
                    )
                    status = "inserted"
                else:
                    skipped += 1
                    status = "skipped"
            except Exception:
                failed += 1
                raise

            if progress_callback is not None:
                progress_callback(
                    {
                        "type": "event",
                        "hidden_phase": hidden_phase,
                        "phase_index": phase_index + 1,
                        "phase_total": count,
                        "done": sample_index,
                        "total": total_planned,
                        "target_label": target_label,
                        "model_prediction": prediction["prediction"],
                        "expert_label": expert["label"],
                        "status": status,
                        "text_preview": _preview_text(text),
                    }
                )

            if interval_seconds > 0:
                time.sleep(interval_seconds)

    if replay_output is not None:
        replay_output.parent.mkdir(parents=True, exist_ok=True)
        with open(replay_output, "w", encoding="utf-8") as file:
            for record in replay_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return PhaseGenerationResult(
        inserted=inserted,
        skipped=skipped,
        failed=failed,
        total_events=store.count_events(),
        replay_cache_path=str(replay_output) if replay_output is not None else None,
    )


def choose_phase_target_label(
    hidden_phase: str,
    index: int,
    allowed_labels: list[str],
    rng: random.Random,
) -> str:
    if hidden_phase == "C_label_distribution":
        candidates = [label for label in C_LABEL_SHIFT_DISTRIBUTION if label in allowed_labels]
        return candidates[index % len(candidates)]
    if hidden_phase == "D_association":
        candidates = [label for label in D_ASSOCIATION_LABELS if label in allowed_labels]
        return candidates[index % len(candidates)]
    if not allowed_labels:
        raise ValueError("allowed_labels must not be empty")
    return allowed_labels[index % len(allowed_labels)]


def _preview_text(text: str, limit: int = 90) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _echo_progress(payload: dict[str, Any]) -> None:
    if payload["type"] == "phase_start":
        click.echo(
            f"[phase] {payload['hidden_phase']} "
            f"planned={payload['phase_total']} "
            f"progress={payload['done']}/{payload['total']}"
        )
        return

    click.echo(
        f"[event] {payload['done']}/{payload['total']} "
        f"{payload['hidden_phase']} "
        f"{payload['phase_index']}/{payload['phase_total']} "
        f"target={payload['target_label']} "
        f"model={payload['model_prediction']} "
        f"expert={payload['expert_label']} "
        f"status={payload['status']} "
        f"text=\"{payload['text_preview']}\""
    )


if __name__ == "__main__":
    main()
