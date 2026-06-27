from __future__ import annotations

import json
import random
import time
from pathlib import Path

import click

from drift.replay.worker import DEFAULT_DB_PATH
from drift.runner.clients import PredictApiClient, SyntheticApiClient
from drift.runner.config import load_runner_config
from drift.runner.pipeline import build_preprocessed_record
from drift.synthetic_api.schemas import ALLOWED_LABELS
from drift.store import PredictionEvent, PredictionEventStore


PHASE_MAP = {
    "normal": "A",
    "lexical_drift": "B",
    "label_distribution_shift": "C",
    "association_drift": "D",
    "recovery": "A",
}


@click.command("run")
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
@click.option("--count", type=int, default=10, show_default=True)
@click.option("--interval-seconds", type=float, default=1.0, show_default=True)
@click.option("--hidden-phase", default="normal", show_default=True)
def main(
    db_path: str,
    config_path: str,
    mode: str,
    count: int,
    interval_seconds: float,
    hidden_phase: str,
):
    config = load_runner_config(config_path, mode)
    result = run_online_traffic(
        db_path=Path(db_path),
        synthetic_client=SyntheticApiClient(
            config.synthetic_api_url,
            timeout_seconds=config.request_timeout_seconds,
        ),
        predict_client=PredictApiClient(
            config.predict_api_url,
            timeout_seconds=config.request_timeout_seconds,
        ),
        count=count,
        interval_seconds=interval_seconds,
        hidden_phase=hidden_phase,
        seed=config.random_seed,
    )
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


def run_online_traffic(
    db_path: Path,
    synthetic_client,
    predict_client,
    count: int,
    interval_seconds: float = 0.0,
    hidden_phase: str = "normal",
    seed: int = 101,
) -> dict[str, int]:
    store = PredictionEventStore(db_path)
    rng = random.Random(seed)
    inserted = 0
    skipped = 0
    api_phase = PHASE_MAP.get(hidden_phase, "A")

    for index in range(count):
        target_label = rng.choice(ALLOWED_LABELS)
        generated = synthetic_client.generate(
            {
                "role": "generator",
                "phase": api_phase,
                "target_label": target_label,
                "constraints": {"length": "medium"},
            }
        )
        text = generated["text"]
        prediction = predict_client.predict(text)
        preprocessed = build_preprocessed_record(text)
        event = PredictionEvent(
            event_id=f"online_{int(time.time() * 1000)}_{index:05d}",
            source_mode="online_synthetic",
            raw_text=text,
            tokens_stemmed=preprocessed["tokens_stemmed"],
            num_of_characters=preprocessed["num_of_characters"],
            num_of_sentences=preprocessed["num_of_sentences"],
            model_prediction=prediction["prediction"],
            model_confidence=prediction.get("confidence"),
            model_probabilities=prediction.get("probabilities", {}),
            hidden_phase=hidden_phase,
            hidden_target_label=target_label,
            metadata={"generator_phase": api_phase},
        )
        if store.insert_prediction_event(event):
            inserted += 1
        else:
            skipped += 1
        if interval_seconds > 0:
            time.sleep(interval_seconds)

    return {
        "inserted": inserted,
        "skipped": skipped,
        "total_events": store.count_events(),
    }


if __name__ == "__main__":
    main()
