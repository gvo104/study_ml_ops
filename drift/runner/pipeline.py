from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from drift.runner.config import RunnerConfig
from drift.runner.scenario import choose_target_label, scenario_phase_to_api_phase
from src.features.numerical import extract_numerical_features
from src.features.preprocess import preprocess_single


@dataclass
class SampleBuildResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]


def build_sample_stream(
    phase_sequence: list[str],
    reference_distribution: dict[str, float],
    config: RunnerConfig,
    synthetic_client,
    predict_client,
) -> SampleBuildResult:
    rng = random.Random(config.random_seed)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for sample_index, phase_name in enumerate(phase_sequence, start=1):
        target_label = choose_target_label(
            phase_name,
            rng,
            reference_distribution,
            config,
        )
        accepted_record = _attempt_build_sample(
            sample_index=sample_index,
            phase_name=phase_name,
            target_label=target_label,
            config=config,
            synthetic_client=synthetic_client,
            predict_client=predict_client,
            rejected_records=rejected,
        )
        if accepted_record is not None:
            accepted.append(accepted_record)

    return SampleBuildResult(accepted=accepted, rejected=rejected)


def build_windows(
    accepted_samples: list[dict[str, Any]],
    window_size: int,
    step_size: int,
) -> list[pd.DataFrame]:
    windows: list[pd.DataFrame] = []
    for start in range(0, len(accepted_samples) - window_size + 1, step_size):
        window_records = accepted_samples[start : start + window_size]
        windows.append(pd.DataFrame(window_records))
    return windows


def _attempt_build_sample(
    sample_index: int,
    phase_name: str,
    target_label: str,
    config: RunnerConfig,
    synthetic_client,
    predict_client,
    rejected_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    api_phase = scenario_phase_to_api_phase(phase_name)
    sample_id = f"sample_{sample_index:05d}"

    for attempt in range(1, config.max_generation_attempts_per_sample + 1):
        generator_payload = {
            "role": "generator",
            "phase": api_phase,
            "target_label": target_label,
            "constraints": {"length": "medium"},
        }
        generated = synthetic_client.generate(generator_payload)
        text = generated["text"]

        expert_payload = {
            "role": "expert",
            "text": text,
            "allowed_labels": config.expert_allowed_labels,
        }
        expert = synthetic_client.label(expert_payload)

        if expert["label"] != target_label:
            rejected_records.append(
                {
                    "sample_id": sample_id,
                    "phase": phase_name,
                    "target_label": target_label,
                    "generator_text": text,
                    "expert_label": expert["label"],
                    "rejection_reason": "expert_label_mismatch",
                    "attempt": attempt,
                    "rejected_at": _utc_now(),
                    "mode": config.mode,
                }
            )
            continue

        preprocessed = build_preprocessed_record(text)
        prediction = predict_client.predict(text)
        return {
            "sample_id": sample_id,
            "phase": phase_name,
            "target_label": target_label,
            "expert_label": expert["label"],
            "expert_confidence": expert["confidence"],
            "expert_reason": expert["reason"],
            "model_prediction": prediction["prediction"],
            "model_confidence": prediction["confidence"],
            "model_probabilities": prediction["probabilities"],
            "statement": preprocessed["statement"],
            "tokens_stemmed": preprocessed["tokens_stemmed"],
            "num_of_characters": preprocessed["num_of_characters"],
            "num_of_sentences": preprocessed["num_of_sentences"],
            "accepted_at": _utc_now(),
            "generation_attempt": attempt,
            "mode": config.mode,
        }

    rejected_records.append(
        {
            "sample_id": sample_id,
            "phase": phase_name,
            "target_label": target_label,
            "generator_text": None,
            "expert_label": None,
            "rejection_reason": "max_attempts_exceeded",
            "attempt": config.max_generation_attempts_per_sample,
            "rejected_at": _utc_now(),
            "mode": config.mode,
        }
    )
    return None


def build_preprocessed_record(text: str) -> dict[str, Any]:
    numerical = extract_numerical_features(text)[0]
    return {
        "statement": text,
        "tokens_stemmed": preprocess_single(text),
        "num_of_characters": int(numerical[0]),
        "num_of_sentences": int(numerical[1]),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
