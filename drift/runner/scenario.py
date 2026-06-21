from __future__ import annotations

import random
from typing import Iterable

from drift.runner.config import RunnerConfig


PHASE_TO_API_PHASE = {
    "A_baseline": "A",
    "B_lexical": "B",
    "C_target_shift": "C",
    "D_association": "D",
    "A_recovery": "A",
}


def build_phase_sequence(phase_counts: dict[str, int]) -> list[str]:
    sequence: list[str] = []
    for phase_name, count in phase_counts.items():
        sequence.extend([phase_name] * count)
    return sequence


def scenario_phase_to_api_phase(phase_name: str) -> str:
    return PHASE_TO_API_PHASE[phase_name]


def choose_target_label(
    phase_name: str,
    rng: random.Random,
    reference_distribution: dict[str, float],
    config: RunnerConfig,
) -> str:
    if phase_name == "C_target_shift":
        distribution = config.target_shift_distribution
    elif config.profile.target_label_strategy == "balanced":
        distribution = _build_balanced_distribution(config.expert_allowed_labels)
    else:
        distribution = reference_distribution
    return _sample_from_distribution(distribution, rng)


def _sample_from_distribution(
    distribution: dict[str, float],
    rng: random.Random,
) -> str:
    items: Iterable[tuple[str, float]] = distribution.items()
    labels = [item[0] for item in items]
    weights = [distribution[label] for label in labels]
    return rng.choices(labels, weights=weights, k=1)[0]


def _build_balanced_distribution(labels: list[str]) -> dict[str, float]:
    if not labels:
        return {}
    weight = 1.0 / len(labels)
    return {label: weight for label in labels}
