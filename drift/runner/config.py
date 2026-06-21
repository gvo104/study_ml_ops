from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from drift.synthetic_api.schemas import ALLOWED_LABELS


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    window_size: int
    step_size: int
    phase_counts: dict[str, int]
    exploratory_thresholds: bool
    target_label_strategy: str
    min_examples_per_class_for_association: int
    min_classes_for_association_metric: int


@dataclass(frozen=True)
class RunnerConfig:
    mode: str
    predict_api_url: str
    synthetic_api_url: str
    data_path: Path
    training_config_path: Path
    reference_dir: Path
    output_root: Path
    expert_allowed_labels: list[str]
    max_generation_attempts_per_sample: int
    top_k_tokens: int
    top_k_label_tokens: int
    random_seed: int
    request_timeout_seconds: float
    target_shift_distribution: dict[str, float]
    profile: ProfileConfig


def load_runner_config(path: Path | str, mode: str) -> RunnerConfig:
    config_path = Path(path)
    with open(config_path, encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    profiles = raw["profiles"]
    if mode not in profiles:
        raise ValueError(f"Unknown runner mode: {mode}")

    profile_raw = profiles[mode]
    profile = ProfileConfig(
        name=mode,
        window_size=profile_raw["window_size"],
        step_size=profile_raw["step_size"],
        phase_counts=dict(profile_raw["phase_counts"]),
        exploratory_thresholds=bool(profile_raw["exploratory_thresholds"]),
        target_label_strategy=profile_raw.get("target_label_strategy", "reference"),
        min_examples_per_class_for_association=profile_raw.get(
            "min_examples_per_class_for_association",
            raw["min_examples_per_class_for_association"],
        ),
        min_classes_for_association_metric=profile_raw.get(
            "min_classes_for_association_metric",
            raw["min_classes_for_association_metric"],
        ),
    )

    target_shift_distribution = dict(raw["target_shift_distribution"])
    _validate_distribution(target_shift_distribution)

    labels = list(raw.get("expert_allowed_labels", ALLOWED_LABELS))

    return RunnerConfig(
        mode=mode,
        predict_api_url=raw["predict_api_url"],
        synthetic_api_url=raw["synthetic_api_url"],
        data_path=Path(raw["data_path"]),
        training_config_path=Path(raw["training_config_path"]),
        reference_dir=Path(raw["reference_dir"]),
        output_root=Path(raw["output_root"]),
        expert_allowed_labels=labels,
        max_generation_attempts_per_sample=raw["max_generation_attempts_per_sample"],
        top_k_tokens=raw["top_k_tokens"],
        top_k_label_tokens=raw["top_k_label_tokens"],
        random_seed=raw.get("random_seed", 101),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", 30.0)),
        target_shift_distribution=target_shift_distribution,
        profile=profile,
    )


def _validate_distribution(distribution: dict[str, float]) -> None:
    total = sum(distribution.values())
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"Distribution must sum to 1.0, got {total}")
