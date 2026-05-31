from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config import DEFAULT_CONFIG_PATH


@dataclass
class ModelConfig:
    name: str = "xgboost"
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    num_leaves: int = 31
    C: float = 1.0
    max_iter: int = 2000
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_jobs: int = -1


@dataclass
class FeaturesConfig:
    tfidf_max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    svd_components: int = 300


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 101
    oversample: bool = True
    repreprocess_text: bool = False
    compute_numerical_from_raw: bool = False


@dataclass
class ExperimentMetaConfig:
    run_name: str = ""
    save_local_artifacts: bool = True
    description: str = ""


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentMetaConfig = field(default_factory=ExperimentMetaConfig)


def _parse_ngram_range(value: list[int] | tuple[int, int]) -> tuple[int, int]:
    return tuple(value)  # type: ignore[return-value]


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    """Load experiment configuration from a YAML file."""
    config_path = Path(path)
    with open(config_path, encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    features_raw = dict(raw.get("features", {}))
    if "ngram_range" in features_raw:
        features_raw["ngram_range"] = _parse_ngram_range(
            features_raw["ngram_range"]
        )

    model_raw = dict(raw.get("model", {}))
    model_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    model_kwargs = {k: v for k, v in model_raw.items() if k in model_fields}

    return ExperimentConfig(
        model=ModelConfig(**model_kwargs),
        features=FeaturesConfig(**features_raw),
        training=TrainingConfig(**raw.get("training", {})),
        experiment=ExperimentMetaConfig(**raw.get("experiment", {})),
    )


def config_to_mlflow_params(config: ExperimentConfig) -> dict[str, Any]:
    """Flatten config for MLflow param logging."""
    params: dict[str, Any] = {
        "model_name": config.model.name,
        "run_name": config.experiment.run_name,
        "tfidf_max_features": config.features.tfidf_max_features,
        "ngram_range": str(config.features.ngram_range),
        "svd_components": config.features.svd_components,
        "test_size": config.training.test_size,
        "oversample": config.training.oversample,
    }

    for field_name in ModelConfig.__dataclass_fields__:
        if field_name == "name":
            continue
        params[field_name] = getattr(config.model, field_name)

    return params
