from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config_loader import ExperimentConfig, ModelConfig


def build_logistic_regression(
    config: ModelConfig | ExperimentConfig,
) -> LogisticRegression:
    if isinstance(config, ExperimentConfig):
        model_cfg = config.model
        random_state = config.training.random_state
    else:
        model_cfg = config
        random_state = 101

    return LogisticRegression(
        C=model_cfg.C,
        max_iter=model_cfg.max_iter,
        random_state=random_state,
        n_jobs=model_cfg.n_jobs,
    )


def build_random_forest(
    config: ModelConfig | ExperimentConfig,
) -> RandomForestClassifier:
    if isinstance(config, ExperimentConfig):
        model_cfg = config.model
        random_state = config.training.random_state
    else:
        model_cfg = config
        random_state = 101

    return RandomForestClassifier(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        random_state=random_state,
        n_jobs=model_cfg.n_jobs,
    )
