from xgboost import XGBClassifier

from src.config_loader import ExperimentConfig, ModelConfig


def build_xgboost(config: ModelConfig | ExperimentConfig) -> XGBClassifier:
    """Build an XGBClassifier from model or experiment config."""
    if isinstance(config, ExperimentConfig):
        model_cfg = config.model
        random_state = config.training.random_state
    else:
        model_cfg = config
        random_state = 101

    return XGBClassifier(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        random_state=random_state,
        eval_metric="mlogloss",
    )
