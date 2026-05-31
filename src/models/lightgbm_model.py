from lightgbm import LGBMClassifier

from src.config_loader import ExperimentConfig, ModelConfig


def build_lightgbm(config: ModelConfig | ExperimentConfig) -> LGBMClassifier:
    if isinstance(config, ExperimentConfig):
        model_cfg = config.model
        random_state = config.training.random_state
    else:
        model_cfg = config
        random_state = 101

    return LGBMClassifier(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        num_leaves=model_cfg.num_leaves,
        subsample=model_cfg.subsample,
        colsample_bytree=model_cfg.colsample_bytree,
        random_state=random_state,
        n_jobs=model_cfg.n_jobs,
        verbose=-1,
    )
