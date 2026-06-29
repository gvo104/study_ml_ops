from typing import Any, Callable

from src.models.sklearn_models import (
    build_logistic_regression,
    build_random_forest,
)
from src.models.xgboost_model import build_xgboost


def _build_lightgbm(config):
    from src.models.lightgbm_model import build_lightgbm

    return build_lightgbm(config)


def _lazy_lightgbm(config):
    try:
        return _build_lightgbm(config)
    except ImportError as exc:
        raise ImportError(
            "lightgbm is not installed. Run: pip install lightgbm"
        ) from exc


MODEL_REGISTRY: dict[str, Callable[..., Any]] = {
    "xgboost": build_xgboost,
    "lightgbm": _lazy_lightgbm,
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
}
