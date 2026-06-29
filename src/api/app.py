import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.api.bootstrap import ensure_dataset, model_available
from src.api.dashboard import DashboardState
from src.config import DEFAULT_CONFIG_PATH
from src.models.registry import load_predictor
from src.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RetrainResponse,
)
from src.api.ui import render_experiments_page, render_inference_page

_predictor = None
_startup_error = None
_dashboard = DashboardState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_runtime()
    yield
    clear_runtime()


def initialize_runtime():
    global _startup_error

    if os.getenv("CI_SMOKE_MODE", "").lower() != "true":
        import nltk
        nltk.download('punkt')

    dataset_ready, dataset_message = ensure_dataset()
    reload_predictor()

    if _predictor is not None:
        if not dataset_ready:
            _startup_error = dataset_message
        return

    if model_available():
        return

    if dataset_ready:
        _startup_error = (
            "Model artifacts were not found. Automatic bootstrap training has "
            "started in the background."
        )
        started = _dashboard.trigger_retraining(
            reload_predictor,
            reason="Bootstrapping model artifacts for the web UI.",
        )
        if not started:
            _startup_error = "Model bootstrap is already in progress."
    else:
        _startup_error = (
            "Model artifacts are not available yet, and the training dataset "
            f"could not be prepared. {dataset_message}"
        )


def reload_predictor():
    global _predictor, _startup_error
    try:
        _predictor = load_predictor()
        _startup_error = None
    except FileNotFoundError:
        _predictor = None
        _startup_error = (
            "Model artifacts are not available yet. Train the model or mount "
            "existing files into models/xgboost/."
        )
    except Exception:
        _predictor = None
        _startup_error = "Model could not be loaded during startup."
        raise


def clear_runtime():
    global _predictor, _startup_error
    _predictor = None
    _startup_error = None


app = FastAPI(
    title="Mental Health Text Classifier",
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def inference_page():
    return render_inference_page()


@app.get("/experiments", response_class=HTMLResponse)
def experiments_page():
    return render_experiments_page()


@app.get("/health", response_model=HealthResponse)
def health():
    if _predictor is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            detail=_startup_error or "Model not loaded",
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        feature_schema_version=_predictor.metadata.get(
            "feature_schema_version"
        ),
        accuracy=_predictor.metadata.get("accuracy"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = _predictor.predict(request.text)
    record = _dashboard.record_prediction(request.text, result)
    result["anomaly_flags"] = record.anomaly_flags
    return PredictResponse(**result)


@app.get("/api/dashboard")
def dashboard_summary():
    return _dashboard.dashboard_summary()


@app.get("/api/experiments")
def experiments_summary():
    return _dashboard.experiments_summary(
        predictor_metadata=_predictor.metadata if _predictor is not None else None,
        configs_dir=app.state.configs_dir,
        startup_error=_startup_error,
    )


@app.post("/api/retrain", response_model=RetrainResponse)
def retrain():
    started = _dashboard.trigger_retraining(reload_predictor)
    if not started:
        raise HTTPException(
            status_code=409,
            detail="Retraining is already in progress.",
        )

    return RetrainResponse(
        status="started",
        message="Retraining has started in the background.",
    )

app.state.configs_dir = DEFAULT_CONFIG_PATH.parent / "experiments"
