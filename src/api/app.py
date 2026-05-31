from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.models.registry import load_predictor
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse

_predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = load_predictor()
    yield
    _predictor = None


app = FastAPI(
    title="Mental Health Text Classifier",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="ok",
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
    return PredictResponse(**result)
