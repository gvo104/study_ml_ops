from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to classify")


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    anomaly_flags: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool = False
    feature_schema_version: str | None = None
    accuracy: float | None = None
    detail: str | None = None


class RetrainResponse(BaseModel):
    status: str
    message: str
