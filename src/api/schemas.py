from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to classify")


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    feature_schema_version: str | None = None
    accuracy: float | None = None
