from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


ALLOWED_LABELS = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal",
]

ALLOWED_PHASES = ["A", "B", "C", "D"]


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style: str | None = None
    length: Literal["short", "medium", "long"] = "medium"


class GeneratorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["generator"]
    phase: Literal["A", "B", "C", "D"]
    target_label: Literal[
        "Anxiety",
        "Bipolar",
        "Depression",
        "Normal",
        "Personality disorder",
        "Stress",
        "Suicidal",
    ]
    constraints: Constraints = Field(default_factory=Constraints)


class ExpertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["expert"]
    text: str = Field(..., min_length=1)
    allowed_labels: list[str] = Field(..., min_length=1)


RunRequest = Annotated[
    Union[GeneratorRequest, ExpertRequest],
    Field(discriminator="role"),
]


class GeneratorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["generator"]
    text: str
    target_label: str
    phase: str


class ExpertResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["expert"]
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


RunResponse = Annotated[
    Union[GeneratorResponse, ExpertResponse],
    Field(discriminator="role"),
]


class HealthResponse(BaseModel):
    status: str
    backend: str
    loaded: bool
