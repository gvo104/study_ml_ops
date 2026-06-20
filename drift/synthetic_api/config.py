from dataclasses import dataclass
from pathlib import Path
import os


DEFAULT_MODEL_PATH = (
    Path("drift")
    / "models"
    / "Qwen2.5-3B-Instruct-GGUF"
    / "qwen2.5-3b-instruct-q4_k_m.gguf"
)


@dataclass(frozen=True)
class SyntheticApiSettings:
    backend: str = "mock"
    model_path: Path = DEFAULT_MODEL_PATH
    host: str = "0.0.0.0"
    port: int = 8001
    temperature: float = 0.3
    max_retries: int = 2


def load_settings() -> SyntheticApiSettings:
    return SyntheticApiSettings(
        backend=os.getenv("SYNTHETIC_LLM_BACKEND", "mock"),
        model_path=Path(
            os.getenv(
                "SYNTHETIC_LLM_MODEL_PATH",
                str(DEFAULT_MODEL_PATH),
            )
        ),
        host=os.getenv("SYNTHETIC_API_HOST", "0.0.0.0"),
        port=int(os.getenv("SYNTHETIC_API_PORT", "8001")),
        temperature=float(os.getenv("SYNTHETIC_LLM_TEMPERATURE", "0.3")),
        max_retries=int(os.getenv("SYNTHETIC_LLM_MAX_RETRIES", "2")),
    )
