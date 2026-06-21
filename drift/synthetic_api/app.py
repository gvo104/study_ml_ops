from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from drift.synthetic_api.config import SyntheticApiSettings, load_settings
from drift.synthetic_api.runtimes import (
    RuntimeErrorWithContext,
    build_runtime,
)
from drift.synthetic_api.schemas import HealthResponse, RunRequest, RunResponse


FRONTEND_PATH = Path(__file__).resolve().parent / "frontend" / "index.html"


@lru_cache(maxsize=1)
def _load_frontend() -> str:
    return FRONTEND_PATH.read_text(encoding="utf-8")


def create_app(settings: SyntheticApiSettings | None = None) -> FastAPI:
    app_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = app_settings
        app.state.runtime = build_runtime(
            backend=app_settings.backend,
            model_path=app_settings.model_path,
            temperature=app_settings.temperature,
            max_retries=app_settings.max_retries,
            max_tokens=app_settings.max_tokens,
            context_window=app_settings.context_window,
            gpu_layers=app_settings.gpu_layers,
        )
        yield
        app.state.runtime = None

    app = FastAPI(
        title="Synthetic LLM API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(_load_frontend())

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        runtime = request.app.state.runtime
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not loaded")

        return HealthResponse(
            status="ok",
            backend=runtime.backend_name,
            loaded=runtime.loaded,
        )

    @app.post("/llm/run", response_model=RunResponse)
    async def run_llm(payload: RunRequest, request: Request):
        runtime = request.app.state.runtime
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not loaded")

        try:
            return await runtime.run(payload)
        except RuntimeErrorWithContext as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
