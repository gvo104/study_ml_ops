from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from drift.synthetic_api.config import SyntheticApiSettings, load_settings
from drift.synthetic_api.runtimes import (
    RuntimeErrorWithContext,
    build_runtime,
)
from drift.synthetic_api.schemas import HealthResponse, RunRequest, RunResponse


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
        )
        yield
        app.state.runtime = None

    app = FastAPI(
        title="Synthetic LLM API",
        version="0.1.0",
        lifespan=lifespan,
    )

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
