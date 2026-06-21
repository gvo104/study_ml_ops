from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST

from drift.monitoring.config import MonitoringSettings, load_settings
from drift.monitoring.exporter import (
    generate_metrics_text,
    load_metrics_snapshot,
    resolve_metrics_path,
)


def create_app(settings: MonitoringSettings | None = None) -> FastAPI:
    app_settings = settings or load_settings()
    app = FastAPI(title="Drift Metrics Exporter", version="0.1.0")

    @app.get("/health")
    async def health():
        snapshot = load_metrics_snapshot(app_settings)
        metrics_path = resolve_metrics_path(app_settings)
        return {
            "status": "ok",
            "file_present": snapshot.file_present,
            "metrics_path": str(metrics_path) if metrics_path is not None else None,
        }

    @app.get("/metrics")
    async def metrics():
        snapshot = load_metrics_snapshot(app_settings)
        payload = generate_metrics_text(snapshot)
        return PlainTextResponse(payload.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
