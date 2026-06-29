from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST

from drift_v2.monitoring.config import MonitoringSettings, load_settings
from drift_v2.monitoring.exporter import (
    generate_metrics_text,
    load_metrics_snapshot,
    load_metrics_snapshots,
    resolve_metrics_path,
)


def create_app(settings: MonitoringSettings | None = None) -> FastAPI:
    app_settings = settings or load_settings()
    app = FastAPI(title="Drift V2 Metrics Exporter", version="0.1.0")

    @app.get("/health")
    async def health():
        snapshots = load_metrics_snapshots(app_settings)
        snapshot = load_metrics_snapshot(app_settings)
        metrics_path = resolve_metrics_path(app_settings)
        return {
            "status": "ok",
            "active": snapshot.active,
            "file_present": snapshot.file_present,
            "metrics_path": str(metrics_path) if metrics_path is not None else None,
            "window_metrics_path": (
                str(snapshot.window_metrics_path)
                if snapshot.window_metrics_path is not None
                else None
            ),
            "recent_events_path": (
                str(snapshot.recent_events_path)
                if snapshot.recent_events_path is not None
                else None
            ),
            "window_count": snapshot.window_count,
            "recent_event_count": len(snapshot.recent_events),
            "run_count": len(snapshots),
            "active_runs": [
                {
                    "run_id": (item.latest_payload or {}).get("run_id", "unknown"),
                    "pipeline": ((item.latest_payload or {}).get("labels") or {}).get("pipeline", "unknown"),
                    "active": item.active,
                    "file_present": item.file_present,
                }
                for item in snapshots
            ],
            "parse_errors": snapshot.parse_errors,
        }

    @app.get("/metrics")
    async def metrics():
        snapshots = load_metrics_snapshots(app_settings)
        payload = generate_metrics_text(snapshots)
        return PlainTextResponse(payload.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
