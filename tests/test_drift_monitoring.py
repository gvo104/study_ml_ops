import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("prometheus_client")

from httpx import ASGITransport, AsyncClient

from drift.monitoring.app import create_app
from drift.monitoring.config import MonitoringSettings
from drift.monitoring.exporter import (
    generate_metrics_text,
    load_metrics_snapshot,
    resolve_metrics_path,
)


def test_resolve_metrics_path_picks_latest_run_directory(tmp_path):
    older = tmp_path / "20260620T100000Z_debug"
    newer = tmp_path / "20260621T192330Z_debug"
    older.mkdir()
    newer.mkdir()
    (older / "prom_metrics_latest.json").write_text("{}", encoding="utf-8")
    (newer / "prom_metrics_latest.json").write_text("{}", encoding="utf-8")

    settings = MonitoringSettings(runs_root=tmp_path)

    resolved = resolve_metrics_path(settings)

    assert resolved == newer / "prom_metrics_latest.json"


def test_explicit_metrics_path_has_priority(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    nested = runs_root / "20260621T192330Z_debug"
    nested.mkdir()
    (nested / "prom_metrics_latest.json").write_text("{}", encoding="utf-8")
    explicit = tmp_path / "manual.json"
    explicit.write_text("{}", encoding="utf-8")

    settings = MonitoringSettings(
        runs_root=runs_root,
        explicit_metrics_path=explicit,
    )

    assert resolve_metrics_path(settings) == explicit


def test_generate_metrics_text_handles_missing_file():
    settings = MonitoringSettings(runs_root=Path("missing"))
    snapshot = load_metrics_snapshot(settings)

    output = generate_metrics_text(snapshot).decode("utf-8")

    assert "drift_metrics_file_present{" in output
    assert 'run_id="unknown"' in output
    assert 'mode="unknown"' in output
    assert 'phase="unknown"' in output
    assert 'status="unknown"' in output
    assert " 0.0" in output
    assert "drift_token_distribution_jsd" not in output


def test_generate_metrics_text_exports_numeric_metrics_and_labels(tmp_path):
    metrics_path = tmp_path / "prom_metrics_latest.json"
    metrics_path.write_text(
        json.dumps(
            {
                "drift_token_distribution_jsd": 0.11,
                "drift_target_distribution_jsd": 0.22,
                "drift_model_expert_disagreement_rate": 0.33,
                "drift_model_expert_macro_f1": 0.44,
                "drift_token_label_association_drift": 0.55,
                "drift_window_status_code": 2,
                "drift_window_index": 10,
                "drift_insufficient_data_flag": 1,
                "drift_window_phase": "D_association",
                "labels": {
                    "run_id": "run-1",
                    "mode": "debug",
                    "phase": "D_association",
                    "status": "critical",
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_metrics_snapshot(
        MonitoringSettings(explicit_metrics_path=metrics_path)
    )

    output = generate_metrics_text(snapshot).decode("utf-8")

    assert "drift_token_distribution_jsd{" in output
    assert 'run_id="run-1"' in output
    assert 'mode="debug"' in output
    assert 'phase="D_association"' in output
    assert 'status="critical"' in output
    assert "0.11" in output
    assert "drift_window_status_code{" in output
    assert "2.0" in output
    assert "drift_window_phase" not in output


@pytest.mark.anyio
async def test_exporter_health_and_metrics_endpoints(tmp_path):
    metrics_path = tmp_path / "prom_metrics_latest.json"
    metrics_path.write_text(
        json.dumps(
            {
                "drift_model_expert_macro_f1": 0.7,
                "labels": {
                    "run_id": "run-2",
                    "mode": "full",
                    "phase": "A_baseline",
                    "status": "ok",
                },
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        MonitoringSettings(explicit_metrics_path=metrics_path)
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        health = await client.get("/health")
        metrics = await client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["file_present"] is True
    assert metrics.status_code == 200
    assert "drift_model_expert_macro_f1{" in metrics.text
    assert 'run_id="run-2"' in metrics.text
    assert 'mode="full"' in metrics.text
    assert 'phase="A_baseline"' in metrics.text
    assert 'status="ok"' in metrics.text
    assert "0.7" in metrics.text
