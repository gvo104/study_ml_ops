import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("prometheus_client")

from httpx import ASGITransport, AsyncClient

from drift_v2.monitoring.app import create_app
from drift_v2.monitoring.config import MonitoringSettings
from drift_v2.monitoring.exporter import (
    generate_metrics_text,
    load_metrics_snapshots,
    load_metrics_snapshot,
    resolve_metrics_path,
)


def test_resolve_metrics_path_picks_latest_run_directory(tmp_path):
    older = tmp_path / "offline_demo_live_v2_debug"
    newer = tmp_path / "z_demo_debug"
    older.mkdir()
    newer.mkdir()
    (older / "prom_metrics_latest.json").write_text("{}", encoding="utf-8")
    (newer / "prom_metrics_latest.json").write_text("{}", encoding="utf-8")

    assert resolve_metrics_path(MonitoringSettings(runs_root=tmp_path)) == newer / "prom_metrics_latest.json"


def test_generate_metrics_text_exports_window_metrics_only(tmp_path):
    run_dir = tmp_path / "offline_demo_live_v2_debug"
    run_dir.mkdir()
    (run_dir / "prom_metrics_latest.json").write_text(
        json.dumps(
            {
                "run_id": "offline_demo_live_v2",
                "drift_v2_runner_mode": "debug",
                "drift_v2_pipeline_mode": "offline",
                "drift_v2_visible_event_count": 40,
                "drift_v2_total_event_count": 120,
                "drift_v2_complete_window_count": 2,
                "drift_v2_token_distribution_jsd": 0.11,
                "drift_v2_model_prediction_distribution_jsd": 0.12,
                "drift_v2_target_distribution_jsd": 0.22,
                "drift_v2_model_expert_disagreement_rate": 0.33,
                "drift_v2_model_expert_macro_f1": 0.77,
                "drift_v2_window_status_code": 1,
                "drift_v2_window_index": 1,
                "labels": {
                    "run_id": "offline_demo_live_v2",
                    "mode": "debug",
                    "status": "warning",
                    "pipeline": "offline",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "window_metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "window_index": 0,
                        "start_event_index": 1,
                        "end_event_index": 20,
                        "phase": "A_baseline",
                        "token_distribution_jsd": 0.1,
                        "model_prediction_distribution_jsd": 0.11,
                        "target_distribution_jsd": 0.12,
                        "model_expert_disagreement_rate": 0.05,
                        "model_expert_macro_f1": 0.91,
                        "overall_status": "ok",
                        "metric_statuses": {
                            "token_distribution_jsd": "ok"
                        }
                    }
                ),
                json.dumps(
                    {
                        "window_index": 1,
                        "start_event_index": 21,
                        "end_event_index": 40,
                        "phase": "B_lexical",
                        "token_distribution_jsd": 0.2,
                        "model_prediction_distribution_jsd": 0.21,
                        "target_distribution_jsd": 0.22,
                        "model_expert_disagreement_rate": 0.15,
                        "model_expert_macro_f1": 0.81,
                        "overall_status": "warning",
                        "metric_statuses": {
                            "token_distribution_jsd": "warning"
                        }
                    }
                )
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "recent_events.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_index": 39,
                        "event_id": "event-39",
                        "created_at": "2026-06-29T00:00:00+00:00",
                        "model_prediction": "Stress",
                        "model_confidence": 0.81,
                        "expert_label": "Stress",
                        "expert_confidence": 0.93
                    },
                    {
                        "event_index": 40,
                        "event_id": "event-40",
                        "created_at": "2026-06-29T00:00:01+00:00",
                        "model_prediction": "Anxiety",
                        "model_confidence": 0.72,
                        "expert_label": "unlabeled",
                        "expert_confidence": None
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    (run_dir / ".active").write_text("offline_demo_live_v2\ndebug\n", encoding="utf-8")

    snapshot = load_metrics_snapshot(MonitoringSettings(runs_root=tmp_path))
    output = generate_metrics_text(snapshot).decode("utf-8")

    assert "drift_v2_visible_event_count{" in output
    assert 'pipeline="offline"' in output
    assert "drift_v2_window_token_distribution_jsd{" in output
    assert "drift_v2_window_metric_status_code{" in output
    assert "drift_v2_event_model_confidence{" not in output
    assert "drift_v2_event_expert_confidence{" not in output
    assert 'phase="B_lexical"' in output


def test_load_metrics_snapshot_ignores_inactive_run(tmp_path):
    run_dir = tmp_path / "offline_demo_live_v2_debug"
    run_dir.mkdir()
    (run_dir / "prom_metrics_latest.json").write_text(
        json.dumps(
            {
                "run_id": "offline_demo_live_v2",
                "drift_v2_runner_mode": "debug",
                "drift_v2_pipeline_mode": "offline",
                "drift_v2_visible_event_count": 120,
                "drift_v2_total_event_count": 120,
                "drift_v2_complete_window_count": 6,
                "drift_v2_window_status_code": 0,
                "drift_v2_window_index": 5,
                "labels": {
                    "run_id": "offline_demo_live_v2",
                    "mode": "debug",
                    "status": "ok",
                    "pipeline": "offline",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "window_metrics.jsonl").write_text("", encoding="utf-8")

    snapshot = load_metrics_snapshot(MonitoringSettings(runs_root=tmp_path))

    assert snapshot.active is False
    assert snapshot.file_present is False
    assert snapshot.window_count == 0
    assert snapshot.latest_payload["labels"]["status"] == "inactive"


def test_generate_metrics_text_exports_multiple_runs(tmp_path):
    offline = tmp_path / "offline_demo_live_v2_debug"
    online = tmp_path / "online_demo_live_v2_debug"
    offline.mkdir()
    online.mkdir()
    for run_dir, run_id, pipeline in (
        (offline, "offline_demo_live_v2", "offline"),
        (online, "online_demo_live_v2", "online"),
    ):
        (run_dir / "prom_metrics_latest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "drift_v2_runner_mode": "debug",
                    "drift_v2_pipeline_mode": pipeline,
                    "drift_v2_visible_event_count": 20,
                    "drift_v2_total_event_count": 20,
                    "drift_v2_complete_window_count": 1,
                    "drift_v2_token_distribution_jsd": 0.11,
                    "drift_v2_window_status_code": 0,
                    "drift_v2_window_index": 0,
                    "labels": {
                        "run_id": run_id,
                        "mode": "debug",
                        "status": "ok",
                        "pipeline": pipeline,
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "window_metrics.jsonl").write_text(
            json.dumps(
                {
                    "window_index": 0,
                    "start_event_index": 1,
                    "end_event_index": 20,
                    "phase": "A_baseline",
                    "token_distribution_jsd": 0.1,
                    "model_prediction_distribution_jsd": 0.11,
                    "target_distribution_jsd": 0.12,
                    "model_expert_disagreement_rate": 0.05,
                    "model_expert_macro_f1": 0.91,
                    "overall_status": "ok",
                    "metric_statuses": {"token_distribution_jsd": "ok"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "recent_events.json").write_text(json.dumps({"events": []}), encoding="utf-8")
        (run_dir / ".active").write_text(f"{run_id}\ndebug\n", encoding="utf-8")

    snapshots = load_metrics_snapshots(MonitoringSettings(runs_root=tmp_path))
    output = generate_metrics_text(snapshots).decode("utf-8")

    assert len(snapshots) == 2
    assert 'run_id="offline_demo_live_v2"' in output
    assert 'run_id="online_demo_live_v2"' in output
    assert 'pipeline="online"' in output


@pytest.mark.anyio
async def test_exporter_health_and_metrics_endpoints(tmp_path):
    run_dir = tmp_path / "offline_demo_live_v2_debug"
    run_dir.mkdir()
    (run_dir / "prom_metrics_latest.json").write_text(
        json.dumps(
            {
                "run_id": "offline_demo_live_v2",
                "drift_v2_runner_mode": "debug",
                "drift_v2_pipeline_mode": "offline",
                "drift_v2_visible_event_count": 20,
                "drift_v2_total_event_count": 120,
                "drift_v2_complete_window_count": 1,
                "drift_v2_window_status_code": 0,
                "drift_v2_window_index": 0,
                "labels": {
                    "run_id": "offline_demo_live_v2",
                    "mode": "debug",
                    "status": "ok",
                    "pipeline": "offline",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "window_metrics.jsonl").write_text("", encoding="utf-8")
    (run_dir / "recent_events.json").write_text(json.dumps({"events": []}), encoding="utf-8")
    (run_dir / ".active").write_text("offline_demo_live_v2\ndebug\n", encoding="utf-8")

    app = create_app(MonitoringSettings(runs_root=tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        health = await client.get("/health")
        metrics = await client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["active"] is True
    assert health.json()["file_present"] is True
    assert health.json()["active_runs"][0]["pipeline"] == "offline"
    assert "drift_v2_visible_event_count{" in metrics.text
