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
    resolve_window_metrics_path,
    select_replay_window,
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


def test_resolve_window_metrics_path_uses_sibling_by_default(tmp_path):
    metrics_path = tmp_path / "run" / "prom_metrics_latest.json"
    metrics_path.parent.mkdir()
    settings = MonitoringSettings()

    resolved = resolve_window_metrics_path(settings, metrics_path)

    assert resolved == metrics_path.parent / "window_metrics.jsonl"


def test_explicit_window_metrics_path_has_priority(tmp_path):
    metrics_path = tmp_path / "run" / "prom_metrics_latest.json"
    explicit = tmp_path / "manual_windows.jsonl"
    settings = MonitoringSettings(explicit_window_metrics_path=explicit)

    resolved = resolve_window_metrics_path(settings, metrics_path)

    assert resolved == explicit


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
                "drift_model_prediction_distribution_jsd": 0.12,
                "drift_model_confidence_mean": 0.81,
                "drift_target_distribution_jsd": 0.22,
                "drift_model_expert_disagreement_rate": 0.33,
                "drift_model_expert_macro_f1": 0.44,
                "drift_expert_confidence_mean": 0.93,
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
    assert "drift_model_prediction_distribution_jsd{" in output
    assert "drift_model_confidence_mean{" in output
    assert "drift_expert_confidence_mean{" in output
    assert 'run_id="run-1"' in output
    assert 'mode="debug"' in output
    assert 'phase="D_association"' in output
    assert 'status="critical"' in output
    assert "0.11" in output
    assert "drift_window_status_code{" in output
    assert "2.0" in output
    assert "drift_window_phase" not in output


def test_window_metrics_history_is_exported(tmp_path):
    run_dir = tmp_path / "20260621T192330Z_debug"
    run_dir.mkdir()
    (run_dir / "prom_metrics_latest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "drift_runner_mode": "debug",
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
    (run_dir / "window_metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "window_index": 0,
                        "phase": "A_baseline",
                        "overall_status": "ok",
                        "token_distribution_jsd": 0.1,
                        "model_prediction_distribution_jsd": 0.15,
                        "model_confidence_mean": 0.82,
                        "target_distribution_jsd": 0.2,
                        "model_expert_disagreement_rate": 0.3,
                        "model_expert_macro_f1": 0.8,
                        "expert_confidence_mean": 0.91,
                        "token_label_association_drift": 0.4,
                        "metric_statuses": {
                            "token_distribution_jsd": "ok",
                            "model_prediction_distribution_jsd": "ok",
                            "model_expert_macro_f1": "warning",
                        },
                    }
                ),
                json.dumps(
                    {
                        "window_index": 1,
                        "phase": "B_lexical",
                        "overall_status": "critical",
                        "token_distribution_jsd": 0.6,
                        "model_prediction_distribution_jsd": 0.65,
                        "model_confidence_mean": 0.52,
                        "target_distribution_jsd": 0.7,
                        "model_expert_disagreement_rate": 0.5,
                        "model_expert_macro_f1": 0.2,
                        "expert_confidence_mean": 0.84,
                        "token_label_association_drift": 1.4,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    snapshot = load_metrics_snapshot(MonitoringSettings(runs_root=tmp_path))
    output = generate_metrics_text(snapshot).decode("utf-8")

    assert snapshot.window_count == 2
    assert "drift_window_token_distribution_jsd{" in output
    assert "drift_window_model_prediction_distribution_jsd{" in output
    assert "drift_window_model_confidence_mean{" in output
    assert "drift_window_expert_confidence_mean{" in output
    assert 'window_index="0"' in output
    assert 'phase="A_baseline"' in output
    assert "drift_window_metric_status_code{" in output
    assert 'metric="model_expert_macro_f1"' in output
    assert "1.0" in output


def test_debug_summary_and_label_diagnostics_are_not_exported(tmp_path):
    run_dir = tmp_path / "20260621T192330Z_debug"
    run_dir.mkdir()
    (run_dir / "prom_metrics_latest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "drift_runner_mode": "debug",
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
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "accepted_samples": 71,
                "rejected_samples": 71,
                "accept_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "label_diagnostics.json").write_text(
        json.dumps(
            {
                "accept_rate_by_label": {"Normal": 0.375},
                "model_expert_disagreement_by_label": {
                    "Stress": {"disagreement_rate": 0.8571}
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_metrics_snapshot(MonitoringSettings(runs_root=tmp_path))
    output = generate_metrics_text(snapshot).decode("utf-8")

    assert "drift_synthetic_" not in output
    assert "drift_model_expert_disagreement_by_label" not in output
    assert 'label="Normal"' not in output
    assert 'label="Stress"' not in output


def test_replay_cursor_uses_step_and_loop():
    windows = [
        {"window_index": 0},
        {"window_index": 1},
        {"window_index": 2},
    ]
    settings = MonitoringSettings(
        replay_enabled=True,
        replay_step_seconds=10,
        replay_loop=True,
        replay_started_at=100.0,
    )

    selected = select_replay_window(windows, settings, now=125.0)

    assert selected == {"window_index": 2}


def test_replay_cursor_clamps_when_loop_disabled():
    windows = [
        {"window_index": 0},
        {"window_index": 1},
    ]
    settings = MonitoringSettings(
        replay_enabled=True,
        replay_step_seconds=10,
        replay_loop=False,
        replay_started_at=100.0,
    )

    selected = select_replay_window(windows, settings, now=155.0)

    assert selected == {"window_index": 1}


def test_replay_changes_latest_metrics_to_current_window(tmp_path):
    run_dir = tmp_path / "20260621T192330Z_debug"
    run_dir.mkdir()
    (run_dir / "prom_metrics_latest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "drift_runner_mode": "debug",
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
    (run_dir / "window_metrics.jsonl").write_text(
        json.dumps(
            {
                "window_index": 0,
                "phase": "A_baseline",
                "overall_status": "ok",
                "token_distribution_jsd": 0.123,
                "model_prediction_distribution_jsd": 0.234,
                "model_confidence_mean": 0.89,
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_metrics_snapshot(
        MonitoringSettings(
            runs_root=tmp_path,
            replay_enabled=True,
            replay_step_seconds=15,
            replay_started_at=0.0,
        )
    )
    output = generate_metrics_text(snapshot).decode("utf-8")

    assert "drift_token_distribution_jsd{" in output
    assert "drift_model_prediction_distribution_jsd{" in output
    assert 'phase="A_baseline"' in output
    assert "0.123" in output
    assert "0.234" in output


def test_malformed_window_metrics_are_reported_but_do_not_break_metrics(tmp_path):
    run_dir = tmp_path / "20260621T192330Z_debug"
    run_dir.mkdir()
    (run_dir / "prom_metrics_latest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "drift_runner_mode": "debug",
                "drift_model_expert_macro_f1": 0.7,
                "labels": {
                    "run_id": "run-1",
                    "mode": "debug",
                    "phase": "A_baseline",
                    "status": "ok",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "window_metrics.jsonl").write_text("{bad json", encoding="utf-8")

    snapshot = load_metrics_snapshot(MonitoringSettings(runs_root=tmp_path))
    output = generate_metrics_text(snapshot).decode("utf-8")

    assert snapshot.parse_errors
    assert "drift_model_expert_macro_f1" in output


@pytest.mark.anyio
async def test_exporter_health_and_metrics_endpoints(tmp_path):
    metrics_path = tmp_path / "prom_metrics_latest.json"
    metrics_path.write_text(
        json.dumps(
            {
                "drift_model_prediction_distribution_jsd": 0.2,
                "drift_model_confidence_mean": 0.82,
                "drift_model_expert_macro_f1": 0.7,
                "drift_expert_confidence_mean": 0.91,
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
    assert health.json()["window_count"] == 0
    assert health.json()["parse_errors"] == []
    assert metrics.status_code == 200
    assert "drift_model_expert_macro_f1{" in metrics.text
    assert "drift_model_prediction_distribution_jsd{" in metrics.text
    assert "drift_model_confidence_mean{" in metrics.text
    assert "drift_expert_confidence_mean{" in metrics.text
    assert 'run_id="run-2"' in metrics.text
    assert 'mode="full"' in metrics.text
    assert 'phase="A_baseline"' in metrics.text
    assert 'status="ok"' in metrics.text
    assert "0.7" in metrics.text
