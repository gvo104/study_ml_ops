import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import src.api.app as api_app
import src.api.dashboard as api_dashboard


class DummyPredictor:
    metadata = {
        "feature_schema_version": "test-schema",
        "accuracy": 0.91,
        "macro_f1": 0.9,
        "created_at": "2026-06-15T00:00:00+00:00",
    }

    def predict(self, text):
        return {
            "prediction": "normal",
            "confidence": 0.82,
            "probabilities": {
                "normal": 0.82,
                "stress": 0.18,
            },
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_app, "load_predictor", lambda: DummyPredictor())
    monkeypatch.setattr(
        api_app,
        "ensure_dataset",
        lambda: (True, "Dataset is ready for the web UI."),
    )
    monkeypatch.setattr(
        api_dashboard,
        "list_mlflow_runs",
        lambda: {
            "enabled": True,
            "connected": True,
            "uri": "http://mlflow:5000",
            "experiment_name": "mental_health_classification",
            "error": None,
            "experiments": [
                {
                    "experiment_id": "1",
                    "name": "mental_health_classification",
                    "lifecycle_stage": "active",
                    "artifact_location": "s3://ml-team/mlflow-artifacts",
                }
            ],
            "runs": [
                {
                    "run_id": "abc123",
                    "experiment_id": "1",
                    "run_name": "xgboost_baseline",
                    "status": "FINISHED",
                    "start_time": 1781500000000,
                    "end_time": 1781500100000,
                    "artifact_uri": "s3://ml-team/mlflow-artifacts/1/abc123/artifacts",
                    "metrics": {"accuracy": 0.91, "macro_f1": 0.9},
                    "params": {"model.name": "xgboost"},
                    "tags": {"mlflow.runName": "xgboost_baseline"},
                }
            ],
        },
    )

    api_app._dashboard.recent_predictions.clear()
    api_app._dashboard.recent_text_lengths.clear()
    api_app._dashboard.drift_notifications.clear()
    api_app._dashboard.prediction_counter = 0
    api_app._dashboard.retraining.state = "idle"
    api_app._dashboard.retraining.message = "Ready to start retraining."

    with TestClient(api_app.app) as test_client:
        yield test_client

    api_app.clear_runtime()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_inference_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Inference Console" in response.text


def test_experiments_page_renders(client):
    response = client.get("/experiments")
    assert response.status_code == 200
    assert "Experiments & Retraining" in response.text


def test_predict_tracks_recent_predictions(client):
    response = client.post(
        "/predict",
        json={"text": "I feel anxious and exhausted lately."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "anomaly_flags" in payload

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    summary = dashboard.json()
    assert summary["recent_predictions"]


def test_experiments_summary_contains_catalog(client):
    response = client.get("/api/experiments")
    assert response.status_code == 200
    payload = response.json()
    assert payload["experiment_configs"]
    assert payload["tracking"]["connected"] is True
    assert payload["mlflow_runs"][0]["run_name"] == "xgboost_baseline"
    assert payload["mlflow_experiments"][0]["name"] == "mental_health_classification"
    assert payload["active_model"]["feature_schema_version"] == "test-schema"


def test_retrain_endpoint_starts_background_job(client, monkeypatch):
    def fake_trigger(reload_predictor, config_path=None, data_path=None, reason=None):
        api_app._dashboard.retraining.state = "running"
        return True

    monkeypatch.setattr(api_app._dashboard, "trigger_retraining", fake_trigger)

    response = client.post("/api/retrain")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "started"
