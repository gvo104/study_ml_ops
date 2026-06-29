import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import src.api.app as api_app


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
