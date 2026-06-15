import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from src.api.app import _dashboard, app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


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
    assert "active_model" in payload


def test_retrain_endpoint_starts_background_job(client, monkeypatch):
    def fake_trigger(reload_predictor):
        _dashboard.retraining.state = "running"
        return True

    monkeypatch.setattr(_dashboard, "trigger_retraining", fake_trigger)

    response = client.post("/api/retrain")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "started"
