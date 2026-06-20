import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient

from drift.synthetic_api.app import create_app
from drift.synthetic_api.config import SyntheticApiSettings
from drift.synthetic_api.schemas import GeneratorRequest
from drift.synthetic_api.runtimes import MockLlmRuntime, SerializedRuntime


@pytest.fixture
def app():
    return create_app(SyntheticApiSettings(backend="mock"))


@pytest.mark.anyio
async def test_synthetic_health_endpoint(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "backend": "mock", "loaded": True}


@pytest.mark.anyio
async def test_generator_response_contract(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/llm/run",
                json={
                    "role": "generator",
                    "phase": "A",
                    "target_label": "Anxiety",
                    "constraints": {"length": "short"},
                },
            )
    assert response.status_code == 200
    payload = response.json()
    assert payload["role"] == "generator"
    assert payload["target_label"] == "Anxiety"
    assert payload["phase"] == "A"
    assert "anxious" in payload["text"].lower()


@pytest.mark.anyio
async def test_expert_response_contract(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/llm/run",
                json={
                    "role": "expert",
                    "text": "I feel anxious and restless and my heart races at night.",
                    "allowed_labels": ["Anxiety", "Stress"],
                },
            )
    assert response.status_code == 200
    payload = response.json()
    assert payload["role"] == "expert"
    assert payload["label"] == "Anxiety"
    assert payload["confidence"] == 0.91


@pytest.mark.anyio
async def test_expert_rejects_synthetic_metadata(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/llm/run",
                json={
                    "role": "expert",
                    "text": "I feel overwhelmed at work.",
                    "allowed_labels": ["Stress"],
                    "phase": "B",
                    "target_label": "Stress",
                },
            )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_unknown_role_rejected(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/llm/run",
                json={"role": "unknown"},
            )
    assert response.status_code == 422


def test_serialized_runtime_prevents_parallel_execution():
    runtime = MockLlmRuntime(delay_seconds=0.01)
    serialized = SerializedRuntime(runtime)

    async def run_twice():
        await asyncio.gather(
            serialized.run(
                GeneratorRequest(
                    role="generator",
                    phase="A",
                    target_label="Stress",
                    constraints={"length": "short"},
                )
            ),
            serialized.run(
                GeneratorRequest(
                    role="generator",
                    phase="B",
                    target_label="Anxiety",
                    constraints={"length": "short"},
                )
            ),
        )

    asyncio.run(run_twice())
    assert runtime.call_count == 2
    assert runtime.max_concurrency == 1
