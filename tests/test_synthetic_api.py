import asyncio
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient

from drift_v2.synthetic_api.app import create_app
from drift_v2.synthetic_api.config import SyntheticApiSettings
from drift_v2.synthetic_api.schemas import GeneratorRequest
from drift_v2.synthetic_api.runtimes import (
    LlamaCppRuntime,
    MockLlmRuntime,
    RuntimeErrorWithContext,
    SerializedRuntime,
    build_runtime,
)


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
async def test_synthetic_frontend_is_served(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/")
    assert response.status_code == 200
    assert "Synthetic API Console" in response.text
    assert "Run Generator" in response.text
    assert "Run Expert" in response.text


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
async def test_generator_phase_d_uses_context_inversion(app):
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/llm/run",
                json={
                    "role": "generator",
                    "phase": "D",
                    "target_label": "Depression",
                    "constraints": {"length": "short"},
                },
            )
    payload = response.json()
    text_lower = payload["text"].lower()
    assert response.status_code == 200
    assert "fine" in text_lower
    assert "numb" in text_lower


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


def test_build_runtime_rejects_unknown_backend():
    with pytest.raises(RuntimeErrorWithContext):
        build_runtime(
            backend="unknown",
            model_path=Path("unused.gguf"),
            temperature=0.3,
            max_retries=2,
        )


def test_llama_cpp_runtime_requires_existing_model_file():
    with pytest.raises(RuntimeErrorWithContext, match="GGUF model not found"):
        build_runtime(
            backend="llama_cpp",
            model_path=Path("missing.gguf"),
            temperature=0.3,
            max_retries=2,
        )


def test_llama_cpp_generator_response_is_canonicalized():
    runtime = object.__new__(LlamaCppRuntime)
    payload = GeneratorRequest(
        role="generator",
        phase="A",
        target_label="Anxiety",
        constraints={"length": "short"},
    )

    response = runtime._build_generator_response(
        payload,
        {
            "role": "patient",
            "text": "I feel anxious and restless.",
            "target_label": "Other",
            "phase": "Z",
        },
    )

    assert response.role == "generator"
    assert response.target_label == "Anxiety"
    assert response.phase == "A"


def test_llama_cpp_expert_response_is_canonicalized():
    runtime = object.__new__(LlamaCppRuntime)

    response = runtime._build_expert_response(
        {
            "role": "labeler",
            "label": "Stress",
            "confidence": 0.82,
            "reason": "Work-related overload.",
        }
    )

    assert response.role == "expert"
    assert response.label == "Stress"


def test_llama_cpp_expert_confidence_percent_is_normalized():
    runtime = object.__new__(LlamaCppRuntime)

    response = runtime._build_expert_response(
        {
            "role": "labeler",
            "label": "Anxiety",
            "confidence": 95,
            "reason": "Clear anxiety symptoms.",
        }
    )

    assert response.role == "expert"
    assert response.confidence == 0.95
