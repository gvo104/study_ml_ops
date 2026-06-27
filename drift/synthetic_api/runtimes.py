import asyncio
import json
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from drift.synthetic_api.prompts import (
    build_expert_system_prompt,
    build_expert_user_prompt,
    build_generator_system_prompt,
    build_generator_user_prompt,
    build_repair_prompt,
)
from drift.synthetic_api.schemas import (
    ExpertRequest,
    ExpertResponse,
    GeneratorRequest,
    GeneratorResponse,
    RunRequest,
    RunResponse,
)


class RuntimeErrorWithContext(RuntimeError):
    """Raised when the runtime cannot produce a valid structured response."""


class LlmRuntime(Protocol):
    backend_name: str
    loaded: bool

    async def run(self, payload: RunRequest) -> RunResponse:
        ...


class MockLlmRuntime:
    backend_name = "mock"
    loaded = True

    def __init__(self, delay_seconds: float = 0.0):
        self.delay_seconds = delay_seconds
        self.call_count = 0
        self.active_calls = 0
        self.max_concurrency = 0

    async def run(self, payload: RunRequest) -> RunResponse:
        self.call_count += 1
        self.active_calls += 1
        self.max_concurrency = max(self.max_concurrency, self.active_calls)

        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)

            if isinstance(payload, GeneratorRequest):
                return GeneratorResponse(
                    role="generator",
                    text=self._generate_text(payload),
                    target_label=payload.target_label,
                    phase=payload.phase,
                )

            return self._classify_text(payload)
        finally:
            self.active_calls -= 1

    def _generate_text(self, payload: GeneratorRequest) -> str:
        phase_templates = {
            "A": {
                "Anxiety": "I feel anxious, restless, and my heart races at night.",
                "Bipolar": "My mood swings fast and I barely sleep during highs.",
                "Depression": "I feel empty, tired, and nothing seems worth doing.",
                "Normal": "I had a regular day, slept well, and feel okay overall.",
                "Personality disorder": "My relationships feel unstable and I react intensely.",
                "Stress": "Work pressure is building up and I cannot relax lately.",
                "Suicidal": "I keep thinking that life is pointless and I want to disappear.",
            },
            "B": {
                "Anxiety": "My nerves are buzzing, I keep spiraling, and nighttime feels tense.",
                "Bipolar": "I am wired for days, talking fast, and sleep feels optional.",
                "Depression": "Everything feels flat, heavy, and pointless even when nothing happened.",
                "Normal": "Today was ordinary; I rested, handled errands, and felt steady.",
                "Personality disorder": "Small shifts in closeness hit hard and I snap before I can slow down.",
                "Stress": "Deadlines are stacking up, my shoulders stay tight, and I cannot switch off.",
                "Suicidal": "I do not want to keep going and disappearing sounds like relief.",
            },
            "C": {
                "Anxiety": "I feel anxious, restless, and my heart races at night.",
                "Bipolar": "My mood swings fast and I barely sleep during highs.",
                "Depression": "I feel empty, tired, and nothing seems worth doing.",
                "Normal": "I had a regular day, slept well, and feel okay overall.",
                "Personality disorder": "My relationships feel unstable and I react intensely.",
                "Stress": "Work pressure is building up and I cannot relax lately.",
                "Suicidal": "I keep thinking that life is pointless and I want to disappear.",
            },
            "D": {
                "Anxiety": "I keep saying I am calm, but I scan every sound and my chest tightens.",
                "Bipolar": "I call it productive, yet I have slept two hours and cannot slow my ideas.",
                "Depression": "I tell people I am fine, but I feel numb and cannot care about anything.",
                "Normal": "I stayed busy with normal errands, slept well, and felt balanced afterward.",
                "Personality disorder": "I seem calm until a delayed reply makes me panic and lash out.",
                "Stress": "I say I am just busy, but the overload is constant and I cannot relax.",
                "Suicidal": "I tell everyone I am fine, but I keep imagining not being here anymore.",
            },
        }
        return phase_templates[payload.phase][payload.target_label]

    def _classify_text(self, payload: ExpertRequest) -> ExpertResponse:
        label_keywords = {
            "Anxiety": ["anxious", "panic", "heart races", "restless", "spiraling", "chest tightens", "scan every sound"],
            "Bipolar": ["mood swings", "highs", "barely sleep", "sleep feels optional", "slept two hours", "cannot slow my ideas"],
            "Depression": ["empty", "tired", "worth doing", "flat", "heavy", "numb", "cannot care"],
            "Normal": ["regular day", "slept well", "okay overall", "felt steady", "felt balanced"],
            "Personality disorder": ["unstable", "react intensely", "relationships", "snap", "lash out", "delayed reply"],
            "Stress": ["work pressure", "cannot relax", "stress", "deadlines", "switch off", "overload"],
            "Suicidal": ["pointless", "disappear", "suicidal", "not being here", "do not want to keep going"],
        }
        text_lower = payload.text.lower()

        for label in payload.allowed_labels:
            for keyword in label_keywords.get(label, []):
                if keyword.lower() in text_lower:
                    return ExpertResponse(
                        role="expert",
                        label=label,
                        confidence=0.91,
                        reason=f"Matched deterministic keyword: {keyword}",
                    )

        return ExpertResponse(
            role="expert",
            label=payload.allowed_labels[0],
            confidence=0.55,
            reason="No deterministic keyword matched; defaulted to first allowed label.",
        )


class SerializedRuntime:
    def __init__(self, runtime: LlmRuntime):
        self.runtime = runtime
        self._lock = asyncio.Lock()

    @property
    def backend_name(self) -> str:
        return self.runtime.backend_name

    @property
    def loaded(self) -> bool:
        return self.runtime.loaded

    async def run(self, payload: RunRequest) -> RunResponse:
        async with self._lock:
            return await self.runtime.run(payload)


class LlamaCppRuntime:
    backend_name = "llama_cpp"

    def __init__(
        self,
        model_path: Path,
        temperature: float = 0.3,
        max_retries: int = 2,
        max_tokens: int = 256,
        context_window: int = 4096,
        gpu_layers: int = -1,
    ):
        self.model_path = Path(model_path)
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.gpu_layers = gpu_layers
        self._model = self._load_model()
        self.loaded = True

    def _load_model(self):
        if not self.model_path.exists():
            raise RuntimeErrorWithContext(
                f"GGUF model not found at '{self.model_path}'."
            )

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeErrorWithContext(
                "llama_cpp is not installed. Install llama-cpp-python before "
                "using SYNTHETIC_LLM_BACKEND=llama_cpp."
            ) from exc

        return Llama(
            model_path=str(self.model_path),
            chat_format="chatml",
            n_ctx=self.context_window,
            n_gpu_layers=self.gpu_layers,
            verbose=False,
        )

    async def run(self, payload: RunRequest) -> RunResponse:
        if isinstance(payload, GeneratorRequest):
            data = await self._complete_json(
                system_prompt=build_generator_system_prompt(),
                user_prompt=build_generator_user_prompt(payload),
            )
            return self._build_generator_response(payload, data)

        data = await self._complete_json(
            system_prompt=build_expert_system_prompt(),
            user_prompt=build_expert_user_prompt(payload),
        )
        return self._build_expert_response(data)

    async def _complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        last_error: Exception | None = None
        current_prompt = user_prompt
        raw_response = ""

        for _ in range(self.max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self._model.create_chat_completion,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": current_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_response = response["choices"][0]["message"]["content"]
                return json.loads(raw_response)
            except Exception as exc:  # pragma: no cover - exercised manually
                last_error = exc
                current_prompt = build_repair_prompt(raw_response or str(exc))

        raise RuntimeErrorWithContext(
            f"Failed to produce valid JSON after retries: {last_error}"
        )

    def _build_generator_response(
        self,
        payload: GeneratorRequest,
        data: dict,
    ) -> GeneratorResponse:
        normalized = dict(data)
        normalized["role"] = "generator"
        normalized["target_label"] = payload.target_label
        normalized["phase"] = payload.phase

        try:
            return GeneratorResponse(**normalized)
        except ValidationError as exc:
            raise RuntimeErrorWithContext(
                f"Generator response validation failed: {exc}"
            ) from exc

    def _build_expert_response(self, data: dict) -> ExpertResponse:
        normalized = dict(data)
        normalized["role"] = "expert"
        confidence = normalized.get("confidence")
        if isinstance(confidence, (int, float)) and confidence > 1:
            normalized["confidence"] = confidence / 100.0

        try:
            return ExpertResponse(**normalized)
        except ValidationError as exc:
            raise RuntimeErrorWithContext(
                f"Expert response validation failed: {exc}"
            ) from exc


def build_runtime(
    backend: str,
    model_path: Path,
    temperature: float,
    max_retries: int,
    max_tokens: int = 256,
    context_window: int = 4096,
    gpu_layers: int = -1,
) -> SerializedRuntime:
    if backend == "mock":
        return SerializedRuntime(MockLlmRuntime())
    if backend == "llama_cpp":
        return SerializedRuntime(
            LlamaCppRuntime(
                model_path=model_path,
                temperature=temperature,
                max_retries=max_retries,
                max_tokens=max_tokens,
                context_window=context_window,
                gpu_layers=gpu_layers,
            )
        )

    raise RuntimeErrorWithContext(
        "Unsupported SYNTHETIC_LLM_BACKEND. Use 'mock' or 'llama_cpp'."
    )
