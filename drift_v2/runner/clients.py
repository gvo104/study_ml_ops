from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class ApiClientError(RuntimeError):
    """Raised when an upstream API call fails."""


@dataclass
class SyntheticApiClient:
    base_url: str
    timeout_seconds: float = 30.0

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/llm/run", payload)

    def label(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/llm/run", payload)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(self.base_url, path, payload, self.timeout_seconds)


@dataclass
class PredictApiClient:
    base_url: str
    timeout_seconds: float = 30.0

    def predict(self, text: str) -> dict[str, Any]:
        return _post_json(
            self.base_url,
            "/predict",
            {"text": text},
            self.timeout_seconds,
        )


def _post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ApiClientError(f"Request to {url} failed: {exc}") from exc
