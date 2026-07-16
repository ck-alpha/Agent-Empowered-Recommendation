"""Minimal Ollama structured-output transport with explicit failure semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol

import requests


class OllamaTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:14b"
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 180.0
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 4096
    num_predict: int = 1024
    keep_alive: str = "5m"


@dataclass(frozen=True)
class OllamaResponse:
    content: str
    usage: Mapping[str, Any] = field(default_factory=dict)


class StructuredLLMClient(Protocol):
    def generate_structured(self, *, system: str, prompt: str, schema: Mapping[str, Any]) -> OllamaResponse:
        ...


class OllamaStructuredClient:
    def __init__(self, config: OllamaConfig | None = None, session: requests.Session | None = None):
        self.config = config or OllamaConfig()
        self.session = session or requests.Session()
        # This project targets a loopback Ollama service. The host environment may
        # define HTTP_PROXY without NO_PROXY, which would incorrectly route local
        # inference through an external proxy and surface as HTTP 502.
        if session is None and self.config.base_url.startswith(("http://127.0.0.1", "http://localhost")):
            self.session.trust_env = False

    def generate_structured(self, *, system: str, prompt: str, schema: Mapping[str, Any]) -> OllamaResponse:
        payload = {
            "model": self.config.model,
            "system": system,
            "prompt": prompt,
            "format": dict(schema),
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
            },
        }
        endpoint = f"{self.config.base_url.rstrip('/')}/api/generate"
        try:
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=(self.config.connect_timeout_seconds, self.config.read_timeout_seconds),
            )
        except requests.RequestException as exc:
            raise OllamaTransportError(
                f"Cannot reach Ollama at {self.config.base_url}; start `ollama serve` and verify model {self.config.model}: {exc}"
            ) from exc
        if response.status_code != 200:
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text
            raise OllamaTransportError(f"Ollama HTTP {response.status_code}: {detail}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaTransportError("Ollama returned a non-JSON HTTP response") from exc
        content = payload.get("response")
        if not isinstance(content, str) or not content.strip():
            raise OllamaTransportError("Ollama returned an empty structured response")
        usage_keys = (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
            "done_reason",
        )
        usage = {key: payload[key] for key in usage_keys if key in payload}
        return OllamaResponse(content=content, usage=usage)

    def health(self) -> Dict[str, Any]:
        endpoint = f"{self.config.base_url.rstrip('/')}/api/tags"
        try:
            response = self.session.get(
                endpoint,
                timeout=(self.config.connect_timeout_seconds, min(10.0, self.config.read_timeout_seconds)),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            return {"healthy": False, "error": str(exc), "model": self.config.model}
        names = [str(model.get("name", "")) for model in payload.get("models", [])]
        return {"healthy": self.config.model in names, "models": names, "model": self.config.model}
