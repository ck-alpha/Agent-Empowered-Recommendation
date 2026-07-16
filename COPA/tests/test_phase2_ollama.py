import json

import pytest

from copa.phase2.ollama import OllamaConfig, OllamaStructuredClient, OllamaTransportError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, endpoint, json, timeout):
        self.calls.append((endpoint, json, timeout))
        return self.response

    def get(self, endpoint, timeout):
        return self.response


def test_ollama_client_sends_json_schema_and_deterministic_options():
    session = FakeSession(
        FakeResponse(payload={"response": '{"ok":true}', "prompt_eval_count": 12, "eval_count": 4})
    )
    client = OllamaStructuredClient(OllamaConfig(), session=session)
    response = client.generate_structured(system="system", prompt="prompt", schema={"type": "object"})
    endpoint, payload, timeout = session.calls[0]
    assert endpoint.endswith("/api/generate")
    assert payload["format"] == {"type": "object"}
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert payload["options"]["seed"] == 42
    assert response.usage["prompt_eval_count"] == 12


def test_ollama_http_error_is_actionable():
    session = FakeSession(FakeResponse(status_code=404, payload={"error": "model not found"}))
    client = OllamaStructuredClient(session=session)
    with pytest.raises(OllamaTransportError, match="model not found"):
        client.generate_structured(system="s", prompt="p", schema={"type": "object"})
