import io
import json
import urllib.error

import pytest

from event_extraction_agent import GroqChatClient, OllamaChatClient


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_client_builds_expected_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"message": {"content": '{"ok": true}'}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    content = OllamaChatClient(model="local-model", host="http://localhost:11434").complete("system", "user")

    assert content == '{"ok": true}'
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "local-model"
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_groq_client_requires_api_key():
    with pytest.raises(ValueError):
        GroqChatClient(api_key="")


def test_groq_client_builds_expected_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers["Authorization"]
        return FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    content = GroqChatClient(api_key="secret", model="groq-model", max_retries=0).complete("system", "user")

    assert content == '{"ok": true}'
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["payload"]["model"] == "groq-model"
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_groq_client_reports_http_errors(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b"bad payload"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Groq request failed"):
        GroqChatClient(api_key="secret", max_retries=0).complete("system", "user")
