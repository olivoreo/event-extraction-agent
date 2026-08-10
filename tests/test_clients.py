import io
import json
import urllib.error

import pytest

from event_extraction_agent import GroqChatClient, GroqDailyRateLimitError, OllamaChatClient
from scripts.run_golden_extraction import _build_client


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


def test_refinement_groq_client_uses_shared_api_key():
    client = _build_client(
        {
            "GROQ_API_KEY": "shared-secret",
            "REFINEMENT_GROQ_API_KEY": "ignored-secret",
            "REFINEMENT_LLM_PROVIDER": "groq",
        },
        prefix="REFINEMENT_",
    )

    assert isinstance(client, GroqChatClient)
    assert client.api_key == "shared-secret"


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


def test_groq_client_uses_strict_json_schema_for_gpt_oss(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    GroqChatClient(api_key="secret", model="openai/gpt-oss-120b").complete_with_schema(
        "system", "user", schema
    )

    assert captured["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "event_extraction",
            "strict": True,
            "schema": schema,
        },
    }


def test_groq_client_falls_back_to_json_object_for_model_without_schema_support(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    GroqChatClient(api_key="secret", model="llama-3.1-8b-instant").complete_with_schema(
        "system", "user", {"type": "object"}
    )

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


@pytest.mark.parametrize("limit_name", ["RPM", "TPM", "ITPM", "OTPM"])
def test_groq_client_waits_for_minute_limits_without_spending_retries(monkeypatch, limit_name):
    calls = 0
    sleeps = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            message = f"Rate limit reached on tokens per minute ({limit_name}). Please try again in 1s."
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                hdrs={"Retry-After": "1"},
                fp=io.BytesIO(json.dumps({"error": {"message": message}}).encode()),
            )
        return FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("event_extraction_agent.agent.time_module.sleep", sleeps.append)

    content = GroqChatClient(api_key="secret", max_retries=0).complete("system", "user")

    assert content == '{"ok": true}'
    assert calls == 2
    assert sleeps == [1.25]


@pytest.mark.parametrize("limit_name", ["RPD", "TPD"])
def test_groq_client_reports_daily_limits_without_waiting(monkeypatch, limit_name):
    sleeps = []

    def fake_urlopen(request, timeout):
        message = f"Rate limit reached on tokens per day ({limit_name})."
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            hdrs={"Retry-After": "3600"},
            fp=io.BytesIO(json.dumps({"error": {"message": message}}).encode()),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("event_extraction_agent.agent.time_module.sleep", sleeps.append)

    with pytest.raises(GroqDailyRateLimitError, match=limit_name):
        GroqChatClient(api_key="secret", max_retries=3).complete("system", "user")

    assert sleeps == []


def test_groq_client_uses_token_reset_when_retry_after_is_missing(monkeypatch):
    calls = 0
    sleeps = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                hdrs={"x-ratelimit-reset-tokens": "2m3.5s"},
                fp=io.BytesIO(b"{}"),
            )
        return FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("event_extraction_agent.agent.time_module.sleep", sleeps.append)

    GroqChatClient(api_key="secret", max_retries=0).complete("system", "user")

    assert sleeps == [123.75]


def test_groq_client_uses_safe_fallback_for_unknown_429(monkeypatch):
    calls = 0
    sleeps = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                hdrs={},
                fp=io.BytesIO(b"{}"),
            )
        return FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("event_extraction_agent.agent.time_module.sleep", sleeps.append)

    GroqChatClient(api_key="secret", max_retries=0).complete("system", "user")

    assert sleeps == [60.25]
