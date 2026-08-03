import json
from urllib.parse import parse_qs, urlparse

import pytest

from event_extraction_agent import SourcePost, VKApiError, VKPostSource, VKSource
from event_extraction_agent.vk import build_vk_post_url, parse_vk_source


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_parse_vk_source_accepts_ids_domains_and_urls():
    assert parse_vk_source("-123") == VKPostSource(owner_id=-123)
    assert parse_vk_source("club123") == VKPostSource(owner_id=-123)
    assert parse_vk_source("public456") == VKPostSource(owner_id=-456)
    assert parse_vk_source("id789") == VKPostSource(owner_id=789)
    assert parse_vk_source("my_domain") == VKPostSource(domain="my_domain")
    assert parse_vk_source("https://vk.com/my_domain") == VKPostSource(domain="my_domain")
    assert parse_vk_source("https://vk.com/wall-123_456") == VKPostSource(owner_id=-123)


def test_vk_source_fetches_posts_and_maps_useful_metadata(monkeypatch):
    captured = []

    def fake_urlopen(request, timeout):
        captured.append((request.full_url, timeout))
        query = parse_qs(urlparse(request.full_url).query)
        if query.get("offset") == ["2"]:
            return FakeResponse({"response": {"items": []}})
        return FakeResponse(
            {
                "response": {
                    "items": [
                        {
                            "id": 10,
                            "owner_id": -123,
                            "date": 1_780_000_000,
                            "text": "  🎭 5 июня\n  в 18:00   пройдет лекция 😊.  ",
                            "attachments": [
                                {"type": "link", "link": {"title": "Не отправлять в агент"}},
                            ],
                        },
                        {
                            "id": 11,
                            "owner_id": -123,
                            "date": 1_780_000_100,
                            "text": "   ",
                            "attachments": [
                                {"type": "photo", "photo": {"text": "Только вложение"}},
                            ],
                        },
                    ],
                    "groups": [{"id": 123, "name": "Лекторий"}],
                }
            }
        )

    monkeypatch.setattr("event_extraction_agent.vk.urlopen", fake_urlopen)

    source = VKSource(
        access_token="secret",
        sources=["https://vk.com/club123"],
        posts_per_source_limit=2,
        timeout_seconds=7,
        rate_limit_per_second=None,
    )

    posts = source.fetch_posts()

    assert len(posts) == 1
    assert isinstance(posts[0], SourcePost)
    assert posts[0].text == "🎭 5 июня\n  в 18:00   пройдет лекция 😊."
    assert posts[0].raw_text == "5 июня в 18:00 пройдет лекция."
    assert "🎭" not in posts[0].raw_text
    assert "\n" not in posts[0].raw_text
    assert "Не отправлять в агент" not in posts[0].raw_text
    assert posts[0].source_name == "Лекторий"
    assert posts[0].source_url == "https://vk.com/wall-123_10"
    assert posts[0].external_id == "vk:wall-123_10"
    assert posts[0].published_at == "2026-05-28T20:26:40+00:00"

    url, timeout = captured[0]
    query = parse_qs(urlparse(url).query)
    assert timeout == 7
    assert query["access_token"] == ["secret"]
    assert query["owner_id"] == ["-123"]
    assert query["count"] == ["2"]
    assert query["filter"] == ["owner"]
    assert query["extended"] == ["1"]


def test_vk_source_fetches_more_raw_items_to_fill_text_post_limit(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        requests.append((query["count"][0], query["offset"][0]))
        if query["offset"] == ["0"]:
            items = [
                {"id": 1, "owner_id": -123, "text": "Первый текстовый пост"},
                {"id": 2, "owner_id": -123, "text": "", "attachments": [{"type": "photo"}]},
            ]
        else:
            items = [{"id": 3, "owner_id": -123, "text": "Второй текстовый пост"}]
        return FakeResponse({"response": {"items": items}})

    monkeypatch.setattr("event_extraction_agent.vk.urlopen", fake_urlopen)

    posts = VKSource(
        access_token="secret",
        sources=[-123],
        posts_per_source_limit=2,
        rate_limit_per_second=None,
    ).fetch_posts()

    assert requests == [("2", "0"), ("1", "2")]
    assert [post.external_id for post in posts] == ["vk:wall-123_1", "vk:wall-123_3"]


def test_vk_source_fetches_multiple_sources(monkeypatch):
    requested_sources = []

    def fake_urlopen(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        requested_sources.append(query.get("domain", query.get("owner_id", [""]))[0])
        owner_id = -1 if requested_sources[-1] == "first" else -2
        post_id = len(requested_sources)
        return FakeResponse(
            {
                "response": {
                    "items": [
                        {
                            "id": post_id,
                            "owner_id": owner_id,
                            "date": None,
                            "text": f"Пост {post_id}",
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr("event_extraction_agent.vk.urlopen", fake_urlopen)

    posts = VKSource(
        access_token="secret",
        sources=["first", "second"],
        posts_per_source_limit=1,
        rate_limit_per_second=None,
    ).fetch_posts()

    assert requested_sources == ["first", "second"]
    assert [post.raw_text for post in posts] == ["Пост 1", "Пост 2"]
    assert [post.external_id for post in posts] == ["vk:wall-1_1", "vk:wall-2_2"]


def test_vk_source_keeps_source_error_and_continues_with_available_sources(monkeypatch):
    def fake_urlopen(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        if query.get("domain") == ["broken"]:
            return FakeResponse({"error": {"error_code": 5, "error_msg": "User authorization failed"}})
        return FakeResponse(
            {
                "response": {
                    "items": [
                        {
                            "id": 1,
                            "owner_id": -2,
                            "date": None,
                            "text": "Доступный пост",
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr("event_extraction_agent.vk.urlopen", fake_urlopen)

    source = VKSource(
        access_token="secret",
        sources=["broken", "available"],
        posts_per_source_limit=1,
        rate_limit_per_second=None,
    )

    result = source.fetch_posts_with_errors()

    assert [post.raw_text for post in result.posts] == ["Доступный пост"]
    assert len(result.errors) == 1
    assert source.errors == result.errors
    assert result.errors[0].code == 5
    assert result.errors[0].source == VKPostSource(domain="broken")
    assert "source=broken" in str(result.errors[0])


def test_vk_source_can_fail_fast_on_source_error(monkeypatch):
    def fake_urlopen(request, timeout):
        return FakeResponse({"error": {"error_code": 5, "error_msg": "User authorization failed"}})

    monkeypatch.setattr("event_extraction_agent.vk.urlopen", fake_urlopen)

    source = VKSource(
        access_token="secret",
        sources=["club123"],
        posts_per_source_limit=1,
        continue_on_source_error=False,
        rate_limit_per_second=None,
    )

    with pytest.raises(VKApiError) as error:
        source.fetch_posts()

    assert error.value.code == 5
    assert error.value.source == VKPostSource(owner_id=-123)


def test_vk_source_retries_temporary_vk_errors(monkeypatch):
    calls = 0
    delays = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponse({"error": {"error_code": 6, "error_msg": "Too many requests per second"}})
        return FakeResponse(
            {
                "response": {
                    "items": [
                        {
                            "id": 1,
                            "owner_id": -123,
                            "date": None,
                            "text": "Пост после retry",
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr("event_extraction_agent.vk.urlopen", fake_urlopen)

    result = VKSource(
        access_token="secret",
        sources=["club123"],
        posts_per_source_limit=1,
        rate_limit_per_second=None,
        max_retries=1,
        retry_backoff_seconds=1.0,
        sleep=delays.append,
    ).fetch_posts_with_errors()

    assert calls == 2
    assert delays == [1.0]
    assert result.errors == []
    assert result.posts[0].raw_text == "Пост после retry"


def test_vk_source_rate_limits_api_requests(monkeypatch):
    now = [0.0]
    delays = []

    def fake_sleep(delay):
        delays.append(delay)
        now[0] += delay

    def fake_urlopen(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        owner_id = -1 if query.get("domain") == ["first"] else -2
        return FakeResponse(
            {
                "response": {
                    "items": [
                        {
                            "id": abs(owner_id),
                            "owner_id": owner_id,
                            "date": None,
                            "text": "Пост",
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr("event_extraction_agent.vk.urlopen", fake_urlopen)

    VKSource(
        access_token="secret",
        sources=["first", "second"],
        posts_per_source_limit=1,
        rate_limit_per_second=20,
        sleep=fake_sleep,
        monotonic=lambda: now[0],
    ).fetch_posts()

    assert delays == [0.05]


def test_vk_source_requires_token_and_sources():
    with pytest.raises(ValueError, match="access token"):
        VKSource(access_token="", sources=["club123"])

    with pytest.raises(ValueError, match="at least one source"):
        VKSource(access_token="secret", sources=[])


def test_build_vk_post_url():
    assert build_vk_post_url(-123, 456) == "https://vk.com/wall-123_456"
