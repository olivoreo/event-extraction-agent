import json
from urllib.parse import parse_qs, urlparse

import pytest

from event_extraction_agent import SourcePost, VKApiError, VKPostSource, VKSource, build_vk_post_url, parse_vk_source


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
        return FakeResponse(
            {
                "response": {
                    "items": [
                        {
                            "id": 10,
                            "owner_id": -123,
                            "date": 1_780_000_000,
                            "text": "  5 июня   в 18:00 пройдет лекция.  ",
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
    )

    posts = source.fetch_posts()

    assert len(posts) == 1
    assert isinstance(posts[0], SourcePost)
    assert posts[0].text == "5 июня в 18:00 пройдет лекция."
    assert "Не отправлять в агент" not in posts[0].text
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

    posts = VKSource(access_token="secret", sources=["first", "second"], posts_per_source_limit=1).fetch_posts()

    assert requested_sources == ["first", "second"]
    assert [post.text for post in posts] == ["Пост 1", "Пост 2"]
    assert [post.external_id for post in posts] == ["vk:wall-1_1", "vk:wall-2_2"]


def test_vk_source_raises_api_error(monkeypatch):
    def fake_urlopen(request, timeout):
        return FakeResponse({"error": {"error_code": 5, "error_msg": "User authorization failed"}})

    monkeypatch.setattr("event_extraction_agent.vk.urlopen", fake_urlopen)

    with pytest.raises(VKApiError) as error:
        VKSource(access_token="secret", sources=["club123"], posts_per_source_limit=1).fetch_posts()

    assert error.value.code == 5


def test_vk_source_requires_token_and_sources():
    with pytest.raises(ValueError, match="access token"):
        VKSource(access_token="", sources=["club123"])

    with pytest.raises(ValueError, match="at least one source"):
        VKSource(access_token="secret", sources=[])


def test_build_vk_post_url():
    assert build_vk_post_url(-123, 456) == "https://vk.com/wall-123_456"
