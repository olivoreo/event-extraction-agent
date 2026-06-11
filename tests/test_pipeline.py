import json

import pytest

from event_extraction_agent import (
    BatchExtractionSettings,
    ExtractionAgent,
    ExtractionPipeline,
    ExtractionStatus,
    SourcePost,
    extract_from_source,
)


class FakeSource:
    def __init__(self, posts: list[SourcePost] | object):
        self.posts = posts
        self.calls = 0

    def fetch_posts(self):
        self.calls += 1
        return self.posts


class FakeLLMClient:
    model = "pipeline-fake-model"

    def __init__(self, contents: list[str]):
        self.contents = contents
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return self.contents[index]


def test_pipeline_fetches_posts_and_runs_batch_extraction():
    posts = [
        SourcePost(text="5 июня в 18:00 пройдет лекция.", external_id="post-1"),
        SourcePost(text="Просто информационный пост.", external_id="post-2"),
    ]
    source = FakeSource(posts)
    agent = ExtractionAgent(
        llm_client=FakeLLMClient(
            [
                json.dumps(
                    {
                        "is_event": True,
                        "skip_reason": None,
                        "event": _event_payload(),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"is_event": False, "skip_reason": "not_event_announcement", "event": None},
                    ensure_ascii=False,
                ),
            ]
        )
    )

    result = ExtractionPipeline(agent=agent, source=source).run()

    assert source.calls == 1
    assert result.total == 2
    assert [outcome.post for outcome in result.outcomes] == posts
    assert [outcome.status for outcome in result.outcomes] == [
        ExtractionStatus.EXTRACTED,
        ExtractionStatus.SKIPPED,
    ]


def test_extract_from_source_uses_batch_settings():
    post = SourcePost(text="5 июня в 18:00 пройдет лекция.", external_id="post-1")
    source = FakeSource([post, post])
    agent = ExtractionAgent(
        llm_client=FakeLLMClient(
            [
                json.dumps(
                    {
                        "is_event": True,
                        "skip_reason": None,
                        "event": _event_payload(),
                    },
                    ensure_ascii=False,
                )
            ]
        )
    )

    result = extract_from_source(
        source=source,
        agent=agent,
        batch_settings=BatchExtractionSettings(skip_duplicates=True),
    )

    assert result.total == 2
    assert result.extracted == 1
    assert result.skipped == 1
    assert result.outcomes[1].errors[0].code == "duplicate_post"


def test_pipeline_rejects_invalid_source_return_shape():
    source = FakeSource(["not a SourcePost"])
    agent = ExtractionAgent(llm_client=FakeLLMClient([]))

    with pytest.raises(TypeError, match=r"list\[SourcePost\]"):
        ExtractionPipeline(agent=agent, source=source).run()


def _event_payload() -> dict[str, object]:
    return {
        "title": "Лекция",
        "description": "Открытая лекция",
        "start_at": "2026-06-05T18:00:00+03:00",
        "end_at": None,
        "timezone": "Europe/Moscow",
        "city": "Волгоград",
        "venue_name": "Лекторий",
        "address": None,
        "event_type": "EducationEvent",
        "attendance_type": "OfflineEventAttendanceMode",
        "event_status": "EventScheduled",
        "language": "ru",
        "source_name": None,
        "source_url": None,
        "raw_text": None,
        "relevant_roles": ["Participant"],
        "industries": None,
        "skills": None,
        "price_text": "free",
        "target_audience_text": None,
    }
