import json

import pytest

from event_extraction_agent import (
    BatchExtractionSettings,
    BatchExtractionResult,
    Event,
    ExtractionAgentConfig,
    ExtractionOutcome,
    ExtractionPipeline,
    ExtractionStatus,
    SourcePost,
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
    client = FakeLLMClient(
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

    result = ExtractionPipeline(source=source, agent_config=ExtractionAgentConfig(main_client=client)).run()

    assert source.calls == 1
    assert result.total == 2
    assert [outcome.post for outcome in result.outcomes] == posts
    assert [outcome.status for outcome in result.outcomes] == [
        ExtractionStatus.EXTRACTED,
        ExtractionStatus.SKIPPED,
    ]


def test_pipeline_uses_batch_settings():
    post = SourcePost(text="5 июня в 18:00 пройдет лекция.", external_id="post-1")
    source = FakeSource([post, post])
    config = ExtractionAgentConfig(
        main_client=FakeLLMClient(
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
        ),
    )

    result = ExtractionPipeline(
        source=source,
        agent_config=config,
        batch_settings=BatchExtractionSettings(skip_duplicates=True),
    ).run()

    assert result.total == 2
    assert result.extracted == 1
    assert result.skipped == 1
    assert result.outcomes[1].errors[0].code == "duplicate_post"


def test_pipeline_can_run_incrementally_with_existing_outcomes():
    post = SourcePost(text="5 июня в 18:00 пройдет лекция.", external_id="post-1")
    existing_outcome = _existing_outcome(post)
    source = FakeSource([SourcePost(text="5 июня\nв 18:00 пройдет лекция.", external_id="post-1")])
    config = ExtractionAgentConfig(main_client=FakeLLMClient([]))

    result = ExtractionPipeline(agent_config=config, source=source, existing_outcomes=[existing_outcome]).run()

    assert result.total == 1
    assert result.cached == 1
    assert result.processed == 0
    assert result.outcomes[0].raw_llm_metadata is not None
    assert result.outcomes[0].raw_llm_metadata["incremental_cached"] is True


def test_pipeline_can_load_previous_result_and_save_next_result(tmp_path):
    post = SourcePost(text="5 июня в 18:00 пройдет лекция.", external_id="post-1")
    previous_result = BatchExtractionResult.from_outcomes([_existing_outcome(post)])
    previous_path = tmp_path / "previous.json"
    next_path = tmp_path / "next.json"
    previous_result.save_json(previous_path)

    result = ExtractionPipeline(
        source=FakeSource([SourcePost(text="5 июня\nв 18:00 пройдет лекция.", external_id="post-1")]),
        agent_config=ExtractionAgentConfig(main_client=FakeLLMClient([])),
        previous_result_path=previous_path,
        save_result_path=next_path,
    ).run()

    loaded = BatchExtractionResult.load_json(next_path)

    assert result.cached == 1
    assert loaded.cached == 1
    assert loaded.outcomes[0].post.external_id == "post-1"


def test_pipeline_rejects_invalid_source_return_shape():
    source = FakeSource(["not a SourcePost"])
    config = ExtractionAgentConfig(main_client=FakeLLMClient([]))

    with pytest.raises(TypeError, match=r"list\[SourcePost\]"):
        ExtractionPipeline(agent_config=config, source=source).run()


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


def _existing_outcome(post: SourcePost) -> ExtractionOutcome:
    return ExtractionOutcome(
        status=ExtractionStatus.EXTRACTED,
        event=Event(**{**_event_payload(), "raw_text": post.raw_text_for_prompt()}),
        post=post,
        raw_llm_metadata={"model": "cached-test-model"},
    )
