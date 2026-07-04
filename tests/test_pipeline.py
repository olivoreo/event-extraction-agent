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
    assert len(result.events) == 1
    assert result.events[0].event.title == "Лекция"
    assert result.events[0].outcome_index == 0
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
    assert len(loaded.events) == 1
    assert loaded.events[0].post.external_id == "post-1"
    assert loaded.outcomes[0].post.external_id == "post-1"


def test_pipeline_can_accumulate_existing_outcomes(tmp_path):
    cached_post = SourcePost(text="5 июня в 18:00 пройдет лекция.", external_id="post-1")
    old_post = SourcePost(text="6 июня в 19:00 пройдет концерт.", external_id="post-2")
    previous_result = BatchExtractionResult.from_outcomes(
        [
            _existing_outcome(cached_post, title="Лекция", start_at="2026-06-05T18:00:00+03:00"),
            _existing_outcome(old_post, title="Концерт", start_at="2026-06-06T19:00:00+03:00"),
        ]
    )
    previous_path = tmp_path / "previous.json"
    next_path = tmp_path / "next.json"
    previous_result.save_json(previous_path)
    client = FakeLLMClient(
        [
            json.dumps(
                {
                    "is_event": True,
                    "skip_reason": None,
                    "event": {
                        **_event_payload(),
                        "title": "Встреча",
                        "start_at": "2026-06-07T20:00:00+03:00",
                    },
                },
                ensure_ascii=False,
            )
        ]
    )

    result = ExtractionPipeline(
        source=FakeSource(
            [
                SourcePost(text="5 июня\nв 18:00 пройдет лекция.", external_id="post-1"),
                SourcePost(text="7 июня в 20:00 пройдет встреча.", external_id="post-3"),
            ]
        ),
        agent_config=ExtractionAgentConfig(main_client=client),
        previous_result_path=previous_path,
        save_result_path=next_path,
        accumulate_existing_outcomes=True,
    ).run()

    loaded = BatchExtractionResult.load_json(next_path)

    assert len(client.calls) == 1
    assert result.total == 3
    assert [outcome.post.external_id for outcome in loaded.outcomes] == ["post-1", "post-3", "post-2"]
    assert {item.event.title for item in loaded.events} == {"Лекция", "Встреча", "Концерт"}


def test_pipeline_prunes_missing_vk_outcomes_inside_current_window(tmp_path):
    cached_post = SourcePost(
        text="10 июля в 18:00 пройдет лекция.",
        published_at="2026-07-10T12:00:00+03:00",
        external_id="vk:wall-1_10",
    )
    missing_inside_window = SourcePost(
        text="8 июля в 19:00 пройдет концерт.",
        published_at="2026-07-08T12:00:00+03:00",
        external_id="vk:wall-1_8",
    )
    old_outside_window = SourcePost(
        text="4 июля в 20:00 пройдет встреча.",
        published_at="2026-07-04T12:00:00+03:00",
        external_id="vk:wall-1_4",
    )
    previous_result = BatchExtractionResult.from_outcomes(
        [
            _existing_outcome(cached_post, title="Лекция", start_at="2026-07-10T18:00:00+03:00"),
            _existing_outcome(missing_inside_window, title="Концерт", start_at="2026-07-08T19:00:00+03:00"),
            _existing_outcome(old_outside_window, title="Встреча", start_at="2026-07-04T20:00:00+03:00"),
        ]
    )
    previous_path = tmp_path / "previous.json"
    previous_result.save_json(previous_path)
    client = FakeLLMClient(
        [
            json.dumps(
                {
                    "is_event": True,
                    "skip_reason": None,
                    "event": {
                        **_event_payload(),
                        "title": "Кинопоказ",
                        "start_at": "2026-07-06T20:00:00+03:00",
                    },
                },
                ensure_ascii=False,
            )
        ]
    )

    result = ExtractionPipeline(
        source=FakeSource(
            [
                SourcePost(
                    text="10 июля\nв 18:00 пройдет лекция.",
                    published_at="2026-07-10T12:00:00+03:00",
                    external_id="vk:wall-1_10",
                ),
                SourcePost(
                    text="6 июля в 20:00 пройдет кинопоказ.",
                    published_at="2026-07-06T12:00:00+03:00",
                    external_id="vk:wall-1_6",
                ),
            ]
        ),
        agent_config=ExtractionAgentConfig(main_client=client),
        previous_result_path=previous_path,
        accumulate_existing_outcomes=True,
    ).run()

    assert len(client.calls) == 1
    assert result.cached == 1
    assert [outcome.post.external_id for outcome in result.outcomes] == [
        "vk:wall-1_10",
        "vk:wall-1_6",
        "vk:wall-1_4",
    ]
    assert {item.event.title for item in result.events} == {"Лекция", "Кинопоказ", "Встреча"}


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


def _existing_outcome(
    post: SourcePost,
    *,
    title: str = "Лекция",
    start_at: str = "2026-06-05T18:00:00+03:00",
) -> ExtractionOutcome:
    return ExtractionOutcome(
        status=ExtractionStatus.EXTRACTED,
        event=Event(
            **{
                **_event_payload(),
                "title": title,
                "start_at": start_at,
                "raw_text": post.raw_text_for_prompt(),
            }
        ),
        post=post,
        raw_llm_metadata={"model": "cached-test-model"},
    )
