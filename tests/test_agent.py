import json

import pytest

from event_extraction_agent import BatchExtractionSettings, ExtractionAgent, ExtractionStatus, SourcePost


RAW_TEXT = (
    "Мы ждем вас в Амфитеатр 5 июня в 18:00 на конкурс "
    "«Мисс и Мистер Студенчество Волгограда - 2026»."
)


class FakeLLMClient:
    def __init__(self, content: str | list[str]):
        self.contents = content if isinstance(content, list) else [content]
        self.model = "fake-model"
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return self.contents[index]


def test_extract_returns_outcome_with_event():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at="2026-06-05T18:00:00+03:00"),
            },
            ensure_ascii=False,
        )
    )
    post = SourcePost(
        text=RAW_TEXT,
        source_name="Центр",
        source_url="https://vk.com/wall-1_1",
        published_at="2026-06-02T10:57:56+00:00",
        external_id="vk:wall-1_1",
    )

    outcome = ExtractionAgent(llm_client=client).extract(post)

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.start_at.isoformat() == "2026-06-05T18:00:00+03:00"
    assert outcome.event.source_name == "Центр"
    assert outcome.event.source_url == "https://vk.com/wall-1_1"
    assert outcome.event.raw_text == RAW_TEXT
    assert outcome.post == post
    assert outcome.raw_llm_metadata == {"llm_model": "fake-model", "external_id": "vk:wall-1_1"}


def test_extract_skips_non_event_without_event():
    client = FakeLLMClient(
        json.dumps({"is_event": False, "skip_reason": "safety_instruction", "event": None}, ensure_ascii=False)
    )

    outcome = ExtractionAgent(llm_client=client).extract(SourcePost(text="Как действовать при опасности."))

    assert outcome.status == ExtractionStatus.SKIPPED
    assert outcome.event is None
    assert outcome.errors[0].code == "safety_instruction"


def test_extract_returns_llm_error_for_malformed_json():
    outcome = ExtractionAgent(llm_client=FakeLLMClient("не json")).extract(
        SourcePost(text="5 июня в 18:00 состоится концерт.")
    )

    assert outcome.status == ExtractionStatus.LLM_ERROR
    assert outcome.event is None
    assert outcome.errors[0].code == "llm_error"


def test_extract_marks_event_without_date_as_invalid():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at=None),
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(SourcePost(text="Скоро пройдет встреча."))

    assert outcome.status == ExtractionStatus.INVALID
    assert outcome.event is None
    assert any(error.field == "start_at" and error.code == "missing_start_at" for error in outcome.errors)


def test_extract_repairs_start_at_from_russian_date_and_time():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at=None),
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="Приглашаем на семейный мастер-класс 1 июня в 18.30 в Амфитеатре.",
            published_at="2026-05-30T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.start_at.isoformat() == "2026-06-01T18:30:00+03:00"


def test_event_type_refinement_uses_fallback_client():
    main_client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at="2026-06-05T18:00:00+03:00"),
                    "event_type": "SocialEvent",
                },
            },
            ensure_ascii=False,
        )
    )
    fallback_client = FakeLLMClient(json.dumps({"event_type": "CompetitionEvent"}, ensure_ascii=False))

    outcome = ExtractionAgent(llm_client=main_client, event_type_llm_client=fallback_client).extract(
        SourcePost(text=RAW_TEXT)
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.event_type == "CompetitionEvent"
    assert len(main_client.calls) == 1
    assert len(fallback_client.calls) == 1
    assert outcome.raw_llm_metadata is not None
    assert outcome.raw_llm_metadata["event_type_refined"] is True


def test_extract_event_raises_for_non_extracted_outcome():
    agent = ExtractionAgent(llm_client=FakeLLMClient("не json"))

    with pytest.raises(ValueError):
        agent.extract_event(SourcePost(text="5 июня в 18:00 состоится концерт."))


def test_extract_many_preserves_order_with_mixed_outcomes_and_duplicates():
    client = FakeLLMClient(
        [
            json.dumps(
                {
                    "is_event": True,
                    "skip_reason": None,
                    "event": _event_payload(start_at="2026-06-05T18:00:00+03:00"),
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {"is_event": False, "skip_reason": "not_event_announcement", "event": None},
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "is_event": True,
                    "skip_reason": None,
                    "event": _event_payload(start_at=None),
                },
                ensure_ascii=False,
            ),
        ]
    )
    posts = [
        SourcePost(text=RAW_TEXT, external_id="vk:1"),
        SourcePost(text=RAW_TEXT, external_id="vk:1"),
        SourcePost(text="Просто новость без события.", external_id="vk:2"),
        SourcePost(text="Скоро пройдет встреча.", external_id="vk:3"),
    ]

    outcomes = ExtractionAgent(llm_client=client).extract_many(posts)

    assert [outcome.post for outcome in outcomes] == posts
    assert [outcome.status for outcome in outcomes] == [
        ExtractionStatus.EXTRACTED,
        ExtractionStatus.SKIPPED,
        ExtractionStatus.SKIPPED,
        ExtractionStatus.INVALID,
    ]
    assert outcomes[1].errors[0].code == "duplicate_post"
    assert outcomes[2].errors[0].code == "not_event_announcement"
    assert outcomes[3].errors[0].code == "missing_start_at"
    assert len(client.calls) == 3


def test_extract_batch_returns_summary_and_applies_error_limit():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at=None),
            },
            ensure_ascii=False,
        )
    )
    posts = [
        SourcePost(text="Скоро пройдет встреча.", external_id="vk:1"),
        SourcePost(text="5 июня в 18:00 состоится концерт.", external_id="vk:2"),
    ]

    result = ExtractionAgent(llm_client=client).extract_batch(
        posts,
        settings=BatchExtractionSettings(max_errors=1),
    )

    assert result.total == 2
    assert result.invalid == 1
    assert result.skipped == 1
    assert result.error_count == 1
    assert result.error_limit_reached is True
    assert [outcome.status for outcome in result.outcomes] == [ExtractionStatus.INVALID, ExtractionStatus.SKIPPED]
    assert result.outcomes[1].errors[0].code == "error_limit_reached"
    assert len(client.calls) == 1


def _event_payload(start_at: str | None) -> dict[str, object]:
    return {
        "title": "Мисс и Мистер Студенчество",
        "description": "Конкурс для студентов",
        "start_at": start_at,
        "end_at": None,
        "timezone": "Europe/Moscow",
        "city": "Волгоград",
        "venue_name": "Амфитеатр",
        "address": None,
        "event_type": "CompetitionEvent",
        "attendance_type": "OfflineEventAttendanceMode",
        "event_status": "EventScheduled",
        "language": "ru",
        "source_name": None,
        "source_url": None,
        "raw_text": None,
        "relevant_roles": ["Participant", "Spectator"],
        "industries": ["Fashion", "PerformingArts"],
        "skills": None,
        "price_text": "free",
        "target_audience_text": None,
        "seniority_level": "legacy",
    }
