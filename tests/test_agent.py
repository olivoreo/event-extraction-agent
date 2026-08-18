import json

import pytest

from event_extraction_agent import (
    BatchExtractionSettings,
    ExtractionAgent,
    ExtractionAgentConfig,
    ExtractionStatus,
    GroqDailyRateLimitError,
    SourcePost,
)
from event_extraction_agent.prompts import SYSTEM_PROMPT, build_extraction_prompt


RAW_TEXT = (
    "Мы ждем вас в Амфитеатр 5 июня в 18:00 на конкурс "
    "«Мисс и Мистер Студенчество Волгограда - 2026»."
)


class FakeLLMClient:
    def __init__(self, content: str | BaseException | list[str | BaseException], model: str = "fake-model"):
        self.contents = content if isinstance(content, list) else [content]
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        content = self.contents[index]
        if isinstance(content, BaseException):
            raise content
        return content


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
    assert outcome.event.start_at.isoformat() == "2026-06-05T18:00:00"
    assert outcome.event.source_name == "Центр"
    assert outcome.event.source_url == "https://vk.com/wall-1_1"
    assert outcome.event.raw_text == RAW_TEXT
    assert outcome.post == post
    assert outcome.raw_llm_metadata is not None
    assert outcome.raw_llm_metadata["llm_model"] == "fake-model"
    assert outcome.raw_llm_metadata["external_id"] == "vk:wall-1_1"
    assert outcome.raw_llm_metadata["active_stage"] == "main_extraction"
    assert outcome.raw_llm_metadata["event_type_refinement"] == "disabled"
    assert outcome.raw_llm_metadata["llm_attempts"] == [
        {"stage": "main_extraction", "model": "fake-model", "success": True}
    ]


def test_extract_sends_raw_text_to_llm_when_present():
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
        text="🎭 5 июня\nв 18:00 пройдет лекция 😊.",
        raw_text="5 июня в 18:00 пройдет лекция.",
    )

    outcome = ExtractionAgent(llm_client=client).extract(post)

    assert "5 июня в 18:00 пройдет лекция." in client.calls[0][1]
    assert "🎭" not in client.calls[0][1]
    assert outcome.event is not None
    assert outcome.event.raw_text == "5 июня в 18:00 пройдет лекция."


def test_extraction_prompt_omits_locally_injected_output_fields():
    prompt = build_extraction_prompt(
        raw_text="5 июня в 18:00 пройдет лекция.",
        source_name="Лекторий",
        source_url="https://vk.com/wall-1_1",
    )

    schema = prompt.split("skip_reason values:", maxsplit=1)[0]

    assert '"raw_text"' not in schema
    assert '"source_name"' not in schema
    assert '"source_url"' not in schema
    assert '"event_status"' not in schema
    assert "Личные истории, мнения, интервью или планы автора" in SYSTEM_PROMPT
    assert "самодостаточная сухая выжимка" in SYSTEM_PROMPT
    assert "ссылки для регистрации, покупки билетов" in SYSTEM_PROMPT
    assert "только на русском языке" in SYSTEM_PROMPT
    assert "их может быть несколько" in SYSTEM_PROMPT
    assert "все и только подходящие способы участия" in SYSTEM_PROMPT
    assert "metadata:" in prompt
    assert "raw_text: 5 июня в 18:00 пройдет лекция." in prompt


def test_agent_can_take_main_client_from_config():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at="2026-06-05T18:00:00+03:00"),
            },
            ensure_ascii=False,
        ),
        model="configured-client",
    )

    outcome = ExtractionAgent(config=ExtractionAgentConfig(main_client=client)).extract(SourcePost(text=RAW_TEXT))

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert len(client.calls) == 1
    assert outcome.raw_llm_metadata is not None
    assert outcome.raw_llm_metadata["main_model"] == "configured-client"


def test_agent_requires_llm_client():
    with pytest.raises(ValueError, match="llm_client is required"):
        ExtractionAgent()


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


def test_extract_drops_skills_when_llm_returns_string_instead_of_list():
    payload = _event_payload(start_at="2026-06-05T18:00:00+03:00")
    payload["skills"] = "music"
    client = FakeLLMClient(
        json.dumps({"is_event": True, "skip_reason": None, "event": payload}, ensure_ascii=False)
    )

    outcome = ExtractionAgent(llm_client=client).extract(SourcePost(text=RAW_TEXT))

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.skills is None


def test_extract_marks_daily_groq_limit_without_agent_retries():
    client = FakeLLMClient(GroqDailyRateLimitError("HTTP 429: tokens per day (TPD) exhausted"))

    outcome = ExtractionAgent(
        llm_client=client,
        config=ExtractionAgentConfig(max_retries=3),
    ).extract(SourcePost(text=RAW_TEXT))

    assert outcome.status == ExtractionStatus.LLM_ERROR
    assert outcome.errors[0].code == "daily_rate_limit_exceeded"
    assert len(client.calls) == 1


def test_extract_batch_stops_calling_client_after_daily_groq_limit():
    client = FakeLLMClient(GroqDailyRateLimitError("HTTP 429: tokens per day (TPD) exhausted"))
    posts = [
        SourcePost(text=f"{day} июня в 18:00 состоится концерт {day}.")
        for day in range(5, 8)
    ]

    result = ExtractionAgent(llm_client=client).extract_batch(posts)

    assert len(client.calls) == 1
    assert result.llm_errors == 3
    assert all(outcome.errors[0].code == "daily_rate_limit_exceeded" for outcome in result.outcomes)


def test_extract_keeps_unknown_price_as_none():
    payload = _event_payload(start_at="2026-06-05T18:00:00+03:00")
    payload["price_text"] = None
    client = FakeLLMClient(
        json.dumps({"is_event": True, "skip_reason": None, "event": payload}, ensure_ascii=False)
    )

    outcome = ExtractionAgent(llm_client=client).extract(SourcePost(text=RAW_TEXT))

    assert outcome.event is not None
    assert outcome.event.price_text is None


def test_extract_allows_event_without_start_at_when_end_at_exists():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at=None),
                    "end_at": "2026-06-20T00:00:00",
                },
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(SourcePost(text="Скоро пройдет встреча."))

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.start_at is None
    assert outcome.event.end_at is not None


def test_extract_rejects_event_without_start_or_end_at():
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
    assert outcome.errors[0].code == "missing_event_date"


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
    assert outcome.event.start_at.isoformat() == "2026-06-01T18:30:00"


def test_extract_repairs_start_at_from_russian_date_without_time_as_midnight():
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
            text="Дистанционный конкурс пройдет 3 августа. Участники отправляют ролики онлайн.",
            published_at="2026-06-24T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.start_at.isoformat() == "2026-08-03T00:00:00"


def test_extract_keeps_llm_start_at_even_when_text_has_different_milestone_date():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at="2026-06-18T00:00:00"),
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="Результаты дистанционного конкурса будут объявлены 3 августа.",
            published_at="2026-06-24T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.start_at.isoformat() == "2026-06-18T00:00:00"


def test_extract_repairs_deadline_used_as_application_start_at():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at="2025-06-23T00:00:00"),
                    "end_at": None,
                },
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="Успейте подать заявку на Международную Премию #МЫВМЕСТЕ до 23 июня.",
            published_at="2025-06-03T13:10:07+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.start_at is None
    assert outcome.event.end_at is not None
    assert outcome.event.end_at.isoformat() == "2025-06-23T00:00:00"


def test_extract_drops_end_at_inferred_from_duration():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at="2026-02-07T10:00:00"),
                    "end_at": "2026-02-07T11:30:00",
                },
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(text="7 февраля в 10:00 пройдет IT заряд. Продолжительность: 1,5 часа.")
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.end_at is None


def test_extract_normalizes_short_timezone_offset_before_duration_check():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at="2026-02-07T10:00:00"),
                    "end_at": "2026-02-07T11:30:00+03",
                },
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(text="7 февраля в 10:00 пройдет IT заряд. Продолжительность: 1,5 часа.")
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.end_at is None


def test_extract_uses_unknown_timezone_without_local_context():
    payload = _event_payload(start_at=None)
    payload["timezone"] = None
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": payload,
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="Результаты дистанционного конкурса будут объявлены 3 августа.",
            published_at="2026-06-24T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.timezone == "unknown"


def test_extract_repairs_consecutive_date_range_with_shared_time():
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
            text="19 и 20 июня в 17:00 пройдет фестиваль народных промыслов.",
            published_at="2026-06-10T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.start_at.isoformat() == "2026-06-19T17:00:00"
    assert outcome.event.end_at is not None
    assert outcome.event.end_at.isoformat() == "2026-06-20T00:00:00"


def test_extract_repairs_missing_end_at_from_dash_date_range():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at="2026-08-21T18:00:00"),
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="Фестиваль русского рока «Наследие» пройдет 21–22 августа. Начало 21 августа в 18:00.",
            published_at="2026-08-01T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.start_at.isoformat() == "2026-08-21T18:00:00"
    assert outcome.event.end_at is not None
    assert outcome.event.end_at.isoformat() == "2026-08-22T00:00:00"


def test_extract_repairs_missing_end_at_from_from_to_date_range():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at="2026-08-21T10:00:00"),
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="Форум пройдет с 21 по 23 августа.",
            published_at="2026-08-01T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.end_at is not None
    assert outcome.event.end_at.isoformat() == "2026-08-23T00:00:00"


def test_extract_repairs_missing_end_at_across_month_boundary():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at="2026-08-31T09:00:00"),
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="Форум пройдет 31 августа — 1 сентября.",
            published_at="2026-08-01T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.end_at is not None
    assert outcome.event.end_at.isoformat() == "2026-09-01T00:00:00"


def test_extract_does_not_turn_nonconsecutive_dates_into_continuous_range():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at="2026-08-21T18:00:00"),
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="Показы пройдут 21 и 25 августа.",
            published_at="2026-08-01T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.end_at is None


def test_extract_splits_non_contiguous_repeated_dates_into_events():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at="2026-05-09T18:00:00"),
                    "end_at": "2026-05-11T18:00:00",
                    "title": "Спектакль «Василий Тёркин»",
                },
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="9 и 11 мая в 18:00 на сцене амфитеатра пройдет спектакль «Василий Тёркин».",
            published_at="2026-04-30T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.events is not None
    assert [event.start_at.isoformat() for event in outcome.events] == [
        "2026-05-09T18:00:00",
        "2026-05-11T18:00:00",
    ]
    assert [event.end_at for event in outcome.events] == [None, None]


def test_extract_keeps_consecutive_listed_dates_as_single_event():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at="2026-01-03T14:00:00"),
                    "end_at": "2026-01-05T17:00:00",
                    "title": "Новогодние молодёжные гуляния",
                },
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="3, 4, 5 января приглашаем на Новогодние гуляния. Время: 14:00 - 17:00.",
            published_at="2025-12-30T12:00:00+00:00",
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.events is None
    assert outcome.event is not None
    assert outcome.event.start_at.isoformat() == "2026-01-03T14:00:00"
    assert outcome.event.end_at is not None
    assert outcome.event.end_at.isoformat() == "2026-01-05T17:00:00"


def test_extract_omits_events_for_single_event_list_from_llm():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at="2026-06-05T18:00:00"),
                "events": [_event_payload(start_at="2026-06-05T18:00:00")],
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(SourcePost(text=RAW_TEXT))

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.events is None


def test_extract_accepts_events_without_duplicate_event_from_llm():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": None,
                "events": [
                    _event_payload(start_at="2026-05-09T18:00:00", title="Спектакль"),
                    _event_payload(start_at="2026-05-11T18:00:00", title="Спектакль"),
                ],
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(SourcePost(text=RAW_TEXT))

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.start_at.isoformat() == "2026-05-09T18:00:00"
    assert outcome.events is not None
    assert [event.start_at.isoformat() for event in outcome.events] == [
        "2026-05-09T18:00:00",
        "2026-05-11T18:00:00",
    ]


def test_extract_strips_timezone_offset_from_llm_datetimes():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at="2026-06-19T17:00:00+04:00"),
                    "end_at": "2026-06-19T21:00:00Z",
                },
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(llm_client=client).extract(SourcePost(text="19 июня в 17:00 пройдет фестиваль."))

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.start_at.isoformat() == "2026-06-19T17:00:00"
    assert outcome.event.end_at is not None
    assert outcome.event.end_at.isoformat() == "2026-06-19T21:00:00"


def test_extract_retries_once_when_end_at_is_before_start_at():
    client = FakeLLMClient(
        [
            json.dumps(
                {
                    "is_event": True,
                    "skip_reason": None,
                    "event": {
                        **_event_payload(start_at="2026-06-05T18:00:00"),
                        "end_at": "2026-06-01T18:00:00",
                    },
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "is_event": True,
                    "skip_reason": None,
                    "event": _event_payload(start_at="2026-06-05T18:00:00"),
                },
                ensure_ascii=False,
            ),
        ]
    )

    outcome = ExtractionAgent(llm_client=client).extract(SourcePost(text="5 июня в 18:00 пройдет встреча."))

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.end_at is None
    assert len(client.calls) == 2
    assert "end_at не может быть раньше start_at" in client.calls[1][1]
    assert outcome.raw_llm_metadata is not None
    assert outcome.raw_llm_metadata["active_stage"] == "invalid_date_repair"
    assert [attempt["stage"] for attempt in outcome.raw_llm_metadata["llm_attempts"]] == [
        "main_extraction",
        "invalid_date_repair",
    ]


def test_extract_overrides_giveaway_result_as_non_announcement():
    client = FakeLLMClient(_event_response("2026-06-26T19:00:00+03:00"))

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(text="ИТОГИ РОЗЫГРЫША. Поздравляем победителя! 26 июня Амфитеатр.")
    )

    assert outcome.status == ExtractionStatus.SKIPPED
    assert outcome.event is None
    assert outcome.errors[0].code == "not_event_announcement"
    assert len(client.calls) == 0


def test_extract_overrides_past_report_as_non_announcement():
    client = FakeLLMClient(_event_response("2026-06-22T18:00:00+03:00"))

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(text="22 июня мы смотрели фильм в амфитеатре. Спасибо каждому, кто разделил этот вечер.")
    )

    assert outcome.status == ExtractionStatus.SKIPPED
    assert outcome.event is None
    assert outcome.errors[0].code == "past_event_report"
    assert len(client.calls) == 0


def test_extract_overrides_sostoyalsya_report_as_non_announcement():
    client = FakeLLMClient(_event_response("2026-06-12T00:00:00+03:00"))

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(text="Сегодня состоялся праздничный концерт. Под открытым небом собрались тысячи людей.")
    )

    assert outcome.status == ExtractionStatus.SKIPPED
    assert outcome.event is None
    assert outcome.errors[0].code == "past_event_report"
    assert len(client.calls) == 0


def test_extract_overrides_visitor_count_report_as_non_announcement():
    client = FakeLLMClient(_event_response("2025-03-24T00:00:00"))

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(text="Форум «Образование — 2025» прошел на Волгоград Арене. Посетителями стали школьники.")
    )

    assert outcome.status == ExtractionStatus.SKIPPED
    assert outcome.event is None
    assert outcome.errors[0].code == "past_event_report"
    assert len(client.calls) == 0


def test_extract_overrides_admission_ad_without_date_as_non_announcement():
    client = FakeLLMClient(_event_response("2026-06-01T00:00:00"))

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text="Университет приглашает выпускников подать документы на программу высшего образования уже этим летом."
        )
    )

    assert outcome.status == ExtractionStatus.SKIPPED
    assert outcome.event is None
    assert outcome.errors[0].code == "not_event_announcement"
    assert len(client.calls) == 0


def test_extract_skips_personal_film_story_when_llm_marks_it_non_announcement():
    client = FakeLLMClient(
        json.dumps({"is_event": False, "skip_reason": "not_event_announcement", "event": None}, ensure_ascii=False)
    )

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text=(
                "Александра Егунова рассказала, как любит проводить свободное время. "
                "Друзья в шутку называют меня Супердевушкой. "
                "Мы всей компанией ждем выхода фильма «Супергерл» и договорились сходить "
                "на премьерный показ в «Мори Синема». "
                "А как вы предпочитаете отдыхать после насыщенной недели?"
            )
        )
    )

    assert outcome.status == ExtractionStatus.SKIPPED
    assert outcome.event is None
    assert outcome.errors[0].code == "not_event_announcement"
    assert len(client.calls) == 1


def test_extract_rejects_announcement_with_guest_opinion_without_date():
    client = FakeLLMClient(_event_response(None, title="Встреча с режиссером"))

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text=(
                "Приглашаем на встречу с режиссером после показа. "
                "Гость рассказал, как готовился к фильму и ответит на вопросы зрителей."
            )
        )
    )

    assert outcome.status == ExtractionStatus.INVALID
    assert outcome.event is None
    assert outcome.errors[0].code == "missing_event_date"
    assert len(client.calls) == 1


def test_extract_overrides_cancellation_update_as_non_announcement():
    client = FakeLLMClient(_event_response("2026-06-01T00:00:00", title="Дискотека выпускников — 2026"))

    outcome = ExtractionAgent(llm_client=client).extract(
        SourcePost(
            text=(
                "ВНИМАНИЕ! ОТМЕНА! Дискотека выпускников — 2026 отменена "
                "в связи с неблагоприятными погодными условиями. "
                "Приносим свои извинения за доставленные неудобства."
            )
        )
    )

    assert outcome.status == ExtractionStatus.SKIPPED
    assert outcome.event is None
    assert outcome.errors[0].code == "not_event_announcement"
    assert len(client.calls) == 0


def test_event_type_refinement_uses_refinement_client():
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
    refinement_client = FakeLLMClient(json.dumps({"event_type": "CompetitionEvent"}, ensure_ascii=False))

    outcome = ExtractionAgent(
        llm_client=main_client,
        refinement_llm_client=refinement_client,
        config=ExtractionAgentConfig(use_event_type_refinement=True),
    ).extract(
        SourcePost(text=RAW_TEXT)
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.event_type == "CompetitionEvent"
    assert len(main_client.calls) == 1
    assert len(refinement_client.calls) == 1
    assert outcome.raw_llm_metadata is not None
    assert outcome.raw_llm_metadata["event_type_refined"] is True
    assert outcome.raw_llm_metadata["event_type_refinement"] == "completed"
    assert outcome.raw_llm_metadata["llm_attempts"] == [
        {"stage": "main_extraction", "model": "fake-model", "success": True},
        {"stage": "event_type_refinement", "model": "fake-model", "success": True},
    ]


def test_title_description_refinement_fixes_unrelated_title():
    main_client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at="2026-06-05T18:00:00+03:00", title="Концерт на набережной"),
                    "description": "Концерт.",
                },
            },
            ensure_ascii=False,
        ),
        model="main-model",
    )
    refinement_client = FakeLLMClient(
        json.dumps(
            {
                "title": "По щучьему велению",
                "description": "Семейный спектакль по мотивам русской сказки.",
            },
            ensure_ascii=False,
        ),
        model="refinement-model",
    )

    outcome = ExtractionAgent(
        llm_client=main_client,
        refinement_llm_client=refinement_client,
        config=ExtractionAgentConfig(use_title_description_refinement=True),
    ).extract(
        SourcePost(
            text='5 июня в 18:00 состоится семейный спектакль "По щучьему велению" для детей и родителей.'
        )
    )

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.title == "По щучьему велению"
    assert outcome.event.description == "Семейный спектакль по мотивам русской сказки."
    assert len(refinement_client.calls) == 1
    assert outcome.raw_llm_metadata is not None
    assert outcome.raw_llm_metadata["title_description_refinement"] == "completed"
    assert outcome.raw_llm_metadata["llm_attempts"] == [
        {"stage": "main_extraction", "model": "main-model", "success": True},
        {"stage": "title_description_refinement", "model": "refinement-model", "success": True},
    ]


def test_title_description_refinement_runs_for_matching_title_and_improves_description():
    main_client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at="2026-06-05T18:00:00+03:00", title="По щучьему велению"),
                    "description": "Приходите на невероятный спектакль!",
                },
            },
            ensure_ascii=False,
        )
    )
    refinement_client = FakeLLMClient(
        json.dumps(
            {
                "title": "По щучьему велению",
                "description": "Семейный спектакль по мотивам русской сказки для детей и родителей.",
            },
            ensure_ascii=False,
        )
    )

    outcome = ExtractionAgent(
        llm_client=main_client,
        refinement_llm_client=refinement_client,
        config=ExtractionAgentConfig(use_title_description_refinement=True),
    ).extract(
        SourcePost(
            text='5 июня в 18:00 состоится семейный спектакль "По щучьему велению" для детей и родителей.'
        )
    )

    assert outcome.event is not None
    assert outcome.event.description == "Семейный спектакль по мотивам русской сказки для детей и родителей."
    assert len(refinement_client.calls) == 1


def test_event_type_refinement_uses_config_refinement_client():
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
        ),
        model="main-model",
    )
    refinement_client = FakeLLMClient(
        json.dumps({"event_type": "CompetitionEvent"}, ensure_ascii=False),
        model="refinement-model",
    )

    outcome = ExtractionAgent(
        config=ExtractionAgentConfig(
            main_client=main_client,
            refinement_client=refinement_client,
            use_event_type_refinement=True,
        )
    ).extract(SourcePost(text=RAW_TEXT))

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.event_type == "CompetitionEvent"
    assert len(refinement_client.calls) == 1
    assert outcome.raw_llm_metadata is not None
    assert outcome.raw_llm_metadata["refinement_model"] == "refinement-model"


def test_config_disables_event_type_refinement():
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
    refinement_client = FakeLLMClient(json.dumps({"event_type": "CompetitionEvent"}, ensure_ascii=False))

    outcome = ExtractionAgent(
        llm_client=main_client,
        refinement_llm_client=refinement_client,
        config=ExtractionAgentConfig(use_event_type_refinement=False),
    ).extract(SourcePost(text=RAW_TEXT))

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert outcome.event is not None
    assert outcome.event.event_type == "SocialEvent"
    assert len(refinement_client.calls) == 0
    assert outcome.raw_llm_metadata is not None
    assert outcome.raw_llm_metadata["event_type_refinement"] == "disabled"
    assert outcome.raw_llm_metadata["refinement_model"] is None


def test_agent_config_controls_prompt_datetime_and_agent_retries():
    client = FakeLLMClient(
        [
            RuntimeError("temporary error"),
            json.dumps(
                {
                    "is_event": True,
                    "skip_reason": None,
                    "event": _event_payload(start_at="2026-06-05T18:00:00+03:00"),
                },
                ensure_ascii=False,
            ),
        ]
    )

    outcome = ExtractionAgent(
        llm_client=client,
        config=ExtractionAgentConfig(
            current_datetime="2026-06-04T12:00:00+03:00",
            max_retries=1,
        ),
    ).extract(SourcePost(text=RAW_TEXT))

    assert outcome.status == ExtractionStatus.EXTRACTED
    assert len(client.calls) == 2
    assert "2026-06-04T12:00:00+03:00" in client.calls[0][1]
    assert outcome.raw_llm_metadata is not None
    assert outcome.raw_llm_metadata["current_datetime"] == "2026-06-04T12:00:00+03:00"
    assert outcome.raw_llm_metadata["max_retries"] == 1


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
                    "event": {
                        **_event_payload(start_at=None),
                        "end_at": "2026-06-20T00:00:00+03:00",
                    },
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
        ExtractionStatus.EXTRACTED,
    ]
    assert outcomes[1].errors[0].code == "duplicate_post"
    assert outcomes[2].errors[0].code == "not_event_announcement"
    assert outcomes[3].event is not None
    assert outcomes[3].event.start_at is None
    assert outcomes[3].event.end_at is not None
    assert len(client.calls) == 3


def test_extract_batch_deduplicates_events_and_keeps_latest_post_event():
    client = FakeLLMClient(
        [
            _event_response(
                "2026-06-19T17:00:00",
                title='Фестиваль народных промыслов "ВЕРЕТЕНО"',
                venue_name="Амфитеатр",
                city="Волгоград",
            ),
            _event_response(
                "2026-06-19T17:00:00",
                title='Фестиваль народных промыслов "ВЕРЕТЕНО"',
                venue_name="Амфитеатр",
                city="Волгоград",
            ),
            _event_response(
                "2026-06-20T17:00:00",
                title="Фестиваль «Веретено»",
                venue_name="Амфитеатр",
                city="Волгоград",
            ),
        ]
    )
    posts = [
        SourcePost(text="19 и 20 июня фестиваль Веретено.", published_at="2026-06-10T12:00:00+00:00"),
        SourcePost(text="19 и 20 июня фестиваль народных промыслов Веретено.", published_at="2026-06-16T12:00:00+00:00"),
        SourcePost(text="20 июня фестиваль Веретено переносится.", published_at="2026-06-19T12:00:00+00:00"),
    ]

    result = ExtractionAgent(llm_client=client).extract_batch(posts)

    assert result.extracted == 3
    assert result.skipped == 0
    assert [outcome.status for outcome in result.outcomes] == [
        ExtractionStatus.EXTRACTED,
        ExtractionStatus.EXTRACTED,
        ExtractionStatus.EXTRACTED,
    ]
    assert len(result.events) == 1
    assert result.events[0].event.title == "Фестиваль «Веретено»"
    assert result.events[0].outcome_index == 2
    assert "duplicate_of" not in result.events[0].model_dump()
    assert len(result.duplicate_events) == 2
    assert [event.duplicate_of for event in result.duplicate_events] == [2, 2]


def test_extract_batch_deduplicates_events_by_title_categories_description_and_keeps_latest():
    client = FakeLLMClient(
        [
            json.dumps(
                {
                    "is_event": True,
                    "skip_reason": None,
                    "event": {
                        **_event_payload("2026-07-08T19:00:00", title="Живая картина", venue_name="Зал Амфитеатра"),
                        "description": "Мастер-класс по живописи на спилах натурального дерева",
                        "event_type": "EducationEvent",
                        "industries": ["PerformingArts", "Art"],
                    },
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "is_event": True,
                    "skip_reason": None,
                    "event": {
                        **_event_payload("2026-07-08T19:00:00", title="Живая Картина", venue_name="Амфитеатр"),
                        "description": "Авторский мастер-класс по созданию уникальной живой картины",
                        "event_type": "EducationEvent",
                        "industries": ["Art", "PerformingArts"],
                    },
                },
                ensure_ascii=False,
            ),
        ]
    )

    result = ExtractionAgent(llm_client=client).extract_batch(
        [
            SourcePost(
                text="8 июля в 19:00 пройдет мастер-класс Живая картина в Зале Амфитеатра.",
                source_url="https://vk.com/wall-45883617_23049",
                published_at="2026-07-07T12:00:00+03:00",
            ),
            SourcePost(
                text="8 июля в 19:00 пройдет авторский мастер-класс Живая Картина в Амфитеатре.",
                source_url="https://vk.com/wall-45883617_23071",
                published_at="2026-07-08T12:00:00+03:00",
            ),
        ]
    )

    assert len(result.events) == 1
    assert result.events[0].post.source_url == "https://vk.com/wall-45883617_23071"
    assert len(result.duplicate_events) == 1
    assert result.duplicate_events[0].post.source_url == "https://vk.com/wall-45883617_23049"


def test_extract_batch_deduplicates_same_event_with_different_event_types():
    payloads = []
    for event_type in ("Festival", "MusicEvent"):
        payload = _event_payload("2026-08-23T19:00:00", title="Наследие")
        payload.update(event_type=event_type, industries=["PerformingArts"])
        payloads.append(json.dumps({"is_event": True, "skip_reason": None, "event": payload}, ensure_ascii=False))

    result = ExtractionAgent(llm_client=FakeLLMClient(payloads)).extract_batch(
        [
            SourcePost(text="23 августа в 19:00 фестиваль «Наследие».", published_at="2026-08-10T12:00:00+03:00"),
            SourcePost(text="23 августа в 19:00 «Наследие».", published_at="2026-08-11T12:00:00+03:00"),
        ]
    )

    assert len(result.events) == 1
    assert result.events[0].event.event_type.value == "MusicEvent"
    assert len(result.duplicate_events) == 1


def test_extract_batch_keeps_different_event_types_when_start_times_differ():
    payloads = []
    for start_at, event_type in (("2026-08-23T19:00:00", "Festival"), ("2026-08-23T20:00:00", "MusicEvent")):
        payload = _event_payload(start_at, title="Наследие")
        payload.update(event_type=event_type, industries=["PerformingArts"])
        payloads.append(json.dumps({"is_event": True, "skip_reason": None, "event": payload}, ensure_ascii=False))

    result = ExtractionAgent(llm_client=FakeLLMClient(payloads)).extract_batch(
        [SourcePost(text="23 августа фестиваль «Наследие»."), SourcePost(text="23 августа концерт «Наследие».")]
    )

    assert len(result.events) == 2
    assert result.duplicate_events == []


def test_extract_batch_deduplicates_events_with_shared_deadline_even_if_start_dates_differ():
    client = FakeLLMClient(
        [
            _event_response(
                "2025-06-03T00:00:00",
                title="Международная Премия #МЫВМЕСТЕ",
                venue_name=None,
                city=None,
            ),
            _event_response(
                "2025-06-23T00:00:00",
                title="Международная Премия #МЫВМЕСТЕ",
                venue_name=None,
                city=None,
            ),
        ]
    )

    result = ExtractionAgent(llm_client=client).extract_batch(
        [
            SourcePost(
                text="Успевайте подать заявку на Международную Премию #МЫВМЕСТЕ до 23 июня.",
                published_at="2025-06-03T13:10:07+00:00",
            ),
            SourcePost(
                text="Подать заявку на Международную Премию #МЫВМЕСТЕ можно до 23 июня.",
                published_at="2025-06-16T12:38:29+00:00",
            ),
        ]
    )

    assert result.extracted == 2
    assert result.skipped == 0
    assert len(result.events) == 1
    assert result.events[0].outcome_index == 1
    assert len(result.duplicate_events) == 1
    assert result.duplicate_events[0].outcome_index == 0
    assert result.duplicate_events[0].duplicate_of == 1
    assert result.outcomes[1].status == ExtractionStatus.EXTRACTED


def test_extract_batch_can_keep_semantic_event_duplicates():
    client = FakeLLMClient(
        [
            _event_response("2026-06-19T17:00:00", title='Фестиваль народных промыслов "ВЕРЕТЕНО"'),
            _event_response("2026-06-20T17:00:00", title="Фестиваль «Веретено»"),
        ]
    )

    result = ExtractionAgent(llm_client=client).extract_batch(
        [
            SourcePost(text="19 июня фестиваль Веретено.", published_at="2026-06-10T12:00:00+00:00"),
            SourcePost(text="20 июня фестиваль Веретено.", published_at="2026-06-19T12:00:00+00:00"),
        ],
        settings=BatchExtractionSettings(skip_event_duplicates=False),
    )

    assert result.extracted == 2
    assert result.skipped == 0
    assert len(result.events) == 2
    assert result.duplicate_events == []


def test_extract_batch_deduplicates_events_inside_multi_event_posts():
    client = FakeLLMClient(
        [
            json.dumps(
                {
                    "is_event": True,
                    "skip_reason": None,
                    "event": None,
                    "events": [
                        _event_payload(
                            start_at="2026-05-09T18:00:00",
                            title="Спектакль «Василий Тёркин»",
                            venue_name="Амфитеатр",
                        ),
                        _event_payload(
                            start_at="2026-05-20T18:00:00",
                            title="Спектакль «Василий Тёркин»",
                            venue_name="Амфитеатр",
                        ),
                    ],
                },
                ensure_ascii=False,
            ),
            _event_response(
                "2026-05-20T18:00:00",
                title="Спектакль «Василий Тёркин»",
                venue_name="Амфитеатр",
                city="Волгоград",
            ),
        ]
    )

    result = ExtractionAgent(llm_client=client).extract_batch(
        [
            SourcePost(
                text="9 и 20 мая в 18:00 пройдет спектакль «Василий Тёркин».",
                published_at="2026-05-01T12:00:00+00:00",
            ),
            SourcePost(
                text="20 мая в 18:00 пройдет спектакль «Василий Тёркин».",
                published_at="2026-05-02T12:00:00+00:00",
            ),
        ]
    )

    assert [outcome.status for outcome in result.outcomes] == [
        ExtractionStatus.EXTRACTED,
        ExtractionStatus.EXTRACTED,
    ]
    assert len(result.events) == 2
    assert [(item.outcome_index, item.event_index) for item in result.events] == [(0, 0), (1, 0)]
    assert len(result.duplicate_events) == 1
    assert (result.duplicate_events[0].outcome_index, result.duplicate_events[0].event_index) == (0, 1)
    assert result.duplicate_events[0].duplicate_of == 2


def test_extract_batch_returns_summary_and_applies_error_limit():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": {
                    **_event_payload(start_at="2026-06-05T18:00:00"),
                    "end_at": "2026-06-01T18:00:00",
                },
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
    assert len(client.calls) == 2


def test_extract_incremental_reuses_unchanged_outcomes_and_processes_changed_posts():
    client = FakeLLMClient(
        json.dumps(
            {
                "is_event": True,
                "skip_reason": None,
                "event": _event_payload(start_at="2026-06-07T18:00:00+03:00"),
            },
            ensure_ascii=False,
        )
    )
    cached_post = SourcePost(
        text="🎭 5 июня\nв 18:00 пройдет лекция.",
        raw_text="5 июня в 18:00 пройдет лекция.",
        external_id="vk:wall-1_1",
    )
    changed_post = SourcePost(text="Старый текст 6 июня в 18:00.", external_id="vk:wall-1_2")
    existing = [
        ExtractionAgent(llm_client=FakeLLMClient(_event_response("2026-06-05T18:00:00+03:00"))).extract(cached_post),
        ExtractionAgent(llm_client=FakeLLMClient(_event_response("2026-06-06T18:00:00+03:00"))).extract(changed_post),
    ]
    posts = [
        SourcePost(text="5 июня\nв 18:00   пройдет лекция.", raw_text="5 июня в 18:00 пройдет лекция.", external_id="vk:wall-1_1"),
        SourcePost(text="Новый текст 7 июня в 18:00.", external_id="vk:wall-1_2"),
    ]

    result = ExtractionAgent(llm_client=client).extract_incremental(posts, existing)

    assert len(client.calls) == 1
    assert result.total == 2
    assert result.cached == 1
    assert result.processed == 1
    assert result.outcomes[0].raw_llm_metadata is not None
    assert result.outcomes[0].raw_llm_metadata["incremental_cached"] is True
    assert result.outcomes[0].post == posts[0]
    assert result.outcomes[1].event is not None
    assert result.outcomes[1].event.start_at.isoformat() == "2026-06-07T18:00:00"


def test_extract_incremental_retries_cached_llm_errors_by_default():
    client = FakeLLMClient(_event_response("2026-06-05T18:00:00+03:00"))
    post = SourcePost(text="5 июня в 18:00 пройдет лекция.", external_id="vk:wall-1_1")
    existing = [
        ExtractionAgent(llm_client=FakeLLMClient("не json")).extract(post),
    ]

    result = ExtractionAgent(llm_client=client).extract_incremental([post], existing)

    assert len(client.calls) == 1
    assert result.cached == 0
    assert result.processed == 1
    assert result.outcomes[0].status == ExtractionStatus.EXTRACTED


def test_extract_incremental_can_keep_cached_llm_errors():
    client = FakeLLMClient(_event_response("2026-06-05T18:00:00+03:00"))
    post = SourcePost(text="5 июня в 18:00 пройдет лекция.", external_id="vk:wall-1_1")
    existing = [
        ExtractionAgent(llm_client=FakeLLMClient("не json")).extract(post),
    ]

    result = ExtractionAgent(llm_client=client).extract_incremental([post], existing, retry_llm_errors=False)

    assert len(client.calls) == 0
    assert result.cached == 1
    assert result.processed == 0
    assert result.outcomes[0].status == ExtractionStatus.LLM_ERROR


def _event_payload(
    start_at: str | None,
    *,
    title: str = "Мисс и Мистер Студенчество",
    venue_name: str | None = "Амфитеатр",
    city: str | None = "Волгоград",
) -> dict[str, object]:
    return {
        "title": title,
        "description": "Конкурс для студентов",
        "start_at": start_at,
        "end_at": None,
        "timezone": "Europe/Moscow",
        "city": city,
        "venue_name": venue_name,
        "address": None,
        "event_type": "CompetitionEvent",
        "attendance_type": "OfflineEventAttendanceMode",
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


def _event_response(
    start_at: str,
    *,
    title: str = "Мисс и Мистер Студенчество",
    venue_name: str | None = "Амфитеатр",
    city: str | None = "Волгоград",
) -> str:
    return json.dumps(
        {
            "is_event": True,
            "skip_reason": None,
            "event": _event_payload(start_at=start_at, title=title, venue_name=venue_name, city=city),
        },
        ensure_ascii=False,
    )
