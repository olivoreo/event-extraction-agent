# event_extraction_agent

`event_extraction_agent` - Python-библиотека для извлечения структурированных данных о мероприятиях из текстовых постов с помощью LLM.

Главный и стабильный способ использования - `ExtractionPipeline`: вы передаете источник постов и `ExtractionAgentConfig`, а на выходе получаете `BatchExtractionResult` со всеми событиями, статусами, ошибками и методами сохранения результата.

## Установка

```bash
pip install event-extraction-agent
```

Для локальной разработки из репозитория:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Быстрый старт

```python
from event_extraction_agent import (
    ExtractionAgentConfig,
    ExtractionPipeline,
    OllamaChatClient,
    SourcePost,
)


class MySource:
    def fetch_posts(self) -> list[SourcePost]:
        return [
            SourcePost(
                text="12 июня в 18:00 пройдет открытая лекция.",
                source_name="Example source",
                source_url="https://example.com/posts/123",
                published_at="2026-06-01T10:00:00+03:00",
                external_id="post-123",
            )
        ]


pipeline = ExtractionPipeline(
    source=MySource(),
    agent_config=ExtractionAgentConfig(
        main_client=OllamaChatClient(model="qwen2.5:3b"),
    ),
)

result = pipeline.run()

for outcome in result.outcomes:
    if outcome.event:
        print(outcome.event.model_dump(mode="json"))
    else:
        print(outcome.status, outcome.errors)
```

Минимально в `ExtractionAgentConfig` нужно передать `main_client`. Уточнение типа события (`use_event_type_refinement`) по умолчанию выключено, чтобы не делать второй LLM-запрос на часть постов.

## Сохранение результата

`pipeline.run()` возвращает `BatchExtractionResult`. Его можно сохранить в JSON:

```python
result = pipeline.run()
result.save_json("events_result.json")
```

И загрузить обратно:

```python
from event_extraction_agent import BatchExtractionResult

previous = BatchExtractionResult.load_json("events_result.json")
```

## Incremental processing

Pipeline может сам загрузить предыдущий результат и сохранить новый:

```python
pipeline = ExtractionPipeline(
    source=source,
    agent_config=agent_config,
    previous_result_path="events_result.json",
    save_result_path="events_result.json",
)

result = pipeline.run()
print(result.cached, result.processed)
```

Incremental-режим пропускает LLM extraction, если у нового поста совпали `external_id` и нормализованный текст для LLM (`raw_text`, если он задан, иначе `text`) с предыдущим результатом. Результаты со статусом `llm_error` по умолчанию обрабатываются повторно.

## VK source

Для VK есть готовый source adapter:

```python
from event_extraction_agent import ExtractionAgentConfig, ExtractionPipeline, OllamaChatClient, VKSource

source = VKSource(
    access_token="vk-service-token",
    sources=[
        "https://vk.com/club123",
        "public456",
        "my_community_domain",
        -789,
    ],
    posts_per_source_limit=20,
)

result = ExtractionPipeline(
    source=source,
    agent_config=ExtractionAgentConfig(
        main_client=OllamaChatClient(model="qwen2.5:3b"),
    ),
    previous_result_path="events_result.json",
    save_result_path="events_result.json",
).run()
```

`VKSource` получает посты через `wall.get`, очищает текст для LLM, добавляет `source_name`, `source_url`, `published_at`, `external_id` и возвращает список `SourcePost`.

Если один VK source недоступен, остальные источники по умолчанию продолжают обрабатываться. Ошибки доступны через `source.errors` или `fetch_posts_with_errors()`:

```python
fetch_result = source.fetch_posts_with_errors()

for error in fetch_result.errors:
    print(error.source, error.code, error)
```

По умолчанию `VKSource` использует rate limit `20` запросов в секунду и retry/backoff для временных ошибок VK, HTTP `429`/`5xx` и сетевых сбоев.

## Настройка агента и клиентов

`ExtractionAgentConfig` управляет поведением агента: клиентами, датой для prompt, паузой между LLM-вызовами, retry на уровне агента и включением дополнительного уточнения `event_type`.

```python
config = ExtractionAgentConfig(
    main_client=main_client,
    current_datetime="2026-06-10T12:00:00+03:00",
    min_request_interval_seconds=1.1,
    max_retries=0,
)
```

`refinement_client` нужен только если включен `use_event_type_refinement=True`. Если он не задан, refinement будет использовать `main_client`.

Поддерживается любой LLM-клиент с методом:

```python
complete(system_prompt: str, user_prompt: str) -> str
```

В пакете есть готовые клиенты:

- `OllamaChatClient`
- `GroqChatClient`

Таймауты и retry HTTP-запросов настраиваются у самих клиентов:

```python
client = GroqChatClient(
    api_key="...",
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    timeout_seconds=60,
    max_retries=3,
)
```

`max_retries` в `ExtractionAgentConfig` повторяет весь `client.complete(...)`. Обычно достаточно оставить его `0` и использовать retry клиента.

## Запуск из `.env`

Скрипты из `scripts/` читают `.sandbox/.env`. Основные параметры:

```env
REQUEST_TIMEOUT_SECONDS=120
MIN_REQUEST_INTERVAL_SECONDS=0
MAX_RETRIES=0
GROQ_MAX_RETRIES=3
USE_EVENT_TYPE_REFINEMENT=false
```

- `REQUEST_TIMEOUT_SECONDS` передается в `OllamaChatClient`/`GroqChatClient` как timeout одного HTTP-запроса.
- `MIN_REQUEST_INTERVAL_SECONDS` задает минимальную паузу между LLM-вызовами агента.
- `MAX_RETRIES` повторяет весь вызов агента после ошибки клиента.
- `GROQ_MAX_RETRIES` повторяет HTTP-запросы Groq-клиента на `429`/`5xx`.
- `USE_EVENT_TYPE_REFINEMENT=false` экономит токены и запросы; включайте только если нужно дополнительно уточнять `event_type`.

## Что возвращает pipeline

`BatchExtractionResult` содержит:

- `outcomes`: результаты по каждому посту;
- `extracted`, `skipped`, `invalid`, `llm_errors`: счетчики статусов;
- `cached`, `processed`: счетчики incremental-режима;
- `error_count`, `error_limit_reached`: информация о batch-лимитах;
- `save_json(path)` и `load_json(path)`.

Каждый `ExtractionOutcome` содержит исходный `SourcePost`, статус обработки, найденное `Event` или структурированные ошибки.

`Event` не содержит `event_status`: агент извлекает только само мероприятие. Посты, которые являются только сообщением об отмене уже существующего события, пропускаются как не-анонсы.

## Границы пакета

В пакет входит extraction-ядро, pipeline, модели, LLM-клиенты для Ollama/Groq и VK source adapter.

Пакет намеренно не включает:

- базу данных;
- HTTP API;
- расписания;
- чтение секретов из `.env`;
- обработку VK-вложений;
- source adapters кроме VK.

Приложение, которое использует библиотеку, отвечает за конфигурацию, секреты, хранение данных и собственные источники.

## Происхождение пакета

`event_extraction_agent` выделен из проекта [olivoreo/event-ai-agent](https://github.com/olivoreo/event-ai-agent). Из исходного проекта перенесено extraction-ядро: модели события, промпты, валидация, исправление ответа LLM и клиенты для Ollama/Groq.

При переносе намеренно не включались backend API, база данных, загрузчики внешних источников и экспериментальные ML-компоненты. Цель пакета - сделать extraction-логику переиспользуемой в других проектах.
