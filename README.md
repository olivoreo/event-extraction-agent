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

for item in result.events:
    print(item.event.model_dump(mode="json"))

for outcome in result.outcomes:
    if outcome.errors:
        print(outcome.status, outcome.errors)
```

Минимально в `ExtractionAgentConfig` нужно передать `main_client`. Уточнения `event_type` и `title`/`description` по умолчанию выключены, чтобы не делать дополнительные LLM-запросы без явного включения.

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

По умолчанию pipeline сохраняет snapshot только по постам, которые вернул текущий `source`. Если нужно накапливать историю, включите `accumulate_existing_outcomes=True`: текущие посты обновят/добавят outcomes, а старые outcomes из `previous_result_path` или `existing_outcomes`, не совпавшие по `external_id`, останутся в новом результате.

Встроенного максимума для `outcomes` нет: библиотека не чистит накопленный кэш автоматически. Пользователь сам задает retention - например, хранит последние N outcomes перед передачей в `existing_outcomes` или перед сохранением результата. Большие файлы вроде `100000` outcomes технически не запрещены, но сохранение/загрузка JSON растут линейно, а пересборка `events` с дедупликацией рассчитана на умеренные batch sizes.

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

`VKSource` получает посты через `wall.get`, очищает текст для LLM, добавляет `source_name`, `source_url`, `published_at`, `external_id` и возвращает список `SourcePost`. В incremental-режиме закрепленный VK-пост, уже присутствующий в предыдущих outcomes, не занимает место в `posts_per_source_limit`: источник дочитывает обычные посты до заданного лимита.

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

`refinement_client` опционален и используется при включенных refinement-флагах. Если он не задан, refinement будет использовать `main_client`.
При Groq основной и refinement-клиент используют один `GROQ_API_KEY`; отдельно задавать ключ для refinement не нужно.

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

### Rate limits Groq

`GroqChatClient` различает минутные и суточные HTTP `429` по типу лимита в ответе Groq:

- `RPM`, `TPM`, `ITPM`, `OTPM`: клиент ждет `Retry-After` с запасом `0.25` секунды и повторяет тот же запрос до успеха. Если заголовка нет, используются `x-ratelimit-reset-tokens`, время из текста ошибки или безопасная пауза `60` секунд. Эти повторы не расходуют `max_retries` клиента или агента и не создают `llm_error`, поэтому batch продолжает обработку с текущего поста;
- `RPD`, `TPD`: клиент не ждет суточного сброса. Текущий и оставшиеся посты получают `llm_error` с кодом `daily_rate_limit_exceeded` без новых API-вызовов; incremental-режим повторит их при следующем запуске по обычной логике `retry_llm_errors=True`.

Для моделей Groq с поддержкой Structured Outputs агент автоматически использует JSON Schema Mode. Для `openai/gpt-oss-20b` и `openai/gpt-oss-120b` включается строгий режим с гарантированным соответствием схеме; остальные клиенты сохраняют прежний JSON-интерфейс.

`start_at` и `end_at` хранят локальные дату и время события без UTC offset; часовой пояс передается отдельно в поле `timezone`. Если LLM или пользователь передает ISO-дату с `Z`/offset, `Event` сохраняет указанные часы и минуты и удаляет информацию о смещении до проверки диапазона дат.

`GROQ_MAX_RETRIES` теперь относится только к временным HTTP `5xx`. Формат и назначение заголовков описаны в [Groq Rate Limits](https://console.groq.com/docs/rate-limits); общий формат ошибок — в [Groq API Error Codes](https://console.groq.com/docs/errors).

## Запуск из `.env`

Скрипты из `scripts/` читают `.sandbox/.env`. Основные параметры:

```env
REQUEST_TIMEOUT_SECONDS=120
MIN_REQUEST_INTERVAL_SECONDS=0
MAX_RETRIES=0
GROQ_MAX_RETRIES=3
USE_EVENT_TYPE_REFINEMENT=false
USE_TITLE_DESCRIPTION_REFINEMENT=false
GOLDEN_POST_LIMIT=3
GOLDEN_MAX_STATUS_MISMATCHES=0
```

- `REQUEST_TIMEOUT_SECONDS` передается в `OllamaChatClient`/`GroqChatClient` как timeout одного HTTP-запроса.
- `MIN_REQUEST_INTERVAL_SECONDS` задает минимальную паузу между LLM-вызовами агента.
- `MAX_RETRIES` повторяет весь вызов агента после ошибки клиента.
- `GROQ_MAX_RETRIES` ограничивает повторы Groq-клиента на HTTP `5xx`; минутные `429` повторяются до успеха отдельно.
- `USE_EVENT_TYPE_REFINEMENT=false` экономит токены и запросы; включайте только если нужно дополнительно уточнять `event_type`.
- `USE_TITLE_DESCRIPTION_REFINEMENT=true` включает отдельную проверку `title` и полной сухой выжимки в `description` для каждого найденного события.
- `price_text` остается `null`, если бесплатность, цена или условие покупки не указаны явно.
- `GOLDEN_POST_LIMIT` ограничивает число первых постов в `scripts/run_golden_extraction.py`; без него обрабатывается весь golden-набор.
- `GOLDEN_MAX_STATUS_MISMATCHES=0` задает допустимое число расхождений статусов с golden-разметкой; при превышении скрипт завершается с ненулевым кодом.

## Что возвращает pipeline

`BatchExtractionResult` содержит:

- `events`: плоский список найденных мероприятий (`ExtractedEvent`) после дедупликации;
- `duplicate_events`: мероприятия, отброшенные дедупликацией;
- `outcomes`: результаты по каждому посту;
- `extracted`, `skipped`, `invalid`, `llm_errors`: счетчики статусов;
- `cached`, `processed`: счетчики incremental-режима;
- `error_count`, `error_limit_reached`: информация о batch-лимитах;
- `save_json(path)` и `load_json(path)`.

Каждый `ExtractedEvent` содержит само `Event`, исходный `SourcePost`, индекс post-outcome и индекс события внутри поста. `duplicate_of` есть только у элементов `duplicate_events` и указывает на оставленное событие во внутреннем плоском списке до дедупликации. По умолчанию для группы semantic-дубликатов свежий event остается основным, а LLM объединяет в него обогащающие поля (`description`, роли, индустрии, skills и аудиторию) из предыдущих версий. Дата, время, место, адрес, цена, формат и source/raw metadata остаются от свежего event. Непротиворечивые смысловые детали из старых версий, включая полезные ссылки и условия регистрации/участия, сохраняются; при конфликте или явном обновлении всегда приоритетна свежая версия. В metadata canonical event сохраняются идентификаторы уже учтенных постов: `external_id`, если он есть, иначе компактный hash нормализованного текста. Поэтому при следующем incremental-run одни и те же версии не отправляются на merge повторно; если появляется новый свежий дубль, merge получает только ближайшую canonical-версию и еще не учтенные источники. Merge можно отключить через `BatchExtractionSettings(merge_event_duplicates=False)`. При ошибке merge сохраняется прежнее поведение: остается свежий event без обогащения. Смердженный event также записывается обратно в соответствующий `ExtractionOutcome`, поэтому сохраняется в incremental-кэше.

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
