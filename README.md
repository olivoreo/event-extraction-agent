# event_extraction_agent

`event_extraction_agent` - небольшая Python-библиотека для извлечения структурированных данных о мероприятиях из текстовых постов с помощью LLM.

Версия `0.2.0` решает задачу extraction для одного подготовленного поста или пачки подготовленных постов: вы передаете `SourcePost` в настроенного агента и получаете `ExtractionOutcome` или `BatchExtractionResult`. Загрузка постов из внешних источников, хранение в базе данных, HTTP API и расписания намеренно не входят в пакет.

## Установка

Локально из этого репозитория:

```bash
python -m pip install -e .
```

Для разработки:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Быстрый старт с Ollama

```python
from event_extraction_agent import ExtractionAgent, OllamaChatClient, SourcePost

client = OllamaChatClient(
    model="qwen2.5:3b",
    host="http://localhost:11434",
)
agent = ExtractionAgent(llm_client=client)

post = SourcePost(
    text="12 июня в 18:00 в городском лектории пройдет открытая лекция.",
    source_name="Example source",
    source_url="https://example.com/posts/123",
    published_at="2026-06-01T10:00:00+03:00",
    external_id="post-123",
)

outcome = agent.extract(post)

if outcome.event:
    print(outcome.event.model_dump(mode="json"))
else:
    print(outcome.status, outcome.errors)
```

## Batch extraction

```python
from event_extraction_agent import BatchExtractionSettings, ExtractionAgent, SourcePost

agent = ExtractionAgent()

posts = [
    SourcePost(text="12 июня в 18:00 пройдет лекция.", external_id="post-1"),
    SourcePost(text="Просто информационный пост.", external_id="post-2"),
    SourcePost(text="12 июня в 18:00 пройдет лекция.", external_id="post-1"),
]

outcomes = agent.extract_many(posts)
result = agent.extract_batch(posts, settings=BatchExtractionSettings(max_errors=3))

print([outcome.status for outcome in outcomes])
print(result.extracted, result.skipped, result.invalid, result.llm_errors)
```

Batch-обработка в `0.2.0` последовательная и сохраняет порядок входных постов. По умолчанию агент пропускает дубли по `external_id`, а если `external_id` нет - по нормализованному тексту. `max_errors` ограничивает число `invalid` и `llm_error`; после достижения лимита оставшиеся посты возвращаются как `skipped` с ошибкой `error_limit_reached`.

## Быстрый старт с Groq

```python
from event_extraction_agent import ExtractionAgent, GroqChatClient, SourcePost

client = GroqChatClient(
    api_key="gsk_...",
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)
agent = ExtractionAgent(llm_client=client)

post = SourcePost(
    text="20 июля в 15:30 пройдет открытый мастер-класс для начинающих дизайнеров.",
    source_name="Example source",
    source_url="https://example.com/posts/456",
    published_at="2026-07-10T12:00:00+03:00",
    external_id="post-456",
)

outcome = agent.extract(post)
```

## Публичные модели

`SourcePost` - входная модель поста:

- `text`
- `source_name`
- `source_url`
- `published_at`
- `external_id`

Обязательное поле только `text`. Остальные поля опциональны, но полезны для трассировки результата и восстановления даты события, если в тексте указан день и месяц без года.

`ExtractionOutcome` - результат работы агента:

- `status`: `extracted`, `skipped`, `invalid` или `llm_error`
- `event`: модель `Event`, если извлечение успешно
- `post`: исходный `SourcePost`
- `errors`: структурированные ошибки
- `raw_llm_metadata`: минимальные отладочные метаданные

`BatchExtractionSettings` - настройки batch-обработки:

- `mode`: сейчас только `sequential`
- `max_errors`: опциональный лимит ошибок `invalid` и `llm_error`
- `skip_empty`: пропуск пустых текстов, если такие попадут в пачку
- `skip_duplicates`: пропуск дублирующихся постов

`BatchExtractionResult` - сводный результат batch-обработки:

- `settings`
- `outcomes`
- `total`
- `extracted`
- `skipped`
- `invalid`
- `llm_errors`
- `error_count`
- `error_limit_reached`

`Event` содержит поля события: название, описание, даты, таймзону, место, тип события, формат участия, статус, язык, метаданные источника, исходный текст, роли, индустрии, навыки, стоимость и текст целевой аудитории.

## Текущий scope

Входит в v0.2.0:

- переиспользуемый Python-пакет
- типизированные входные и выходные модели
- Ollama-compatible chat client
- Groq OpenAI-compatible chat client
- промпты, валидация, легкое исправление ответа модели и уточнение типа события
- последовательная batch-обработка подготовленных `SourcePost`
- лимит ошибок, сохранение порядка, пропуск дублей и явная batch-сводка

Не входит в v0.2.0:

- загрузка постов из внешних источников
- разбор вложений
- хранение в базе данных
- FastAPI/backend routes
- scheduled jobs
- batch JSON import/export и source-specific ingestion
- локальные ML-классификаторы

Приложение, которое использует этот пакет, отвечает за конфигурацию, секреты, хранение данных и source-specific ingestion.

## Происхождение пакета

`event_extraction_agent` выделен из проекта `event-ai-agent`, который использовался как основа для первой версии библиотеки. Из исходного проекта перенесено только extraction-ядро: модели события, промпты, валидация, легкое исправление ответа LLM и клиенты для Ollama/Groq.

При переносе намеренно не включались backend API, база данных, загрузчики внешних источников и экспериментальные ML-компоненты. Цель пакета - сделать extraction-логику переиспользуемой в других проектах.
