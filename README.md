# event_extraction_agent

`event_extraction_agent` - небольшая Python-библиотека для извлечения структурированных данных о мероприятиях из текстовых постов с помощью LLM.

Это первая версия пакета. Сейчас он решает только задачу extraction: вы передаете подготовленный текстовый пост в настроенного агента и получаете `ExtractionOutcome`. Загрузка постов из внешних источников, хранение в базе данных, HTTP API, расписания и batch-пайплайны намеренно не входят в пакет.

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

`Event` содержит поля события для v1: название, описание, даты, таймзону, место, тип события, формат участия, статус, язык, метаданные источника, исходный текст, роли, индустрии, навыки, стоимость и текст целевой аудитории.

## Текущий scope

Входит в v1:

- переиспользуемый Python-пакет
- типизированные входные и выходные модели
- Ollama-compatible chat client
- Groq OpenAI-compatible chat client
- промпты, валидация, легкое исправление ответа модели и уточнение типа события

Не входит в v1:

- загрузка постов из внешних источников
- разбор вложений
- хранение в базе данных
- FastAPI/backend routes
- scheduled jobs
- batch JSON import/export
- локальные ML-классификаторы

Приложение, которое использует этот пакет, отвечает за конфигурацию, секреты, хранение данных и source-specific ingestion.
