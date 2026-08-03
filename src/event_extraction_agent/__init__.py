"""Public API for the event extraction package."""

from event_extraction_agent.agent import (
    ExtractionAgent,
    GroqChatClient,
    GroqDailyRateLimitError,
    LLMClient,
    OllamaChatClient,
)
from event_extraction_agent.models import (
    AttendanceType,
    BatchExtractionResult,
    BatchExtractionSettings,
    BatchMode,
    DuplicateExtractedEvent,
    Event,
    ExtractedEvent,
    EventType,
    ExtractionAgentConfig,
    ExtractionError,
    ExtractionOutcome,
    ExtractionStatus,
    SourcePost,
    drop_event,
)
from event_extraction_agent.pipeline import ExtractionPipeline
from event_extraction_agent.sources import SourceAdapter
from event_extraction_agent.vk import (
    VKApiError,
    VKFetchResult,
    VKPostSource,
    VKSource,
)

__all__ = [
    "AttendanceType",
    "BatchExtractionResult",
    "BatchExtractionSettings",
    "BatchMode",
    "DuplicateExtractedEvent",
    "Event",
    "ExtractedEvent",
    "EventType",
    "ExtractionAgentConfig",
    "ExtractionAgent",
    "ExtractionError",
    "ExtractionOutcome",
    "ExtractionStatus",
    "ExtractionPipeline",
    "GroqChatClient",
    "GroqDailyRateLimitError",
    "LLMClient",
    "OllamaChatClient",
    "SourceAdapter",
    "SourcePost",
    "VKApiError",
    "VKFetchResult",
    "VKPostSource",
    "VKSource",
    "drop_event",
]
