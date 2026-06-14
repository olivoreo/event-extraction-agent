"""Public API for the event extraction package."""

from event_extraction_agent.agent import (
    ExtractionAgent,
    GroqChatClient,
    LLMClient,
    OllamaChatClient,
)
from event_extraction_agent.models import (
    AttendanceType,
    BatchExtractionResult,
    BatchExtractionSettings,
    BatchMode,
    Event,
    EventStatus,
    EventType,
    ExtractionAgentConfig,
    ExtractionError,
    ExtractionOutcome,
    ExtractionStatus,
    SourcePost,
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
    "Event",
    "EventStatus",
    "EventType",
    "ExtractionAgentConfig",
    "ExtractionAgent",
    "ExtractionError",
    "ExtractionOutcome",
    "ExtractionStatus",
    "ExtractionPipeline",
    "GroqChatClient",
    "LLMClient",
    "OllamaChatClient",
    "SourceAdapter",
    "SourcePost",
    "VKApiError",
    "VKFetchResult",
    "VKPostSource",
    "VKSource",
]
