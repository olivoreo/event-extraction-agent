from __future__ import annotations

from typing import Protocol, runtime_checkable

from event_extraction_agent.models import SourcePost


@runtime_checkable
class SourceAdapter(Protocol):
    """Minimal interface for external post sources."""

    def fetch_posts(self) -> list[SourcePost]:
        """Return prepared posts for extraction."""
