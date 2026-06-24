"""Local filesystem integration for active-learning image sources."""

from active_learning.integrations.local.source import (
    LocalImageProviderSource,
    discover_local_images,
)

__all__ = ["LocalImageProviderSource", "discover_local_images"]
