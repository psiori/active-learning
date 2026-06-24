"""Core active-learning types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SampleId = str
ArtifactMapping = dict[SampleId, Any]


@dataclass(slots=True)
class SelectionResult:
    """Selection output shared across selectors and sinks."""

    selected_ids: list[SampleId]
    scores: dict[SampleId, float] = field(default_factory=dict)
    artifacts: dict[str, ArtifactMapping] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
