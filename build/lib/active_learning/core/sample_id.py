"""Utilities for parsing active-learning sample identifiers."""

from __future__ import annotations

from datetime import datetime
import os

_ID_TS_FORMATS = (
    "%Y-%m-%dT%H_%M_%S_%fZ",
    "%Y-%m-%dT%H_%M_%SZ",
)


def datetime_from_sample_id(sample_id: str) -> datetime | None:
    """Parse a timestamp from a sample ID if it matches known filename forms."""
    base = os.path.splitext(os.path.basename(str(sample_id)))[0]
    for fmt in _ID_TS_FORMATS:
        try:
            return datetime.strptime(base, fmt)
        except ValueError:
            continue
    return None
