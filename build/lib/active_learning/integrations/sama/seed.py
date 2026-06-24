"""Sama sink helpers."""

from __future__ import annotations

from interface.service.sama import Sama


def create_batch(
    export_id: str,
    urls: list[str],
    *,
    project_id: int,
    priority: int = 0,
) -> dict:
    """Submit URLs to Sama; returns dict with ``batch_id`` and ``creation_status``."""
    sama = Sama(project_id=project_id)
    return sama.create_batch(export_id, urls, priority=priority)
