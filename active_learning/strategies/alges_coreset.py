"""Composite ALGES + coreset selection recipe."""

from __future__ import annotations

from active_learning.core.image_provider import ImageProvider
from active_learning.core.selection import (
    SelectionProgress,
    SelectionProgressSink,
    run_alges_selection,
    run_coreset_selection,
)
from active_learning.core.types import SelectionResult


def run_alges_coreset_selection(
    candidate_ids: list[str],
    *,
    seed_ids: list[str],
    image_provider: ImageProvider,
    cache_root: str,
    model,
    model_name: str,
    model_path: str,
    feature_model: str,
    n: int,
    method: str,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int | None,
    progress: bool = False,
    selection_progress: SelectionProgressSink | None = None,
) -> SelectionResult:
    prog = SelectionProgress(selection_progress) if selection_progress else None
    scope_alges = prog.span(0, 55) if prog else None
    scope_core = prog.span(55, 44) if prog else None

    stage1_n = min(len(candidate_ids), max(n, n * 4))
    stage1 = run_alges_selection(
        candidate_ids,
        image_provider=image_provider,
        cache_root=cache_root,
        model=model,
        model_name=model_name,
        model_path=model_path,
        n=stage1_n,
        method=method,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        progress=progress,
        progress_scope=scope_alges,
    )
    result = run_coreset_selection(
        stage1.selected_ids,
        seed_ids=seed_ids,
        image_provider=image_provider,
        cache_root=cache_root,
        n=n,
        feature_model=feature_model,
        seed=seed,
        progress=progress,
        progress_scope=scope_core,
    )
    result.details.update(
        {"selector": "alges_coreset", "stage1_ids": stage1.selected_ids},
    )
    return result
