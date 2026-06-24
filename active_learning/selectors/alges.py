"""ALGES selector over cached gradient embeddings."""

from __future__ import annotations

from typing import Any

from active_learning.selectors._helpers import matrix_for_ids
from active_learning.strategies.alges import kmeans_pp_select
from active_learning.core.types import SelectionResult


def select_alges(
    candidate_ids: list[str],
    n: int,
    artifacts: dict[str, dict[str, Any]],
    *,
    seed: int | None = None,
    progress: bool = False,
) -> SelectionResult:
    embedding_map = artifacts["alges_embeddings"]
    embeddings = matrix_for_ids(embedding_map, candidate_ids)
    selected_indices = kmeans_pp_select(embeddings, n=n, seed=seed, progress=progress)
    selected_ids = [candidate_ids[index] for index in selected_indices]
    return SelectionResult(
        selected_ids=selected_ids,
        artifacts={
            "alges_embeddings": {
                sample_id: embedding_map[sample_id] for sample_id in selected_ids
            },
        },
        details={"selector": "alges"},
    )
