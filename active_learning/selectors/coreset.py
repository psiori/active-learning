"""Coreset selector over ID-keyed artifacts."""

from __future__ import annotations

from typing import Any

from active_learning.strategies.coreset import greedy_k_center
from active_learning.core.types import SelectionResult
from active_learning.selectors._helpers import ids_with_artifacts, matrix_for_ids


def select_coreset(
    candidate_ids: list[str],
    n: int,
    artifacts: dict[str, dict[str, Any]],
    *,
    seed_ids: list[str] | None = None,
    seed: int | None = None,
) -> SelectionResult:
    feature_map = artifacts["features"]
    pool_ids = ids_with_artifacts(candidate_ids, feature_map)
    if not pool_ids:
        return SelectionResult(
            selected_ids=[],
            artifacts={"features": {}},
            details={"selector": "coreset"},
        )
    pool_features = matrix_for_ids(feature_map, pool_ids)
    seed_features = None
    if seed_ids:
        seed_feature_map = artifacts.get("seed_features", feature_map)
        seed_list = ids_with_artifacts(seed_ids, seed_feature_map)
        seed_features = matrix_for_ids(seed_feature_map, seed_list)
    selected_indices = greedy_k_center(
        pool_features,
        n=n,
        seed_features=seed_features,
        seed=seed,
    )
    selected_ids = [pool_ids[index] for index in selected_indices]
    return SelectionResult(
        selected_ids=selected_ids,
        artifacts={
            "features": {
                sample_id: feature_map[sample_id] for sample_id in selected_ids
            },
        },
        details={"selector": "coreset"},
    )
