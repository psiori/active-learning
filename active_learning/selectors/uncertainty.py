"""Uncertainty-based selectors."""

from __future__ import annotations

from typing import Any


from active_learning.selectors._helpers import (
    ids_with_artifacts,
    matrix_for_ids,
    vector_for_ids,
)
from active_learning.strategies.coreset import greedy_k_center
from active_learning.strategies.uncertainty_coreset import uncertainty_coreset
from active_learning.core.types import SelectionResult


def select_uncertainty_topk(
    candidate_ids: list[str],
    n: int,
    artifacts: dict[str, dict[str, Any]],
) -> SelectionResult:
    uncertainty_map = artifacts["uncertainty"]
    pool_ids = ids_with_artifacts(candidate_ids, uncertainty_map)
    ranked_ids = sorted(
        pool_ids,
        key=lambda sample_id: float(uncertainty_map[sample_id]),
        reverse=True,
    )[: min(n, len(pool_ids))]
    return SelectionResult(
        selected_ids=ranked_ids,
        scores={
            sample_id: float(uncertainty_map[sample_id]) for sample_id in ranked_ids
        },
        artifacts={
            "uncertainty": {
                sample_id: uncertainty_map[sample_id] for sample_id in ranked_ids
            },
        },
        details={"selector": "uncertainty_topk"},
    )


def select_uncertainty_coreset(
    candidate_ids: list[str],
    n: int,
    artifacts: dict[str, dict[str, Any]],
    *,
    seed_ids: list[str] | None = None,
    alpha: float = 0.5,
    seed: int | None = None,
) -> SelectionResult:
    feature_map = artifacts["features"]
    uncertainty_map = artifacts["uncertainty"]
    pool_ids = [
        sid for sid in candidate_ids if sid in feature_map and sid in uncertainty_map
    ]
    if not pool_ids:
        return SelectionResult(
            selected_ids=[],
            scores={},
            artifacts={"features": {}, "uncertainty": {}},
            details={"selector": "uncertainty_coreset", "alpha": alpha},
        )
    pool_features = matrix_for_ids(feature_map, pool_ids)
    uncertainty = vector_for_ids(uncertainty_map, pool_ids)
    seed_features = None
    if seed_ids:
        seed_feature_map = artifacts.get("seed_features", feature_map)
        seed_list = ids_with_artifacts(seed_ids, seed_feature_map)
        seed_features = matrix_for_ids(seed_feature_map, seed_list)
    selected_indices = uncertainty_coreset(
        pool_features,
        uncertainty,
        n=n,
        seed_features=seed_features,
        alpha=alpha,
        seed=seed,
    )
    selected_ids = [pool_ids[index] for index in selected_indices]
    return SelectionResult(
        selected_ids=selected_ids,
        scores={
            sample_id: float(uncertainty_map[sample_id]) for sample_id in selected_ids
        },
        artifacts={
            "features": {
                sample_id: feature_map[sample_id] for sample_id in selected_ids
            },
            "uncertainty": {
                sample_id: uncertainty_map[sample_id] for sample_id in selected_ids
            },
        },
        details={"selector": "uncertainty_coreset", "alpha": alpha},
    )


def select_uncertainty_topk_then_coreset(
    candidate_ids: list[str],
    n: int,
    artifacts: dict[str, dict[str, Any]],
    *,
    seed_ids: list[str] | None = None,
    candidate_multiplier: int = 4,
    seed: int | None = None,
) -> SelectionResult:
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive")
    stage1_n = min(len(candidate_ids), max(n, n * candidate_multiplier))
    stage1 = select_uncertainty_topk(candidate_ids, stage1_n, artifacts)
    feature_map = artifacts["features"]
    pruned = select_coreset_from_features(
        stage1.selected_ids,
        n=n,
        feature_map=feature_map,
        seed_ids=seed_ids,
        seed_feature_map=artifacts.get("seed_features", feature_map),
        seed=seed,
    )
    pruned.details.update(
        {
            "selector": "uncertainty_topk_then_coreset",
            "stage1_ids": stage1.selected_ids,
            "stage1_scores": stage1.scores,
            "stage1_n": stage1_n,
        },
    )
    pruned.artifacts["uncertainty"] = {
        sample_id: artifacts["uncertainty"][sample_id]
        for sample_id in pruned.selected_ids
    }
    pruned.scores = {
        sample_id: float(artifacts["uncertainty"][sample_id])
        for sample_id in pruned.selected_ids
    }
    return pruned


def select_coreset_from_features(
    candidate_ids: list[str],
    *,
    n: int,
    feature_map: dict[str, Any],
    seed_ids: list[str] | None,
    seed_feature_map: dict[str, Any],
    seed: int | None,
) -> SelectionResult:
    pool_ids = ids_with_artifacts(candidate_ids, feature_map)
    if not pool_ids:
        return SelectionResult(
            selected_ids=[],
            artifacts={"features": {}},
        )
    pool_features = matrix_for_ids(feature_map, pool_ids)
    seed_features = None
    if seed_ids:
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
    )
