"""Tests for selector modules."""

from __future__ import annotations

import numpy as np

from active_learning.selectors import (
    select_alges,
    select_coreset,
    select_uncertainty_coreset,
    select_uncertainty_topk,
    select_uncertainty_topk_then_coreset,
)


def test_select_uncertainty_topk_returns_descending_scores():
    result = select_uncertainty_topk(
        ["a", "b", "c"],
        2,
        {"uncertainty": {"a": 0.2, "b": 0.9, "c": 0.4}},
    )
    assert result.selected_ids == ["b", "c"]


def test_select_coreset_uses_seed_features():
    artifacts = {
        "features": {
            "a": np.array([0.0]),
            "b": np.array([5.0]),
            "c": np.array([10.0]),
        },
        "seed_features": {"seed": np.array([0.0])},
    }
    result = select_coreset(["a", "b", "c"], 2, artifacts, seed_ids=["seed"], seed=7)
    assert result.selected_ids == ["c", "b"]


def test_select_uncertainty_coreset_can_reduce_to_pure_uncertainty():
    artifacts = {
        "features": {
            "a": np.array([0.0]),
            "b": np.array([1.0]),
            "c": np.array([2.0]),
        },
        "seed_features": {"seed": np.array([0.0])},
        "uncertainty": {"a": 0.1, "b": 0.9, "c": 0.8},
    }
    result = select_uncertainty_coreset(
        ["a", "b", "c"],
        2,
        artifacts,
        seed_ids=["seed"],
        alpha=0.0,
        seed=3,
    )
    assert result.selected_ids == ["b", "c"]


def test_select_uncertainty_topk_then_coreset_returns_stage_details():
    artifacts = {
        "features": {
            "a": np.array([0.0]),
            "b": np.array([5.0]),
            "c": np.array([10.0]),
            "d": np.array([15.0]),
        },
        "seed_features": {"seed": np.array([0.0])},
        "uncertainty": {"a": 0.1, "b": 0.8, "c": 0.9, "d": 0.2},
    }
    result = select_uncertainty_topk_then_coreset(
        ["a", "b", "c", "d"],
        1,
        artifacts,
        seed_ids=["seed"],
        candidate_multiplier=2,
        seed=1,
    )
    assert result.details["stage1_ids"] == ["c", "b"]
    assert len(result.selected_ids) == 1


def test_select_alges_returns_unique_ids():
    result = select_alges(
        ["a", "b", "c"],
        2,
        {
            "alges_embeddings": {
                "a": np.array([10.0, 0.0]),
                "b": np.array([0.0, 10.0]),
                "c": np.array([1.0, 1.0]),
            },
        },
        seed=2,
    )
    assert len(result.selected_ids) == 2
    assert len(set(result.selected_ids)) == 2
