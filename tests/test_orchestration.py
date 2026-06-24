"""Tests for explicit orchestration helpers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from active_learning.core.selection import (
    run_alges_selection,
    run_coreset_selection,
    run_uncertainty_selection,
)


class DummyProvider:
    pass


def test_run_coreset_selection_computes_features_then_selects():
    with patch(
        "active_learning.core.selection.score_features",
        return_value={
            "a": np.array([0.0]),
            "b": np.array([5.0]),
            "seed": np.array([0.0]),
        },
    ):
        result = run_coreset_selection(
            ["a", "b"],
            seed_ids=["seed"],
            image_provider=DummyProvider(),
            cache_root="/tmp/cache",
            n=1,
            seed=0,
        )
    assert result.selected_ids == ["b"]


def test_run_uncertainty_selection_routes_to_topk():
    with (
        patch(
            "active_learning.core.selection.score_entropy",
            return_value={"a": 0.2, "b": 0.9, "c": 0.5},
        ),
        patch(
            "active_learning.core.selection.score_features",
            return_value={
                "a": np.array([0.0]),
                "b": np.array([1.0]),
                "c": np.array([2.0]),
            },
        ),
    ):
        result = run_uncertainty_selection(
            ["a", "b", "c"],
            seed_ids=[],
            image_provider=DummyProvider(),
            cache_root="/tmp/cache",
            n=2,
            uncertainty_kind="entropy",
            uncertainty_model=object(),
            strategy="uncertainty_topk",
        )
    assert result.selected_ids == ["b", "c"]


def test_run_alges_selection_computes_embeddings_then_selects():
    with patch(
        "active_learning.core.selection.score_alges_embeddings",
        return_value={
            "a": np.array([10.0, 0.0]),
            "b": np.array([0.0, 10.0]),
            "c": np.array([1.0, 1.0]),
        },
    ):
        result = run_alges_selection(
            ["a", "b", "c"],
            image_provider=DummyProvider(),
            cache_root="/tmp/cache",
            model=object(),
            n=2,
            seed=5,
        )
    assert len(result.selected_ids) == 2
