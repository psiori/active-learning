"""Uncertainty-weighted coreset selection (Confident Coreset).

Combines diversity (coreset distances) with model uncertainty using
multiplicative scoring:

    score(i) = norm(min_distance(i))^alpha * norm(uncertainty(i))^(1-alpha)

Reference: "Confident Coreset for Active Learning in Medical Image Analysis"
(Smailagic et al., 2020) — tested on segmentation with alpha=0.5.
"""

import numpy as np

from active_learning.strategies.coreset import _euclidean_to_point, _init_min_distances


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if constant."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def uncertainty_coreset(
    pool_features: np.ndarray,
    uncertainty: np.ndarray,
    n: int,
    seed_features: np.ndarray | None = None,
    alpha: float = 0.5,
    seed: int | None = None,
) -> list[int]:
    """Select n points using uncertainty-weighted greedy K-Center.

    At each step, selects the point with the highest combined score:
        score = norm(distance)^alpha * norm(uncertainty)^(1-alpha)

    Args:
        pool_features: [N, D] feature matrix for the unlabeled pool.
        uncertainty: [N] per-image uncertainty scores. Higher = more uncertain.
        n: Number of points to select.
        seed_features: [M, D] feature matrix for already-labeled images.
        alpha: Trade-off between diversity and uncertainty.
            1.0 = pure diversity (equivalent to greedy_k_center)
            0.5 = equal weight (recommended for segmentation)
            0.0 = pure uncertainty
        seed: Random seed for the initial random pick.

    Returns:
        List of n indices into pool_features.
    """
    n_pool = pool_features.shape[0]
    n = min(n, n_pool)

    if len(uncertainty) != n_pool:
        raise ValueError(
            f"uncertainty length ({len(uncertainty)}) must match pool size ({n_pool})"
        )

    norm_uncertainty = _min_max_normalize(uncertainty)

    min_distances, selected = _init_min_distances(pool_features, seed_features, seed)
    n -= len(selected)

    selected_set = set(selected)

    for _ in range(n):
        norm_dist = _min_max_normalize(min_distances)
        scores = np.power(norm_dist, alpha) * np.power(norm_uncertainty, 1 - alpha)

        # Zero out already-selected points
        for s in selected_set:
            scores[s] = -1

        idx = int(np.argmax(scores))
        selected.append(idx)
        selected_set.add(idx)

        new_dists = _euclidean_to_point(pool_features, pool_features[idx])
        min_distances = np.minimum(min_distances, new_dists)

    return selected
