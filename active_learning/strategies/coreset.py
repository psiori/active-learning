"""Greedy K-Center coreset selection."""

import numpy as np
from scipy.spatial.distance import cdist


def _euclidean_to_point(pool: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of pool to a single point.

    Faster than cdist for single-point queries: avoids scipy overhead
    and uses einsum to skip intermediate allocations.
    """
    diff = pool - point
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def _init_min_distances(
    pool_features: np.ndarray,
    seed_features: np.ndarray | None,
    seed: int | None,
) -> tuple[np.ndarray, list[int]]:
    """Initialize min_distances array and selected list.

    Returns (min_distances, selected) where selected may contain the
    first random pick if no seed_features are provided.
    """
    n_pool = pool_features.shape[0]

    if seed_features is not None and len(seed_features) > 0:
        chunk_size = 256
        min_distances = np.full(n_pool, np.inf)
        for i in range(0, len(seed_features), chunk_size):
            chunk = seed_features[i : i + chunk_size]
            dists = cdist(pool_features, chunk, metric="euclidean")
            chunk_min = dists.min(axis=1)
            min_distances = np.minimum(min_distances, chunk_min)
        return min_distances, []
    else:
        min_distances = np.full(n_pool, np.inf)
        rng = np.random.RandomState(seed)
        first = rng.randint(n_pool)
        min_distances = np.minimum(
            min_distances, _euclidean_to_point(pool_features, pool_features[first])
        )
        return min_distances, [first]


def greedy_k_center(
    pool_features: np.ndarray,
    n: int,
    seed_features: np.ndarray | None = None,
    seed: int | None = None,
) -> list[int]:
    """Select n points from pool that maximize minimum distance to selected set.

    Pure diversity-based selection. For uncertainty-aware selection,
    use uncertainty_coreset() instead.

    Args:
        pool_features: [N, D] feature matrix for the unlabeled pool.
        n: Number of points to select.
        seed_features: [M, D] feature matrix for already-labeled images.
        seed: Random seed for the initial random pick.

    Returns:
        List of n indices into pool_features.
    """
    n_pool = pool_features.shape[0]
    n = min(n, n_pool)

    min_distances, selected = _init_min_distances(pool_features, seed_features, seed)
    n -= len(selected)

    for _ in range(n):
        idx = int(np.argmax(min_distances))
        selected.append(idx)
        new_dists = _euclidean_to_point(pool_features, pool_features[idx])
        min_distances = np.minimum(min_distances, new_dists)

    return selected
