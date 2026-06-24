"""Shared helpers for uncertainty-map aggregation."""

import numpy as np


def entropy_map(probs: np.ndarray) -> np.ndarray:
    """Compute entropy along the last dimension."""
    eps = 1e-10
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def aggregate_uncertainty_map(
    uncertainty_map: np.ndarray,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
) -> np.ndarray:
    """Reduce [B, H, W] or [T, B, H, W] maps to image-level scores."""
    if aggregation == "mean":
        return uncertainty_map.mean(axis=(-2, -1))
    if aggregation == "max":
        return uncertainty_map.max(axis=(-2, -1))
    if aggregation != "topk_mean":
        raise ValueError(
            f"aggregation must be 'mean', 'topk_mean', or 'max', got {aggregation!r}"
        )

    flat = uncertainty_map.reshape(*uncertainty_map.shape[:-2], -1)
    k = max(1, int(np.ceil(flat.shape[-1] * topk_fraction)))
    topk = np.partition(flat, flat.shape[-1] - k, axis=-1)[..., -k:]
    return topk.mean(axis=-1)
