"""Shared helpers for uncertainty-map aggregation."""

import numpy as np


def entropy_map(probs: np.ndarray) -> np.ndarray:
    """Compute entropy along the last dimension."""
    eps = 1e-10
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def target_class_mask(
    probs: np.ndarray,
    target_classes: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray | None:
    """Return a [B, H, W] mask for predicted classes of interest."""
    if not target_classes:
        return None
    predicted = np.argmax(probs, axis=-1)
    return np.isin(predicted, list(target_classes))


def aggregate_uncertainty_map(
    uncertainty_map: np.ndarray,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Reduce [B, H, W] or [T, B, H, W] maps to image-level scores."""
    if target_mask is not None:
        return _aggregate_masked_uncertainty_map(
            uncertainty_map,
            target_mask,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
        )
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


def _aggregate_masked_uncertainty_map(
    uncertainty_map: np.ndarray,
    target_mask: np.ndarray,
    *,
    aggregation: str,
    topk_fraction: float,
) -> np.ndarray:
    if aggregation not in {"mean", "topk_mean", "max"}:
        raise ValueError(
            f"aggregation must be 'mean', 'topk_mean', or 'max', got {aggregation!r}"
        )

    target_mask = np.asarray(target_mask, dtype=bool)
    if target_mask.shape != uncertainty_map.shape:
        if (
            uncertainty_map.ndim == target_mask.ndim + 1
            and target_mask.shape == uncertainty_map.shape[1:]
        ):
            target_mask = target_mask[np.newaxis, ...]
        target_mask = np.broadcast_to(target_mask, uncertainty_map.shape)

    flat_values = uncertainty_map.reshape(-1, uncertainty_map.shape[-2] * uncertainty_map.shape[-1])
    flat_mask = target_mask.reshape(flat_values.shape)
    scores: list[float] = []
    for row, mask in zip(flat_values, flat_mask, strict=True):
        selected = row[mask]
        if selected.size == 0:
            scores.append(0.0)
        elif aggregation == "mean":
            scores.append(float(selected.mean()))
        elif aggregation == "max":
            scores.append(float(selected.max()))
        else:
            k = max(1, int(np.ceil(selected.size * topk_fraction)))
            topk = np.partition(selected, selected.size - k)[-k:]
            scores.append(float(topk.mean()))
    return np.asarray(scores, dtype=uncertainty_map.dtype).reshape(
        uncertainty_map.shape[:-2]
    )
