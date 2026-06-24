"""Single-pass predictive-entropy uncertainty provider."""

from typing import Callable

import numpy as np
from tqdm import tqdm

from active_learning.providers.aggregation import (
    aggregate_uncertainty_map,
    entropy_map,
)
from active_learning.providers.inference import (
    build_infer_fn,
    extract_probs,
    iter_image_batches,
)


def entropy_scores_for_batch(
    infer,
    batch,
    *,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
) -> np.ndarray:
    """Single-pass predictive entropy per image for one TF batch."""
    result = infer(batch)
    probs = extract_probs(result).numpy()
    entropy = entropy_map(probs)
    per_image = aggregate_uncertainty_map(
        entropy,
        aggregation=aggregation,
        topk_fraction=topk_fraction,
    )
    return np.asarray(per_image, dtype=np.float32).reshape(-1)


def make_entropy_score_batch_fn(
    unet,
    *,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
):
    model = unet.model
    infer = build_infer_fn(model, training=False)

    def score_batch(batch):
        return entropy_scores_for_batch(
            infer,
            batch,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
        )

    return score_batch


def entropy_provider(
    unet,
    image_size: tuple[int, int] = (320, 240),
    batch_size: int = 16,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    *,
    progress: bool = False,
) -> Callable[[list[str]], np.ndarray]:
    """Create a single-pass entropy uncertainty provider."""

    def provider(image_paths: list[str]) -> np.ndarray:
        return compute_entropy_uncertainty(
            unet,
            image_paths,
            image_size,
            batch_size,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
            progress=progress,
        )

    return provider


def compute_entropy_uncertainty(
    unet,
    image_paths,
    image_size,
    batch_size,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    *,
    progress: bool = False,
) -> np.ndarray:
    model = unet.model
    infer = build_infer_fn(model, training=False)
    all_uncertainties = []

    batches = iter_image_batches(image_paths, batch_size, image_size)
    if progress:
        n_tf = (len(image_paths) + batch_size - 1) // batch_size
        batches = tqdm(batches, total=n_tf, desc="Entropy", leave=True)
    for batch in batches:
        per_image = entropy_scores_for_batch(
            infer,
            batch,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
        )
        all_uncertainties.extend(per_image.tolist())

    return np.array(all_uncertainties, dtype=np.float32).reshape(-1, 1)
