"""BALD uncertainty provider built on MC-dropout predictions."""

from typing import Callable

import numpy as np
from tqdm import tqdm

from active_learning.providers.aggregation import (
    aggregate_uncertainty_map,
    entropy_map,
    target_class_mask,
)
from active_learning.providers.inference import (
    build_infer_fn,
    collect_mc_probs,
    iter_image_batches,
)
from active_learning.providers.unet import enable_mc_dropout


def bald_scores_for_batch(
    infer,
    batch,
    n_iterations: int,
    *,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    target_classes: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Image-level BALD scores for a single TF batch ``batch`` (shape [B, H, W, C])."""
    mc_probs = collect_mc_probs(infer, batch, n_iterations)
    prob_mean = mc_probs.mean(axis=0)
    predictive_entropy = entropy_map(prob_mean)
    expected_entropy = entropy_map(mc_probs).mean(axis=0)
    bald_map = np.maximum(predictive_entropy - expected_entropy, 0.0)
    per_image = aggregate_uncertainty_map(
        bald_map,
        aggregation=aggregation,
        topk_fraction=topk_fraction,
        target_mask=target_class_mask(prob_mean, target_classes),
    )
    return np.asarray(per_image, dtype=np.float32).reshape(-1)


def make_bald_score_batch_fn(
    mc_unet,
    n_iterations: int,
    *,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    target_classes: list[int] | tuple[int, ...] | None = None,
):
    """Return ``score_batch(batch_tensor) -> np.ndarray`` sharing one ``infer`` fn."""
    model = mc_unet.model
    infer = build_infer_fn(model, training=True)

    def score_batch(batch):
        return bald_scores_for_batch(
            infer,
            batch,
            n_iterations,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
            target_classes=target_classes,
        )

    return score_batch


def bald_provider(
    unet,
    n_iterations: int = 5,
    image_size: tuple[int, int] = (320, 240),
    batch_size: int = 8,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    target_classes: list[int] | tuple[int, ...] | None = None,
    *,
    progress: bool = False,
) -> Callable[[list[str]], np.ndarray]:
    """Create a BALD uncertainty provider using MC dropout."""
    mc_unet = enable_mc_dropout(unet)

    def provider(image_paths: list[str]) -> np.ndarray:
        return compute_bald_uncertainty(
            mc_unet,
            image_paths,
            n_iterations,
            image_size,
            batch_size,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
            target_classes=target_classes,
            progress=progress,
        )

    return provider


def compute_bald_uncertainty(
    mc_unet,
    image_paths,
    n_iterations,
    image_size,
    batch_size,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    target_classes: list[int] | tuple[int, ...] | None = None,
    *,
    progress: bool = False,
) -> np.ndarray:
    """Compute image-level BALD scores from MC-dropout predictions."""
    model = mc_unet.model
    infer = build_infer_fn(model, training=True)
    all_uncertainties = []

    batches = iter_image_batches(image_paths, batch_size, image_size)
    if progress:
        n_tf = (len(image_paths) + batch_size - 1) // batch_size
        batches = tqdm(batches, total=n_tf, desc="BALD", leave=True)
    for batch in batches:
        per_image = bald_scores_for_batch(
            infer,
            batch,
            n_iterations,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
            target_classes=target_classes,
        )
        all_uncertainties.extend(per_image.tolist())

    return np.array(all_uncertainties, dtype=np.float32).reshape(-1, 1)
