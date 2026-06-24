"""ALGES: Active Learning with Gradient Embeddings for Segmentation.

Adapts BADGE for segmentation by computing per-pixel gradient embeddings
from the last layer of a segmentation network and aggregating them into
per-image embeddings. Uses k-means++ initialization for batch selection.

Reference: Aklilu & Yeung, "ALGES: Active Learning with Gradient Embeddings
for Semantic Segmentation of Laparoscopic Surgical Images", MLHC 2022.

The per-pixel gradient of cross-entropy loss w.r.t. last layer weights is:
    dL/dw_k at pixel (i,j) = (p_ijk - I(k == y_hat_ij)) * z_ij

where p_ijk is the softmax probability, y_hat_ij is the predicted class,
and z_ij is the penultimate layer activation. This is analytical — no
autograd needed.
"""

import logging

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_gradient_embeddings(
    probs: np.ndarray,
    penultimate: np.ndarray,
    method: str = "image",
) -> np.ndarray:
    """Compute gradient embeddings for one image.

    Args:
        probs: [H, W, K] softmax probabilities from the last layer.
        penultimate: [H, W, d] activations from the penultimate layer.
        method: "image" for image-level (Eq. 4, average across pixels)
                or "semantic" for semantic-level (Eq. 6, group by class).

    Returns:
        Gradient embedding vector of shape [K * d].
    """
    H, W, K = probs.shape
    d = penultimate.shape[2]

    # Predicted class per pixel
    y_hat = np.argmax(probs, axis=2)  # [H, W]

    # One-hot encoding of predictions: [H, W, K]
    one_hot = np.eye(K)[y_hat]

    # Per-pixel gradient factor: (p - one_hot_y) for each class k -> [H, W, K]
    grad_factor = probs - one_hot

    if method == "image":
        # Eq. 4: average gradient across all pixels per class
        # For each class k: (1/HW) * sum_ij (p_ijk - I(k=y_hat)) * z_ij
        # Result: [K, d]
        embedding = np.zeros((K, d), dtype=np.float32)
        for k in range(K):
            # [H, W, 1] * [H, W, d] -> [H, W, d], then mean over H,W
            embedding[k] = np.mean(
                grad_factor[:, :, k : k + 1] * penultimate, axis=(0, 1)
            )

    elif method == "semantic":
        # Eq. 6: sum gradients per predicted class
        # For class k: sum over pixels where y_hat == k
        embedding = np.zeros((K, d), dtype=np.float32)
        for k in range(K):
            mask = y_hat == k  # [H, W]
            if mask.any():
                # Sum the full per-pixel gradient embedding for pixels of class k
                # Each pixel's embedding is [(p_1 - I(1=k))*z, ..., (p_K - I(K=k))*z]
                # but Eq. 6 sums only the k-th block for pixels predicted as class k
                embedding[k] = np.sum(
                    grad_factor[:, :, k : k + 1][mask] * penultimate[mask],
                    axis=0,
                )

    else:
        raise ValueError(f"method must be 'image' or 'semantic', got '{method}'")

    return embedding.ravel()  # [K * d]


def compute_gradient_embeddings_batch(
    all_probs: list[np.ndarray],
    all_penultimate: list[np.ndarray],
    method: str = "image",
) -> np.ndarray:
    """Compute gradient embeddings for a batch of images.

    Args:
        all_probs: List of [H, W, K] softmax probability arrays.
        all_penultimate: List of [H, W, d] penultimate activation arrays.
        method: "image" or "semantic".

    Returns:
        np.ndarray of shape [N, K*d] where N is the number of images.
    """
    embeddings = []
    for probs, penultimate in zip(all_probs, all_penultimate):
        emb = compute_gradient_embeddings(probs, penultimate, method=method)
        embeddings.append(emb)
    return np.stack(embeddings, axis=0)


def kmeans_pp_select(
    embeddings: np.ndarray,
    n: int,
    seed: int | None = None,
    *,
    progress: bool = False,
) -> list[int]:
    """Select n points using k-means++ initialization.

    k-means++ naturally selects points that are both high-magnitude
    (uncertain) and spread out (diverse), making it ideal for BADGE/ALGES.

    Points are selected with probability proportional to their squared
    distance to the nearest already-selected point.

    Args:
        embeddings: [N, D] feature matrix (gradient embeddings).
        n: Number of points to select.
        seed: Random seed.

    Returns:
        List of n indices into embeddings.
    """
    rng = np.random.RandomState(seed)
    N = embeddings.shape[0]
    n = min(n, N)

    # First point: probability proportional to squared norm
    # (high-magnitude embeddings = uncertain images get picked first)
    norms_sq = np.sum(embeddings**2, axis=1)
    total = norms_sq.sum()
    if total < 1e-12:
        # All embeddings are zero (model is fully confident on everything)
        return list(rng.choice(N, size=n, replace=False))

    probs = norms_sq / total
    first = rng.choice(N, p=probs)
    selected = [first]

    # Min squared distance to nearest selected point
    min_dist_sq = np.sum((embeddings - embeddings[first]) ** 2, axis=1)

    picks = range(n - 1)
    if progress:
        picks = tqdm(
            picks,
            desc="ALGES k-means++",
            unit="pick",
        )
    for _ in picks:
        # Select next point proportional to min_dist_sq
        total = min_dist_sq.sum()
        if total < 1e-12:
            # All remaining points are at distance 0
            remaining = [i for i in range(N) if i not in set(selected)]
            selected.extend(
                rng.choice(remaining, size=n - len(selected), replace=False).tolist()
            )
            break

        probs = min_dist_sq / total
        idx = rng.choice(N, p=probs)
        selected.append(idx)

        # Update min distances
        new_dist_sq = np.sum((embeddings - embeddings[idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)

    return selected


def alges_select(
    all_probs: list[np.ndarray],
    all_penultimate: list[np.ndarray],
    n: int,
    method: str = "image",
    seed: int | None = None,
    *,
    progress: bool = False,
) -> list[int]:
    """Full ALGES pipeline: compute gradient embeddings then k-means++ select.

    Args:
        all_probs: List of [H, W, K] softmax outputs per image.
        all_penultimate: List of [H, W, d] penultimate activations per image.
        n: Number of images to select.
        method: "image" or "semantic" gradient aggregation.
        seed: Random seed.

    Returns:
        List of n indices into the input lists.
    """
    logger.info("Computing gradient embeddings (%s-level)...", method)
    embeddings = []
    gradient_iter = zip(all_probs, all_penultimate)
    if progress:
        gradient_iter = tqdm(
            gradient_iter,
            total=len(all_probs),
            desc="Gradient embeddings",
        )
    for probs, pen in gradient_iter:
        emb = compute_gradient_embeddings(probs, pen, method=method)
        embeddings.append(emb)
    embeddings = np.stack(embeddings, axis=0)

    logger.info(
        "Embeddings shape: %s, running k-means++ selection...", embeddings.shape
    )
    return kmeans_pp_select(embeddings, n, seed=seed, progress=progress)
