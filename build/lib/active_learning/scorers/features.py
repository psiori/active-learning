"""Feature extraction and embedding scorer keyed by sample ID."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from active_learning.core.image_provider import ImageProvider
from active_learning.scorers._cache import (
    load_array,
    sample_npy_path,
    save_array,
)
from active_learning.core.types import SampleId


def get_default_extractor() -> Callable[[np.ndarray], np.ndarray]:
    """Return a ResNet50 feature extractor (2048-d embeddings)."""
    model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
    )
    return model


def compute_image_stats(img_array: np.ndarray) -> dict:
    """Compute cheap pixel-level statistics from an image array."""
    gray = np.mean(img_array, axis=2)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    n_pixels = gray.shape[0] * gray.shape[1]
    saturated = np.sum((gray < 5) | (gray > 250)) / n_pixels
    gray_uint8 = gray.astype(np.uint8)
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    return {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "channel_mean_r": float(r.mean()),
        "channel_mean_g": float(g.mean()),
        "channel_mean_b": float(b.mean()),
        "saturated_pct": float(saturated),
        "laplacian_var": float(laplacian.var()),
    }


STAT_KEYS = [
    "brightness_mean",
    "brightness_std",
    "channel_mean_r",
    "channel_mean_g",
    "channel_mean_b",
    "saturated_pct",
    "laplacian_var",
]


def extract_features(
    image_paths: list[str],
    extractor: Optional[Callable] = None,
    preprocess_fn=None,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    *,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature vectors and image stats for a list of images."""
    if extractor is None:
        extractor = get_default_extractor()

    all_features = []
    all_stats = []
    batches = range(0, len(image_paths), batch_size)
    if progress:
        batches = tqdm(batches, desc="Extracting features")
    for i in batches:
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        batch_stats = []
        for path in batch_paths:
            img = tf.keras.utils.load_img(path, target_size=image_size)
            img_array = tf.keras.utils.img_to_array(img)
            batch_stats.append(compute_image_stats(img_array))
            batch_images.append(img_array)

        batch = np.stack(batch_images, axis=0)
        if preprocess_fn is not None:
            batch = preprocess_fn(batch)
        else:
            batch = tf.keras.applications.resnet50.preprocess_input(batch)
        features = extractor(batch, training=False)
        if hasattr(features, "numpy"):
            features = features.numpy()
        all_features.append(features)

        stats_array = np.array(
            [[s[k] for k in STAT_KEYS] for s in batch_stats],
            dtype=np.float32,
        )
        all_stats.append(stats_array)

    return np.concatenate(all_features, axis=0), np.concatenate(all_stats, axis=0)


def score_features(
    sample_ids: list[SampleId],
    image_provider: ImageProvider,
    cache_root: str | Path,
    *,
    model_name: str = "resnet50",
    extractor: Callable | None = None,
    preprocess_fn=None,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    use_highres: bool = False,
    progress: bool = False,
    on_step: Callable[[int, int], None] | None = None,
) -> dict[SampleId, np.ndarray]:
    namespace = f"scorers/features/{model_name}"
    features_by_id: dict[SampleId, np.ndarray] = {}
    missing_ids: list[SampleId] = []

    for sample_id in sample_ids:
        cache_path = sample_npy_path(cache_root, namespace, sample_id)
        cached = load_array(cache_path)
        if cached is None:
            missing_ids.append(sample_id)
            continue
        features_by_id[sample_id] = cached

    n_total = len(sample_ids)
    n_miss = len(missing_ids)
    n_cached = n_total - n_miss
    if on_step is not None and n_miss == 0:
        on_step(n_total, max(n_total, 1))

    if on_step is not None and n_miss > 0:
        on_step(n_cached, max(n_total, 1))

    processed = 0
    for start in range(0, len(missing_ids), batch_size):
        batch_ids = missing_ids[start : start + batch_size]
        if not batch_ids:
            continue
        batch_paths = (
            image_provider.get_highres_batch(
                batch_ids, progress=progress, allow_missing=True
            )
            if use_highres
            else image_provider.get_lowres_batch(
                batch_ids, progress=progress, allow_missing=True
            )
        )
        present = [(sid, p) for sid, p in zip(batch_ids, batch_paths, strict=True) if p]
        if not present:
            continue
        ok_ids, paths_ok = zip(*present)
        ok_ids, paths_ok = list(ok_ids), list(paths_ok)
        batch_features, _ = extract_features(
            paths_ok,
            extractor=extractor,
            preprocess_fn=preprocess_fn,
            batch_size=batch_size,
            image_size=image_size,
            progress=progress,
        )
        for sample_id, feature in zip(ok_ids, batch_features, strict=True):
            feature = np.asarray(feature, dtype=np.float32)
            save_array(sample_npy_path(cache_root, namespace, sample_id), feature)
            features_by_id[sample_id] = feature

        processed += len(ok_ids)
        if on_step is not None:
            on_step(n_cached + processed, max(n_total, 1))

    return features_by_id
