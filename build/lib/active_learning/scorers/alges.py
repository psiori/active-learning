"""ALGES embedding scorer keyed by sample ID."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from active_learning.core.image_provider import ImageProvider
from active_learning.providers.inference import extract_probs_and_penultimate
from active_learning.providers.unet import unet_cache_namespace
from active_learning.scorers._cache import (
    load_array,
    sample_npy_path,
    save_array,
)
from active_learning.scorers.uncertainty import _materialize_paths_ordered
from active_learning.strategies.alges import compute_gradient_embeddings_batch
from active_learning.core.types import SampleId


def score_alges_embeddings(
    sample_ids: list[SampleId],
    image_provider: ImageProvider,
    cache_root: str | Path,
    *,
    model,
    model_name: str | None = None,
    model_path: str | None = None,
    image_size: tuple[int, int] = (320, 240),
    method: str = "semantic",
    batch_size: int = 8,
    use_highres: bool = False,
    progress: bool = False,
    on_step: Callable[[int, int], None] | None = None,
) -> dict[SampleId, np.ndarray]:
    model_namespace = unet_cache_namespace(
        model,
        model_name=model_name,
        model_path=model_path,
    )
    namespace = f"scorers/alges/{model_namespace}/{method}"
    embeddings: dict[SampleId, np.ndarray] = {}
    missing_ids: list[SampleId] = []
    n_total = len(sample_ids)

    sample_iter = sample_ids
    if progress:
        sample_iter = tqdm(
            sample_ids,
            desc="ALGES cache",
            unit="img",
        )
    for sample_id in sample_iter:
        cache_path = sample_npy_path(cache_root, namespace, sample_id)
        cached = load_array(cache_path)
        if cached is None:
            missing_ids.append(sample_id)
            continue
        embeddings[sample_id] = cached

    n_cached = n_total - len(missing_ids)

    if not missing_ids:
        if on_step is not None:
            on_step(n_total, max(n_total, 1))
        return embeddings

    ok_ids, all_paths = _materialize_paths_ordered(
        missing_ids,
        image_provider,
        use_highres=use_highres,
    )
    if not ok_ids:
        if on_step is not None:
            on_step(n_cached, max(n_total, 1))
        return embeddings

    infer_on_step = None
    if on_step is not None:
        on_step(n_cached, max(n_total, 1))

        def infer_on_step(done: int, _total_paths: int) -> None:
            on_step(n_cached + done, max(n_total, 1))

    probs, penultimate = extract_probs_and_penultimate(
        model,
        all_paths,
        image_size=image_size,
        batch_size=batch_size,
        progress=progress,
        on_step=infer_on_step,
    )
    batch_embeddings = compute_gradient_embeddings_batch(
        probs,
        penultimate,
        method=method,
    )
    embedding_iter = zip(ok_ids, batch_embeddings, strict=True)
    if progress:
        embedding_iter = tqdm(
            embedding_iter,
            total=len(ok_ids),
            desc="ALGES embeddings",
            unit="img",
        )
    for sample_id, embedding in embedding_iter:
        embedding = np.asarray(embedding, dtype=np.float32)
        save_array(sample_npy_path(cache_root, namespace, sample_id), embedding)
        embeddings[sample_id] = embedding

    return embeddings
