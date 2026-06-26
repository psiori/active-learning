"""Uncertainty scorers backed by model providers."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from active_learning.core.image_provider import ImageProvider
from active_learning.providers.bald import make_bald_score_batch_fn
from active_learning.providers.entropy import make_entropy_score_batch_fn
from active_learning.providers.inference import iter_image_batches
from active_learning.providers.mc_dropout import make_mc_dropout_score_batch_fn
from active_learning.providers.unet import unet_cache_namespace
from active_learning.scorers._cache import (
    load_json,
    sample_json_path,
    save_json,
)
from active_learning.core.types import SampleId

# Chunk size for materializing local paths before building one ``tf.data`` pipeline.
_MATERIALIZE_CHUNK_SIZE = 2048


def _materialize_paths_ordered(
    sample_ids: list[SampleId],
    image_provider: ImageProvider,
    *,
    use_highres: bool,
) -> tuple[list[SampleId], list[str]]:
    get_batch = (
        functools.partial(image_provider.get_highres_batch, allow_missing=True)
        if use_highres
        else functools.partial(image_provider.get_lowres_batch, allow_missing=True)
    )
    ok_ids: list[SampleId] = []
    paths: list[str] = []
    n = len(sample_ids)
    for start in range(0, n, _MATERIALIZE_CHUNK_SIZE):
        chunk = sample_ids[start : start + _MATERIALIZE_CHUNK_SIZE]
        chunk_paths = get_batch(chunk, progress=False)
        for sid, path in zip(chunk, chunk_paths, strict=True):
            if path is not None:
                ok_ids.append(sid)
                paths.append(path)
    return ok_ids, paths


def score_entropy(
    sample_ids: list[SampleId],
    image_provider: ImageProvider,
    cache_root: str | Path,
    *,
    unet,
    model_name: str | None = None,
    model_path: str | None = None,
    image_size: tuple[int, int] = (320, 240),
    batch_size: int = 16,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    target_classes: list[int] | tuple[int, ...] | None = None,
    use_highres: bool = False,
    progress: bool = False,
    on_step: Callable[[int, int], None] | None = None,
) -> dict[SampleId, float]:
    model_namespace = unet_cache_namespace(
        unet,
        model_name=model_name,
        model_path=model_path,
    )
    return _score_uncertainty(
        sample_ids=sample_ids,
        image_provider=image_provider,
        cache_root=cache_root,
        namespace=_uncertainty_namespace(
            model_namespace,
            "entropy",
            target_classes=target_classes,
        ),
        batch_size=batch_size,
        image_size=image_size,
        use_highres=use_highres,
        progress_label="Entropy",
        progress=progress,
        on_step=on_step,
        score_batch_factory=lambda: make_entropy_score_batch_fn(
            unet,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
            target_classes=target_classes,
        ),
    )


def score_mc_dropout(
    sample_ids: list[SampleId],
    image_provider: ImageProvider,
    cache_root: str | Path,
    *,
    mc_unet,
    model_name: str | None = None,
    model_path: str | None = None,
    iterations: int = 5,
    image_size: tuple[int, int] = (320, 240),
    batch_size: int = 8,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    target_classes: list[int] | tuple[int, ...] | None = None,
    use_highres: bool = False,
    progress: bool = False,
    on_step: Callable[[int, int], None] | None = None,
) -> dict[SampleId, float]:
    model_namespace = unet_cache_namespace(
        mc_unet,
        model_name=model_name,
        model_path=model_path,
    )
    return _score_uncertainty(
        sample_ids=sample_ids,
        image_provider=image_provider,
        cache_root=cache_root,
        namespace=_uncertainty_namespace(
            model_namespace,
            f"mc_dropout_t{iterations}",
            target_classes=target_classes,
        ),
        batch_size=batch_size,
        image_size=image_size,
        use_highres=use_highres,
        progress_label=f"MC dropout (T={iterations})",
        progress=progress,
        on_step=on_step,
        score_batch_factory=lambda: make_mc_dropout_score_batch_fn(
            mc_unet,
            iterations,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
            target_classes=target_classes,
        ),
    )


def score_bald(
    sample_ids: list[SampleId],
    image_provider: ImageProvider,
    cache_root: str | Path,
    *,
    mc_unet,
    model_name: str | None = None,
    model_path: str | None = None,
    iterations: int = 5,
    image_size: tuple[int, int] = (320, 240),
    batch_size: int = 8,
    aggregation: str = "topk_mean",
    topk_fraction: float = 0.10,
    target_classes: list[int] | tuple[int, ...] | None = None,
    use_highres: bool = False,
    progress: bool = False,
    on_step: Callable[[int, int], None] | None = None,
) -> dict[SampleId, float]:
    model_namespace = unet_cache_namespace(
        mc_unet,
        model_name=model_name,
        model_path=model_path,
    )
    return _score_uncertainty(
        sample_ids=sample_ids,
        image_provider=image_provider,
        cache_root=cache_root,
        namespace=_uncertainty_namespace(
            model_namespace,
            f"bald_t{iterations}",
            target_classes=target_classes,
        ),
        batch_size=batch_size,
        image_size=image_size,
        use_highres=use_highres,
        progress_label=f"BALD (T={iterations} MC passes)",
        progress=progress,
        on_step=on_step,
        score_batch_factory=lambda: make_bald_score_batch_fn(
            mc_unet,
            iterations,
            aggregation=aggregation,
            topk_fraction=topk_fraction,
            target_classes=target_classes,
        ),
    )


def _score_uncertainty(
    *,
    sample_ids: list[SampleId],
    image_provider: ImageProvider,
    cache_root: str | Path,
    namespace: str,
    batch_size: int,
    image_size: tuple[int, int],
    use_highres: bool,
    progress_label: str,
    progress: bool,
    score_batch_factory: Callable[[], Callable[[Any], np.ndarray]],
    on_step: Callable[[int, int], None] | None = None,
) -> dict[SampleId, float]:
    scores: dict[SampleId, float] = {}
    missing_ids: list[SampleId] = []
    n_total = len(sample_ids)

    for sample_id in sample_ids:
        cache_path = sample_json_path(cache_root, namespace, sample_id)
        cached = load_json(cache_path)
        if cached is None:
            missing_ids.append(sample_id)
            continue
        scores[sample_id] = float(cached["score"])

    n_cached = n_total - len(missing_ids)

    if not missing_ids:
        if on_step is not None:
            on_step(n_total, max(n_total, 1))
        return scores

    ok_ids, all_paths = _materialize_paths_ordered(
        missing_ids,
        image_provider,
        use_highres=use_highres,
    )
    if not ok_ids:
        if on_step is not None:
            on_step(n_cached, max(n_total, 1))
        return scores

    n_ok = len(ok_ids)
    if on_step is not None:
        on_step(n_cached, max(n_total, 1))

    score_batch = score_batch_factory()
    dataset = iter_image_batches(all_paths, batch_size, image_size)

    idx = 0
    bar = None
    try:
        if progress:
            bar = tqdm(
                total=n_ok,
                desc=progress_label,
                unit="img",
                leave=True,
            )
        for batch in dataset:
            scores_1d = np.asarray(score_batch(batch), dtype=np.float64).reshape(-1)
            n_batch = int(scores_1d.shape[0])
            batch_ids = ok_ids[idx : idx + n_batch]
            if len(batch_ids) != n_batch:
                raise RuntimeError(
                    f"Batch/id alignment error: got {n_batch} scores but "
                    f"{len(batch_ids)} ids (cursor idx={idx}).",
                )
            for sample_id, score_value in zip(batch_ids, scores_1d, strict=True):
                value = float(score_value)
                save_json(
                    sample_json_path(cache_root, namespace, sample_id),
                    {"score": value},
                )
                scores[sample_id] = value
            idx += n_batch
            if bar is not None:
                bar.update(n_batch)
            if on_step is not None:
                on_step(n_cached + idx, max(n_total, 1))
    finally:
        if bar is not None:
            bar.close()

    if idx != n_ok:
        raise RuntimeError(
            f"Scored {idx} ids but expected {n_ok} (namespace={namespace!r}).",
        )

    return scores


def _uncertainty_namespace(
    model_namespace: str,
    provider_segment: str,
    *,
    target_classes: list[int] | tuple[int, ...] | None = None,
) -> str:
    class_segment = "all_classes"
    if target_classes:
        class_segment = "classes_" + "_".join(str(int(c)) for c in target_classes)
    return f"scorers/uncertainty/{model_namespace}/{provider_segment}/{class_segment}"
