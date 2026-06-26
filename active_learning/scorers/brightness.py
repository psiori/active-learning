"""Brightness scorer operating on sample IDs via an ImageProvider."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from active_learning.core.image_provider import ImageProvider, _pool_max_workers
from active_learning.scorers._cache import load_json, sample_json_path, save_json
from active_learning.scorers.features import compute_image_stats
from active_learning.core.types import SampleId


BRIGHTNESS_NAMESPACE = "scorers/brightness"

# Persisted when low-res bytes cannot be decoded; keeps float-only values for loaders.
_UNREADABLE_BRIGHTNESS_CACHE = {
    "_unreadable": 1.0,
    "brightness_mean": -1.0,
    "brightness_std": 0.0,
}


def _is_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    return status_code == 404


def _lowres_paths_parallel(
    image_provider: ImageProvider,
    sample_ids: list[SampleId],
    *,
    progress: bool,
) -> dict[SampleId, str | None]:
    unique = list(dict.fromkeys(sample_ids))
    if not unique:
        return {}
    multiplicity = Counter(sample_ids)
    max_workers = _pool_max_workers(len(unique))

    def load_one(sid: SampleId) -> tuple[SampleId, str | None]:
        try:
            return sid, image_provider.get_lowres(sid)
        except FileNotFoundError:
            return sid, None
        except Exception as exc:
            if _is_not_found_error(exc):
                return sid, None
            raise

    out: dict[SampleId, str | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sid = {executor.submit(load_one, sid): sid for sid in unique}
        pbar = None
        if progress:
            pbar = tqdm(
                total=len(sample_ids),
                desc="Low-res images",
                unit="img",
                leave=True,
            )
        for fut in as_completed(future_to_sid):
            sid, path = fut.result()
            out[sid] = path
            if pbar is not None:
                pbar.update(multiplicity[sid])
        if pbar is not None:
            pbar.close()
    return out


def score_brightness(
    sample_ids: list[SampleId],
    image_provider: ImageProvider,
    cache_root: str | Path,
    *,
    progress: bool = False,
) -> dict[SampleId, dict[str, float]]:
    stats_by_id: dict[SampleId, dict[str, float]] = {}
    missing_ids: list[SampleId] = []

    for sample_id in sample_ids:
        cache_path = sample_json_path(cache_root, BRIGHTNESS_NAMESPACE, sample_id)
        cached = load_json(cache_path)
        if cached is None:
            missing_ids.append(sample_id)
            continue
        stats_by_id[sample_id] = {k: float(v) for k, v in cached.items()}

    if missing_ids:
        paths_by_id = _lowres_paths_parallel(
            image_provider,
            missing_ids,
            progress=progress,
        )
        for sample_id in missing_ids:
            if sample_id in stats_by_id:
                continue
            image_path = paths_by_id.get(sample_id)
            if image_path is None:
                save_json(
                    sample_json_path(cache_root, BRIGHTNESS_NAMESPACE, sample_id),
                    dict(_UNREADABLE_BRIGHTNESS_CACHE),
                )
                stats_by_id[sample_id] = {
                    k: float(v) for k, v in _UNREADABLE_BRIGHTNESS_CACHE.items()
                }
                continue
            try:
                with Image.open(image_path) as img:
                    rgb = img.convert("RGB")
                    img_array = np.asarray(rgb, dtype=np.float32)
                stats = {
                    key: float(value)
                    for key, value in compute_image_stats(img_array).items()
                }
            except (OSError, ValueError, UnidentifiedImageError):
                save_json(
                    sample_json_path(cache_root, BRIGHTNESS_NAMESPACE, sample_id),
                    dict(_UNREADABLE_BRIGHTNESS_CACHE),
                )
                stats_by_id[sample_id] = {
                    k: float(v) for k, v in _UNREADABLE_BRIGHTNESS_CACHE.items()
                }
                continue
            save_json(
                sample_json_path(cache_root, BRIGHTNESS_NAMESPACE, sample_id),
                stats,
            )
            stats_by_id[sample_id] = stats

    return stats_by_id


def filter_by_brightness(
    sample_ids: list[SampleId],
    image_provider: ImageProvider,
    cache_root: str | Path,
    *,
    min_brightness: float = 0.0,
    max_brightness: float = 220.0,
    progress: bool = False,
) -> tuple[list[SampleId], dict[SampleId, dict[str, float]]]:
    """Filter sample IDs by mean brightness using cached brightness statistics."""
    if min_brightness <= 0 and max_brightness <= 0:
        return list(sample_ids), {}

    stats = score_brightness(sample_ids, image_provider, cache_root, progress=progress)
    filtered_ids = [
        sample_id
        for sample_id in sample_ids
        if (
            not stats[sample_id].get("_unreadable")
            and (
                min_brightness <= 0
                or stats[sample_id]["brightness_mean"] >= min_brightness
            )
            and (
                max_brightness <= 0
                or stats[sample_id]["brightness_mean"] <= max_brightness
            )
        )
    ]
    return filtered_ids, stats
