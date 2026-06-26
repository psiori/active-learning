"""Backend-agnostic mosaic rendering sink."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from active_learning.core.config import SeedConfig
from active_learning.core.image_provider import ImageProvider
from active_learning.providers import load_unet
from active_learning.core.types import SelectionResult


def mosaic_segment(label: str) -> str:
    """Normalize an arbitrary label into a filesystem-friendly path segment."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(label).strip())
    return s.strip("_") or "x"


def mosaic_output_path(cfg: SeedConfig) -> str:
    """Path for the selection mosaic under the configured output directory."""
    base_dir = os.path.dirname(cfg.export.mosaic_path) or "."
    base_dir = os.path.join(base_dir, "mosaics")
    project = cfg.export.project or cfg.export.prefix or "project"
    name = (
        f"mosaic_{mosaic_segment(project)}_{mosaic_segment(cfg.query.start)}_"
        f"{mosaic_segment(cfg.query.end)}_{mosaic_segment(cfg.selection.n_select)}_"
        f"{mosaic_segment(cfg.selection.strategy)}.jpg"
    )
    return os.path.join(base_dir, name)


def overlay_mosaic_output_path(cfg: SeedConfig) -> str:
    """Return the default output path for the prediction-overlay mosaic image."""
    base_dir = os.path.dirname(cfg.export.mosaic_path) or "."
    base_dir = os.path.join(base_dir, "mosaics_overlay")
    project = cfg.export.project or cfg.export.prefix or "project"
    name = (
        f"mosaic_{mosaic_segment(project)}_{mosaic_segment(cfg.query.start)}_"
        f"{mosaic_segment(cfg.query.end)}_{mosaic_segment(cfg.selection.n_select)}_"
        f"{mosaic_segment(cfg.selection.strategy)}_overlay.jpg"
    )
    return os.path.join(base_dir, name)


def prediction_mask_output_dir(cfg: SeedConfig) -> str:
    """Return the directory used to store per-image overlay mask PNGs."""
    base_dir = os.path.dirname(cfg.export.mosaic_path) or "."
    return os.path.join(base_dir, "overlay_masks")


def save_prediction_mask(mask_path: str, probs: np.ndarray) -> None:
    """Save a predicted segmentation mask as a transparent RGBA overlay."""
    if probs.ndim != 3:
        raise ValueError(f"Expected probs with shape [H, W, K], got {probs.shape}")

    labels = np.argmax(probs, axis=-1).astype(np.uint8)
    palette = np.array(
        [
            (0, 0, 0, 0),
            (255, 99, 71, 255),
            (0, 180, 0, 255),
            (30, 144, 255, 255),
            (255, 215, 0, 255),
            (186, 85, 211, 255),
            (0, 206, 209, 255),
            (255, 140, 0, 255),
        ],
        dtype=np.uint8,
    )
    rgba = palette[labels % len(palette)]
    Image.fromarray(rgba, mode="RGBA").save(mask_path)


def render_mosaic(
    result: SelectionResult,
    image_provider: ImageProvider,
    output_path: str | Path,
    *,
    use_highres: bool = False,
    resize_height: int = 150,
    max_images: int | None = 200,
    rows: int | None = None,
    cols: int | None = None,
    progress: bool = False,
) -> str:
    """Render a grid mosaic for the selected images and write it to disk.

    Images are loaded through ``image_provider``, resized to a common target
    height, centered in uniformly sized tiles, and composited onto a dark
    background. When ``rows`` and ``cols`` are not fully specified, the layout
    is inferred from the number of images.
    """
    sample_ids = sorted(result.selected_ids)[: max_images or len(result.selected_ids)]
    if not sample_ids:
        raise ValueError("Cannot render a mosaic for an empty selection")

    raw_paths = (
        image_provider.get_highres_batch(
            sample_ids, progress=progress, allow_missing=True
        )
        if use_highres
        else image_provider.get_lowres_batch(
            sample_ids, progress=progress, allow_missing=True
        )
    )
    image_paths = [p for p in raw_paths if p is not None]
    if not image_paths:
        raise ValueError("No image paths could be loaded (all blobs missing on disk).")
    images = [_load_resized(path, resize_height) for path in image_paths]

    if cols is None and rows is None:
        cols = min(len(images), 10)
        rows = int(math.ceil(len(images) / cols))
    elif cols is None:
        cols = int(math.ceil(len(images) / max(rows or 1, 1)))
    elif rows is None:
        rows = int(math.ceil(len(images) / cols))

    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    mosaic = Image.new(
        "RGB",
        (cols * tile_width, rows * tile_height),
        color=(20, 20, 20),
    )

    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        tile = Image.new("RGB", (tile_width, tile_height), color=(20, 20, 20))
        x = (tile_width - image.width) // 2
        y = (tile_height - image.height) // 2
        tile.paste(image, (x, y))
        mosaic.paste(tile, (col * tile_width, row * tile_height))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(output)
    return str(output)


def _load_resized(path: str, resize_height: int) -> Image.Image:
    """Load an image, resize it to the requested height, and honor EXIF rotation."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    new_width = max(1, int(width * (resize_height / max(height, 1))))
    image = image.resize((new_width, resize_height))
    return ImageOps.exif_transpose(image)


def render_overlay_mosaic(
    result: SelectionResult,
    image_provider: ImageProvider,
    *,
    cfg,
    progress: bool = False,
) -> str:
    """Render a mosaic with predicted segmentation masks overlaid on each image.

    This helper loads the ALGES model configured in ``cfg``, predicts
    segmentation probabilities for the selected images, saves one transparent
    mask per image, and delegates the final compositing to
    ``vis.overlay_mosaic_paths``.
    """
    sample_ids = sorted(result.selected_ids)
    if not sample_ids:
        raise ValueError("Cannot render an overlay mosaic for an empty selection")

    model_def = cfg.models[cfg.alges.model]
    unet = load_unet(model_def.path)
    raw_paths = image_provider.get_highres_batch(
        sample_ids, progress=progress, allow_missing=True
    )
    kept: list[tuple[str, str]] = [
        (sid, p) for sid, p in zip(sample_ids, raw_paths, strict=True) if p
    ]
    if not kept:
        raise ValueError("No image paths for overlay (all blobs missing on disk).")
    sample_ids, image_paths = zip(*kept)
    sample_ids, image_paths = list(sample_ids), list(image_paths)
    probs_batch = _predict_overlay_probs(
        unet,
        image_paths,
        image_size=tuple(model_def.image_size),
        batch_size=cfg.alges.batch_size,
    )

    mask_dir = Path(prediction_mask_output_dir(cfg))
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_paths: list[str] = []
    for sample_id, source_path, probs in zip(
        sample_ids,
        image_paths,
        probs_batch,
        strict=True,
    ):
        stem = Path(source_path).stem or _safe_mask_name(sample_id)
        mask_path = mask_dir / f"{stem}.mask.png"
        save_prediction_mask(str(mask_path), probs)
        mask_paths.append(str(mask_path))

    output_path = overlay_mosaic_output_path(cfg)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    overlay_mosaic_paths(
        image_paths,
        mask_paths,
        str(output_path),
        resize_height=150,
        rows=10,
        cols=20,
    )
    return str(output_path)


def overlay_mask(
    image_path: str,
    mask_path: str,
    alpha: float = 0.5,
    background_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> np.ndarray:
    """Overlay an RGBA mask over an image while preserving transparent background."""
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("RGBA")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    mask_arr = np.array(mask)
    background = (mask_arr[..., 3] == 0) | np.all(
        mask_arr[..., :3] == background_color[:3],
        axis=-1,
    )
    mask_arr[..., 3] = int(max(0.0, min(alpha, 1.0)) * 255)
    mask_arr[background, 3] = 0
    mask = Image.fromarray(mask_arr, mode="RGBA")

    image.paste(mask, (0, 0), mask)
    return np.array(image.convert("RGB"))


def overlay_mosaic_paths(
    image_paths: list[str],
    mask_paths: list[str],
    save_path: str,
    resize_height: int = 416,
    resize_width: int | None = None,
    max_images: int = 100,
    rows: int | None = None,
    cols: int | None = None,
    alpha: float = 0.5,
    show_tile_labels: list[str] | None = None,
) -> None:
    """Create a mosaic from aligned image and mask paths."""
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Number of image paths ({len(image_paths)}) must match number of mask paths ({len(mask_paths)})",
        )
    if show_tile_labels is not None and len(show_tile_labels) != len(image_paths):
        raise ValueError(
            f"Number of tile labels ({len(show_tile_labels)}) must match number of paths ({len(image_paths)})",
        )

    if max_images <= 0:
        max_images = len(image_paths)
    image_paths = image_paths[:max_images]
    mask_paths = mask_paths[:max_images]
    if show_tile_labels is not None:
        show_tile_labels = show_tile_labels[:max_images]
    if not image_paths:
        raise ValueError("Cannot render an overlay mosaic for an empty path list")

    tiles = [
        Image.fromarray(overlay_mask(image_path, mask_path, alpha=alpha))
        for image_path, mask_path in zip(image_paths, mask_paths, strict=True)
    ]
    if resize_width is None:
        width, height = tiles[0].size
        resize_width = max(1, round(width * (resize_height / max(height, 1))))
    if rows is None and cols is None:
        cols = min(len(tiles), 10)
        rows = int(math.ceil(len(tiles) / cols))
    elif cols is None:
        cols = int(math.ceil(len(tiles) / max(rows or 1, 1)))
    elif rows is None:
        rows = int(math.ceil(len(tiles) / cols))

    canvas = Image.new(
        "RGB", (cols * resize_width, rows * resize_height), (128, 128, 128)
    )
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        width, height = tile.size
        scale = min(resize_width / max(width, 1), resize_height / max(height, 1))
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = tile.resize(new_size)
        x = col * resize_width + (resize_width - resized.width) // 2
        y = row * resize_height + (resize_height - resized.height) // 2
        canvas.paste(resized, (x, y))

    used_rows = int(math.ceil(len(tiles) / cols))
    canvas = canvas.crop((0, 0, cols * resize_width, used_rows * resize_height))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def _predict_overlay_probs(unet, image_paths: list[str], image_size, batch_size: int):
    """Run batched model inference and return one probability tensor per image."""
    width, height = image_size
    probs_batches = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_images = []
        for path in batch_paths:
            image = Image.open(path).convert("RGB").resize((width, height))
            batch_images.append(np.asarray(image, dtype=np.float32) / 255.0)
        batch = np.stack(batch_images, axis=0)
        result = unet(batch)
        if isinstance(result, dict):
            probs = np.asarray(result["probs"])
        else:
            probs = np.asarray(result[0])
        probs_batches.extend(probs)
    return probs_batches


def _safe_mask_name(sample_id: str) -> str:
    """Create a bounded filesystem-safe stem for generated mask filenames."""
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in str(sample_id))
    sanitized = sanitized.strip("_")
    return sanitized[:64] or "sample"
