"""TensorFlow inference helpers for active-learning providers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def iter_image_batches(image_paths, batch_size, image_size):
    """Yield tf.data batches of loaded images in deterministic order."""

    def load_one(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (image_size[1], image_size[0]), antialias=True)
        return img

    ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(image_paths))
    ds = ds.map(load_one, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    return ds


def build_infer_fn(model, training: bool = False):
    """Compile a reusable inference callable."""

    @tf.function(reduce_retracing=True)
    def infer(batch):
        return model(batch, training=training)

    return infer


def extract_probs(result):
    """Extract probability output from model inference result."""
    if isinstance(result, dict):
        return result["probs"]
    return result[0]


def collect_mc_probs(infer, batch, n_iterations: int) -> np.ndarray:
    """Collect stochastic predictions for one batch as [T, B, H, W, K]."""
    all_probs = []
    for _ in range(n_iterations):
        result = infer(batch)
        all_probs.append(extract_probs(result).numpy())
    return np.stack(all_probs, axis=0)


def extract_probs_and_penultimate(
    model,
    image_paths: list[str],
    image_size: tuple[int, int] = (320, 240),
    batch_size: int = 8,
    *,
    progress: bool = False,
    on_step: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run model inference and extract probs + penultimate activations."""
    infer = build_infer_fn(model, training=False)
    all_probs = []
    all_penultimate = []

    n_paths = len(image_paths)
    if n_paths == 0:
        return all_probs, all_penultimate

    batches = iter_image_batches(image_paths, batch_size, image_size)
    if progress:
        batches = tqdm(
            batches,
            total=(n_paths + batch_size - 1) // batch_size,
            desc="Model inference",
        )
    done = 0
    for batch in batches:
        probs_batch, pen_batch = infer(batch)
        probs_batch = probs_batch.numpy()
        pen_batch = pen_batch.numpy()

        for j in range(len(probs_batch)):
            all_probs.append(probs_batch[j])
            all_penultimate.append(pen_batch[j])

        done += len(probs_batch)
        if on_step is not None:
            on_step(done, n_paths)

    return all_probs, all_penultimate
