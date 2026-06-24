"""Regression tests for the tf.data UNet batch loader."""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image

from active_learning.providers.inference import iter_image_batches


def test_preserves_order():
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for index, value in enumerate((32, 160, 224)):
            arr = np.full((8, 8, 3), value, dtype=np.uint8)
            path = os.path.join(d, f"img_{index}.png")
            Image.fromarray(arr).save(path)
            paths.append(path)

        batches = list(iter_image_batches(paths, batch_size=2, image_size=(4, 4)))
        assert len(batches) == 2
        assert tuple(batches[0].shape) == (2, 4, 4, 3)
        assert tuple(batches[1].shape) == (1, 4, 4, 3)
        first_vals = [float(batch[0, 0, 0, 0]) for batch in batches]
        second_vals = [float(batch[-1, 0, 0, 0]) for batch in batches]
        assert round(first_vals[0], 5) == round(32 / 255.0, 5)
        assert round(second_vals[0], 5) == round(160 / 255.0, 5)
        assert round(second_vals[1], 5) == round(224 / 255.0, 5)
