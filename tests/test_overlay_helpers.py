"""Tests for shared overlay mosaic helpers."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from active_learning.sinks.mosaic import save_prediction_mask
from active_learning.sinks.mosaic import render_overlay_mosaic
from active_learning.sinks.mosaic import overlay_mask
from active_learning.sinks.mosaic import overlay_mosaic_paths
from active_learning.core.types import SelectionResult


class TestOverlayHelpers(unittest.TestCase):
    def test_overlay_mosaic_paths_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            images_dir = os.path.join(tmp, "images")
            masks_dir = os.path.join(tmp, "masks")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            image_paths = []
            mask_paths = []
            for i in range(3):
                image_path = os.path.join(images_dir, f"img_{i}.png")
                mask_path = os.path.join(masks_dir, f"img_{i}.png")
                img = np.full((32, 32, 3), 40 + i * 40, dtype=np.uint8)
                cv2.imwrite(image_path, img)
                probs = np.zeros((32, 32, 3), dtype=np.float32)
                probs[..., i % 3] = 1.0
                save_prediction_mask(mask_path, probs)
                image_paths.append(image_path)
                mask_paths.append(mask_path)

            output = os.path.join(tmp, "overlay.jpg")
            overlay_mosaic_paths(
                image_paths,
                mask_paths,
                output,
                resize_height=32,
                rows=1,
                cols=3,
            )

            self.assertTrue(os.path.exists(output))
            out = cv2.imread(output)
            self.assertIsNotNone(out)
            self.assertGreater(out.shape[0], 0)
            self.assertGreater(out.shape[1], 0)

    def test_save_prediction_mask_writes_colored_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mask.png")
            probs = np.zeros((16, 16, 4), dtype=np.float32)
            probs[..., 2] = 1.0
            save_prediction_mask(path, probs)

            self.assertTrue(os.path.exists(path))
            img = np.array(Image.open(path))
            self.assertEqual(img.shape[:2], (16, 16))
            self.assertTrue(np.any(img != 255))

    def test_save_prediction_mask_uses_fully_transparent_black_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mask.png")
            probs = np.zeros((8, 8, 3), dtype=np.float32)
            probs[..., 0] = 1.0
            save_prediction_mask(path, probs)

            img = np.array(Image.open(path))
            self.assertTrue(np.all(img[..., 3] == 0))
            self.assertTrue(np.all(img[..., :3] == 0))

    def test_overlay_mask_ignores_transparent_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "image.png")
            mask_path = os.path.join(tmp, "mask.png")

            img = np.full((16, 16, 3), 80, dtype=np.uint8)
            cv2.imwrite(image_path, img)

            probs = np.zeros((16, 16, 3), dtype=np.float32)
            probs[:8, :8, 1] = 1.0
            probs[8:, 8:, 0] = 1.0
            save_prediction_mask(mask_path, probs)

            overlay = overlay_mask(image_path, mask_path, alpha=0.5)

            self.assertEqual(overlay.shape, (16, 16, 3))
            self.assertTrue(np.all(overlay[12, 12] == 80))
            self.assertTrue(np.any(overlay[:8, :8] != 80))

    def test_generate_overlay_mosaic_writes_overlay_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "img_a.png")
            Image.new("RGB", (32, 32), color=(80, 80, 80)).save(image_path)

            class FakeProvider:
                def __init__(self):
                    self.highres_calls = 0
                    self.lowres_calls = 0

                def get_lowres_batch(
                    self, sample_ids, *, progress=False, allow_missing=False
                ):
                    self.lowres_calls += 1
                    return [image_path for _ in sample_ids]

                def get_highres_batch(
                    self, sample_ids, *, progress=False, allow_missing=False
                ):
                    self.highres_calls += 1
                    return [image_path for _ in sample_ids]

            class FakeUnet:
                def __call__(self, batch):
                    probs = np.zeros((len(batch), 16, 16, 3), dtype=np.float32)
                    probs[..., 1] = 1.0
                    return {"probs": probs}

            cfg = SimpleNamespace(
                query=SimpleNamespace(
                    use_full_res_images=False,
                    start="2026-04-01",
                    end="2026-04-02",
                ),
                export=SimpleNamespace(
                    project="proj",
                    prefix="pref",
                    mosaic_path=os.path.join(tmp, "mosaic.jpg"),
                ),
                selection=SimpleNamespace(
                    n_select=1,
                    strategy="coreset",
                ),
                alges=SimpleNamespace(
                    model="overlay_unet",
                    batch_size=2,
                ),
                models={
                    "overlay_unet": SimpleNamespace(
                        path="/tmp/model.zip",
                        image_size=(16, 16),
                    ),
                },
            )

            provider = FakeProvider()
            with patch(
                "active_learning.sinks.mosaic.load_unet",
                return_value=FakeUnet(),
            ):
                output_path = render_overlay_mosaic(
                    SelectionResult(selected_ids=["sample-a"]),
                    provider,
                    cfg=cfg,
                )

            self.assertTrue(os.path.exists(output_path))
            out = cv2.imread(output_path)
            self.assertIsNotNone(out)
            self.assertGreater(out.shape[0], 0)
            self.assertGreater(out.shape[1], 0)
            self.assertEqual(provider.highres_calls, 1)
            self.assertEqual(provider.lowres_calls, 0)


if __name__ == "__main__":
    unittest.main()
