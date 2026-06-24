"""Tests for brightness scoring and filtering."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from active_learning.core.image_provider import ImageProvider
from active_learning.scorers.brightness import filter_by_brightness


class FakeBrightnessSource:
    def __init__(self, tmpdir: str):
        self.tmpdir = Path(tmpdir)
        self.paths = {}
        for name, value in {"dark": 10, "mid": 120, "bright": 250}.items():
            path = self.tmpdir / f"{name}.png"
            Image.new("RGB", (8, 8), color=(value, value, value)).save(path)
            self.paths[name] = str(path)

    def fetch_lowres(self, sample_id: str) -> str:
        return self.paths[sample_id]

    def fetch_highres(self, sample_id: str) -> str:
        return self.paths[sample_id]

    def fetch_metadata(self, sample_id: str, fields=None) -> dict[str, object]:
        return {}


def test_filter_by_brightness_uses_stats_cache():
    with tempfile.TemporaryDirectory() as tmp:
        provider = ImageProvider(
            FakeBrightnessSource(tmp),
            cache_root=Path(tmp) / "provider-cache",
        )
        sample_ids = ["dark", "mid", "bright"]
        filtered, stats = filter_by_brightness(
            sample_ids,
            provider,
            Path(tmp) / "score-cache",
            min_brightness=50,
            max_brightness=200,
            progress=False,
        )
        assert filtered == ["mid"]
        assert set(stats) == set(sample_ids)
