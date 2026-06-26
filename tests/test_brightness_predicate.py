"""Tests for brightness scoring and filtering."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from active_learning.core.image_provider import ImageProvider
from active_learning.scorers.brightness import filter_by_brightness, score_brightness


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


class StatusCodeError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class ErrorSource:
    def __init__(self, status_code: int):
        self.status_code = status_code

    def fetch_lowres(self, sample_id: str) -> str:
        raise StatusCodeError(self.status_code)

    def fetch_highres(self, sample_id: str) -> str:
        raise StatusCodeError(self.status_code)

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


def test_score_brightness_treats_provider_404_as_missing():
    with tempfile.TemporaryDirectory() as tmp:
        provider = ImageProvider(
            ErrorSource(404),
            cache_root=Path(tmp) / "provider-cache",
        )

        stats = score_brightness(
            ["missing"],
            provider,
            Path(tmp) / "score-cache",
            progress=False,
        )

        assert stats["missing"]["_unreadable"] == 1.0
        assert stats["missing"]["brightness_mean"] == -1.0


def test_score_brightness_reraises_provider_non_404():
    with tempfile.TemporaryDirectory() as tmp:
        provider = ImageProvider(
            ErrorSource(500),
            cache_root=Path(tmp) / "provider-cache",
        )

        with pytest.raises(StatusCodeError):
            score_brightness(
                ["error"],
                provider,
                Path(tmp) / "score-cache",
                progress=False,
            )
