"""Tests for ImageProvider caching and metadata fallback."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from active_learning.core.image_provider import ImageProvider


class FakeSource:
    def __init__(self, tmpdir: str):
        self.tmpdir = Path(tmpdir)
        self.calls = {"lowres": 0, "highres": 0, "metadata": []}
        self.lowres = self.tmpdir / "low.png"
        self.highres = self.tmpdir / "high.png"
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(self.lowres)
        Image.new("RGB", (16, 16), color=(30, 40, 50)).save(self.highres)

    def fetch_lowres(self, sample_id: str) -> str:
        self.calls["lowres"] += 1
        return str(self.lowres)

    def fetch_highres(self, sample_id: str) -> str:
        self.calls["highres"] += 1
        return str(self.highres)

    def fetch_metadata(self, sample_id: str, fields=None) -> dict[str, object]:
        self.calls["metadata"].append(tuple(fields or ()))
        data = {"origin": "miniportal", "camera": "cam_trolley"}
        if fields:
            return {field: data[field] for field in fields}
        return data


def test_image_provider_caches_paths_and_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        source = FakeSource(tmp)
        provider = ImageProvider(
            source,
            cache_root=Path(tmp) / "cache",
            retained_metadata_fields=("origin", "camera"),
        )
        first_low = provider.get_lowres("sample-1")
        second_low = provider.get_lowres("sample-1")
        assert first_low == second_low
        assert source.calls["lowres"] == 1
        first_high = provider.get_highres("sample-1")
        second_high = provider.get_highres("sample-1")
        assert first_high == second_high
        assert source.calls["highres"] == 1
        metadata = provider.get_metadata("sample-1", fields=("origin",))
        assert metadata == {"origin": "miniportal"}
        metadata = provider.get_metadata("sample-1", fields=("origin", "camera"))
        assert metadata == {"origin": "miniportal", "camera": "cam_trolley"}
        assert source.calls["metadata"] == [("origin",), ("camera",)]


def test_get_lowres_batch_rejects_duplicate_ids():
    with tempfile.TemporaryDirectory() as tmp:
        source = FakeSource(tmp)
        provider = ImageProvider(
            source,
            cache_root=Path(tmp) / "cache",
            retained_metadata_fields=("origin", "camera"),
        )
        with pytest.raises(ValueError, match="get_lowres_batch"):
            provider.get_lowres_batch(["a", "b", "a"])


def test_get_highres_batch_rejects_duplicate_ids():
    with tempfile.TemporaryDirectory() as tmp:
        source = FakeSource(tmp)
        provider = ImageProvider(
            source,
            cache_root=Path(tmp) / "cache",
            retained_metadata_fields=("origin", "camera"),
        )
        with pytest.raises(ValueError, match="get_highres_batch"):
            provider.get_highres_batch(["x", "x"])


def test_iter_lowres_rejects_duplicate_ids():
    with tempfile.TemporaryDirectory() as tmp:
        source = FakeSource(tmp)
        provider = ImageProvider(
            source,
            cache_root=Path(tmp) / "cache",
            retained_metadata_fields=("origin", "camera"),
        )
        with pytest.raises(ValueError, match="iter_lowres"):
            list(provider.iter_lowres(["p", "p"]))


class UnreadableFetchSource:
    def __init__(self, path: Path) -> None:
        self.path = path

    def fetch_lowres(self, sample_id: str) -> str:
        return str(self.path)

    def fetch_highres(self, sample_id: str) -> str:
        return str(self.path)

    def fetch_metadata(self, sample_id: str, fields=None) -> dict[str, object]:
        return {}


def test_get_lowres_propagates_pil_error_for_unreadable_source_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        garbage = tmp_path / "not-an-image.bin"
        garbage.write_bytes(b"\xff\x00\x01not png")
        src = UnreadableFetchSource(garbage)
        provider = ImageProvider(src, cache_root=tmp_path / "cache")
        with pytest.raises(UnidentifiedImageError) as excinfo:
            provider.get_lowres("sample-z")
        assert "blob_id=" in str(excinfo.value) and "sample-z" in str(excinfo.value)
