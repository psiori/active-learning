from __future__ import annotations

from pathlib import Path

from PIL import Image

from active_learning.integrations.local import (
    LocalImageProviderSource,
    discover_local_images,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(path)


def test_discover_local_images_recurses_and_sorts_relative_ids(tmp_path):
    _write_image(tmp_path / "z.JPG")
    _write_image(tmp_path / "nested" / "a.PNG")
    _write_image(tmp_path / "nested" / "deep" / "b.webp")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    discovered = discover_local_images(tmp_path)

    assert list(discovered) == [
        "nested/a.PNG",
        "nested/deep/b.webp",
        "z.JPG",
    ]


def test_local_image_provider_source_resolves_paths_and_metadata(tmp_path):
    image_path = tmp_path / "one.bmp"
    _write_image(image_path)
    source = LocalImageProviderSource.from_directory(tmp_path)

    assert source.sample_ids == ["one.bmp"]
    assert source.fetch_lowres("one.bmp") == str(image_path.resolve())
    assert source.fetch_highres("one.bmp") == str(image_path.resolve())
    assert source.fetch_metadata("one.bmp") == {}
