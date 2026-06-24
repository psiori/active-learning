"""Local filesystem-backed ImageProviderSource implementation."""

from __future__ import annotations

from pathlib import Path

from active_learning.core.image_provider import ImageProviderSource
from active_learning.core.types import SampleId


SUPPORTED_LOCAL_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
)


def discover_local_images(images_dir: str | Path) -> dict[SampleId, Path]:
    """Return stable relative-path sample IDs mapped to local image paths."""
    root = Path(images_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    paths = [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_LOCAL_IMAGE_EXTENSIONS
    ]
    mapping = {
        path.relative_to(root).as_posix(): path
        for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix())
    }
    if not mapping:
        suffixes = ", ".join(sorted(SUPPORTED_LOCAL_IMAGE_EXTENSIONS))
        raise ValueError(f"No supported images found in {root} ({suffixes})")
    return mapping


class LocalImageProviderSource(ImageProviderSource):
    """Resolve local relative-path sample IDs to image files on disk."""

    def __init__(self, sample_paths: dict[SampleId, str | Path]) -> None:
        self.sample_paths = {
            sample_id: Path(path).expanduser().resolve()
            for sample_id, path in sample_paths.items()
        }

    @classmethod
    def from_directory(cls, images_dir: str | Path) -> "LocalImageProviderSource":
        return cls(discover_local_images(images_dir))

    @property
    def sample_ids(self) -> list[SampleId]:
        return sorted(self.sample_paths)

    def fetch_lowres(self, sample_id: SampleId) -> str:
        return str(self._path_for(sample_id))

    def fetch_highres(self, sample_id: SampleId) -> str:
        return str(self._path_for(sample_id))

    def fetch_metadata(
        self,
        sample_id: SampleId,
        fields=None,
    ) -> dict[str, object]:
        return {}

    def _path_for(self, sample_id: SampleId) -> Path:
        try:
            return self.sample_paths[sample_id]
        except KeyError as exc:
            raise FileNotFoundError(f"No local image for sample ID {sample_id!r}") from exc
