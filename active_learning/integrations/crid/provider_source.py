"""CRID-backed ImageProviderSource implementation."""

from __future__ import annotations

from pathlib import Path

from azure.common import AzureHttpError
from PIL import Image

from active_learning.core.image_provider import ImageProviderSource
from interface.service.dataset import DatasetService


def _staging_file_is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
    except Exception:
        return False
    return True


def thumbnail_blob_from_blob(blob: str) -> str:
    return (
        blob.replace("/images/", "/thumbnails/")
        .replace(".png", ".jpg")
        .replace(
            ".ppm",
            ".jpg",
        )
    )


class CridImageProviderSource(ImageProviderSource):
    """Fetches CRID images and retained metadata by blob ID."""

    def __init__(self, crid, download_root: str | Path):
        self.crid = crid
        self.download_root = Path(download_root)
        self.download_root.mkdir(parents=True, exist_ok=True)

    def fetch_lowres(self, sample_id: str) -> str:
        return self._download_blob(
            thumbnail_blob_from_blob(sample_id),
            sample_id=sample_id,
        )

    def fetch_highres(self, sample_id: str) -> str:
        return self._download_blob(sample_id, sample_id=sample_id)

    def fetch_metadata(
        self,
        sample_id: str,
        fields: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        docs = self.crid.elasticsearch_client.find_by_ids([sample_id]).get("docs", [])
        if not docs:
            return {}
        source = docs[0].get("_source", {})
        if fields:
            return {field: source.get(field) for field in fields}
        return dict(source)

    def _download_blob(self, blob: str, *, sample_id: str | None = None) -> str:
        local_path = self.download_root / DatasetService._get_filename(blob)
        if local_path.exists():
            if _staging_file_is_readable_image(local_path):
                return str(local_path)
            try:
                local_path.unlink()
            except OSError:
                pass
        container = self.crid.config.azure.data_container
        destination_file = str(local_path)
        fp = open(destination_file, "ab")
        try:
            self.crid.azure_client.get_blob_to_stream(container, blob, fp)
        except AzureHttpError as e:
            fp.close()
            if local_path.exists():
                local_path.unlink()
            if e.status_code == 404:
                raise FileNotFoundError(
                    f"Blob not found in Azure ({container}): {blob!r}"
                ) from e
            raise
        except Exception:
            fp.close()
            if local_path.exists():
                local_path.unlink()
            raise
        fp.flush()
        fp.close()
        return str(local_path)
