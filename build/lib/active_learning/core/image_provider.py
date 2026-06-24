"""Image access for active learning: fetch from a source, cache under *cache_root*.

``ImageProvider`` stores manifests for materialized source paths so downstream code
always resolves a predictable path per sample ID while the source may implement its
own download or staging layout.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from PIL import Image
from tqdm import tqdm

from active_learning.core.types import SampleId


class ImageProviderSource(Protocol):
    """Source that knows how to materialize images and metadata for sample IDs."""

    def fetch_lowres(self, sample_id: SampleId) -> str:
        """Return a local filesystem path for the low-resolution image."""

    def fetch_highres(self, sample_id: SampleId) -> str:
        """Return a local filesystem path for the high-resolution image."""

    def fetch_metadata(
        self,
        sample_id: SampleId,
        fields: Sequence[str] | None = None,
    ) -> dict[str, object]:
        """Return retained metadata fields for *sample_id*."""


def _sample_digest(sample_id: SampleId) -> str:
    """Stable SHA-1 hex digest of *sample_id* (UTF-8) for sharded cache paths."""
    return hashlib.sha1(sample_id.encode("utf-8")).hexdigest()


def _pool_max_workers(n: int) -> int:
    """Worker count for parallel I/O, capped by *n* and a small CPU-based upper bound."""
    if n < 2:
        return 1
    return min(32, n, max(4, (os.cpu_count() or 4) * 4))


def _validate_image(path: Path, sample_id: SampleId) -> None:
    """Check that *path* exists and decodes as an image."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist for sample {sample_id}")
    try:
        with Image.open(path) as img:
            img.load()
    except Exception as e:
        detail = str(e) or repr(e)
        raise type(e)(f"{detail} [blob_id={sample_id!r}]") from e


def _unlink_image_manifest(manifest_path: Path) -> None:
    try:
        if manifest_path.exists():
            manifest_path.unlink()
    except OSError:
        pass


#: Upper bound on concurrent ``get_lowres`` / ``get_highres`` tasks while iterating
#: with ``iter_lowres`` / ``iter_highres`` (sliding window over ``Future``s).
_MATERIALIZE_PREFETCH_AHEAD = 10


def _parallel_map_ordered(
    func: Callable[[SampleId], str],
    ids: list[SampleId],
    *,
    progress: bool,
    desc: str,
    unit: str,
) -> list[str]:
    """Run *func* per id in a thread pool; return paths in the same order as *ids*.

    When *progress* is true, completion order is wrapped with ``tqdm`` (bar advances
    per finished task, not per input index).
    """
    n = len(ids)
    if n == 0:
        return []
    if n == 1:
        return [func(ids[0])]
    max_workers = _pool_max_workers(n)
    out: list[str] = [""] * n
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(func, ids[i]): i for i in range(n)}
        completed = as_completed(future_to_index)
        if progress:
            completed = tqdm(
                completed,
                total=n,
                desc=desc,
                unit=unit,
                leave=True,
            )
        for fut in completed:
            idx = future_to_index[fut]
            out[idx] = fut.result()
    return out


def _raise_if_duplicate_sample_ids(ids: list[SampleId], *, context: str) -> None:
    counts = Counter(ids)
    dupes = [sid for sid, n in counts.items() if n > 1]
    if not dupes:
        return
    dupes.sort(key=str)
    parts = [f"{sid!r} (×{counts[sid]})" for sid in dupes[:30]]
    tail = f", … (+{len(dupes) - 30} more)" if len(dupes) > 30 else ""
    raise ValueError(
        f"{context}: duplicate sample_ids in one sequence ({len(ids)} entries, "
        f"{len(counts)} unique). Duplicates: {', '.join(parts)}{tail}",
    )


class ImageProvider:
    """Materializes low/high-res image paths and retained metadata on demand.

    Low/high images are resolved through a JSON manifest per sample under
    ``{cache_root}/provider/{lowres|highres}/`` so repeated lookups skip re-fetching.
    The manifest records the source-owned local path returned by the provider source.
    Batches use a thread pool; iterators prefetch only up to
    ``_MATERIALIZE_PREFETCH_AHEAD`` tasks ahead of the next yield.
    Duplicate *sample_id* values in a single batch or iterator sequence are rejected
    with ``ValueError`` so upstream pool/query bugs are not masked.
    """

    def __init__(
        self,
        source: ImageProviderSource,
        cache_root: str | Path,
        retained_metadata_fields: Sequence[str] | None = None,
        ignore_missing_blobs: bool = False,
    ):
        """Configure *source*, on-disk *cache_root*, and optional metadata field hints."""
        self.source = source
        self.cache_root = Path(cache_root)
        self.provider_root = self.cache_root / "provider"
        self.lowres_root = self.provider_root / "lowres"
        self.highres_root = self.provider_root / "highres"
        self.metadata_root = self.provider_root / "metadata"
        self.retained_metadata_fields = tuple(retained_metadata_fields or ())
        self.ignore_missing_blobs = ignore_missing_blobs

        self.lowres_root.mkdir(parents=True, exist_ok=True)
        self.highres_root.mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(parents=True, exist_ok=True)

    def get_lowres(self, sample_id: SampleId) -> str | None:
        """Return a local path to the cached low-resolution image for *sample_id*."""
        try:
            return self._materialize_image(
                sample_id=sample_id,
                root=self.lowres_root,
                fetcher=self.source.fetch_lowres,
            )
        except FileNotFoundError:
            if self.ignore_missing_blobs:
                return None
            raise

    def get_highres(self, sample_id: SampleId) -> str | None:
        """Return a local path to the cached full-resolution image for *sample_id*."""
        try:
            return self._materialize_image(
                sample_id=sample_id,
                root=self.highres_root,
                fetcher=self.source.fetch_highres,
            )
        except FileNotFoundError:
            if self.ignore_missing_blobs:
                return None
            raise

    def get_lowres_batch(
        self,
        sample_ids: Sequence[SampleId],
        *,
        progress: bool = False,
        allow_missing: bool = True,
    ) -> list[str] | list[str | None]:
        """Materialize many low-res paths in parallel; order matches *sample_ids*.

        Args:
            progress: If true, show a tqdm bar over completed downloads.
            allow_missing: If true, return ``None`` for blobs that are missing
                instead of raising.

        Raises:
            ValueError: If *sample_ids* contains the same id more than once.
        """
        ids = list(sample_ids)
        if not ids:
            return []
        _raise_if_duplicate_sample_ids(ids, context="get_lowres_batch")
        fetcher: Callable[[SampleId], str | None] = self.get_lowres
        if allow_missing:

            def fetcher(sid: SampleId) -> str | None:
                try:
                    return self.get_lowres(sid)
                except FileNotFoundError:
                    return None

        return _parallel_map_ordered(
            fetcher,
            ids,
            progress=progress,
            desc="Low-res images",
            unit="img",
        )

    def get_highres_batch(
        self,
        sample_ids: Sequence[SampleId],
        *,
        progress: bool = False,
        allow_missing: bool = True,
    ) -> list[str] | list[str | None]:
        """Materialize many high-res paths in parallel; order matches *sample_ids*.

        Args:
            progress: If true, show a tqdm bar over completed downloads.
            allow_missing: If true, return ``None`` for blobs that are missing
                instead of raising.

        Raises:
            ValueError: If *sample_ids* contains the same id more than once.
        """
        ids = list(sample_ids)
        if not ids:
            return []
        _raise_if_duplicate_sample_ids(ids, context="get_highres_batch")
        fetcher: Callable[[SampleId], str | None] = self.get_highres
        if allow_missing:

            def fetcher(sid: SampleId) -> str | None:
                try:
                    return self.get_highres(sid)
                except FileNotFoundError:
                    return None

        return _parallel_map_ordered(
            fetcher,
            ids,
            progress=progress,
            desc="High-res images",
            unit="img",
        )

    def _iter_prefetched(
        self,
        sample_ids: Iterable[SampleId],
        materialize: Callable[[SampleId], str],
        *,
        context: str,
    ) -> Iterator[tuple[SampleId, str]]:
        """Yield ``(sample_id, path)`` in input order with bounded concurrent *materialize*."""
        ids = list(sample_ids)
        _raise_if_duplicate_sample_ids(ids, context=context)
        n = len(ids)
        if n == 0:
            return
        if n == 1:
            sid = ids[0]
            yield sid, materialize(sid)
            return
        max_ahead = _MATERIALIZE_PREFETCH_AHEAD
        max_workers = min(_pool_max_workers(n), max_ahead)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[int, Future[str]] = {}
            next_submit = 0
            for i in range(n):
                while len(futures) < max_ahead and next_submit < n:
                    futures[next_submit] = executor.submit(
                        materialize,
                        ids[next_submit],
                    )
                    next_submit += 1
                fut = futures.pop(i)
                yield ids[i], fut.result()

    def iter_lowres(
        self,
        sample_ids: Iterable[SampleId],
    ) -> Iterator[tuple[SampleId, str]]:
        """Yield ``(id, cached low-res path)`` in *sample_ids* order with prefetch."""
        yield from self._iter_prefetched(
            sample_ids,
            self.get_lowres,
            context="iter_lowres",
        )

    def iter_highres(
        self,
        sample_ids: Iterable[SampleId],
    ) -> Iterator[tuple[SampleId, str]]:
        """Yield ``(id, cached high-res path)`` in *sample_ids* order with prefetch."""
        yield from self._iter_prefetched(
            sample_ids,
            self.get_highres,
            context="iter_highres",
        )

    def get_metadata(
        self,
        sample_id: SampleId,
        fields: Sequence[str] | None = None,
    ) -> dict[str, object]:
        """Load or merge JSON metadata for *sample_id* under the provider cache.

        If *fields* is omitted, uses ``retained_metadata_fields`` from construction,
        or fetches/merges the full record when that set is empty and nothing is cached.
        """
        requested_fields = tuple(fields or self.retained_metadata_fields)
        cache_path = self._metadata_cache_path(sample_id)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            cached = {}

        missing_fields = (
            [field for field in requested_fields if field not in cached]
            if requested_fields
            else []
        )
        if missing_fields or (not requested_fields and not cached):
            fetched = self.source.fetch_metadata(
                sample_id,
                missing_fields or requested_fields or None,
            )
            cached.update(fetched or {})
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cached, sort_keys=True, ensure_ascii=True),
                encoding="utf-8",
            )

        if requested_fields:
            return {field: cached.get(field) for field in requested_fields}
        return dict(cached)

    def _materialize_image(self, sample_id, root: Path, fetcher) -> str:
        """Ensure *sample_id* is available under *root* and return the cached file path.

        Uses a manifest JSON next to the image file; if the manifest references a
        valid local image, returns immediately. Otherwise calls *fetcher*
        (typically blocking I/O), validates the returned local path, writes the
        manifest, and returns.
        """
        manifest_path = self._image_manifest_path(root, sample_id)
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
            else:
                # Entered only when manifest JSON parsing succeeds without raising.
                if path_str := manifest.get("path"):
                    source_path = Path(path_str)
                    try:
                        _validate_image(source_path, sample_id)
                        return str(source_path)
                    except Exception:
                        pass
            # Something failed about reading the image or manifest, so
            # delete the manifest so we rewrite it cleanly.
            _unlink_image_manifest(manifest_path)

        # Redownload and check new image.
        source_path = Path(fetcher(sample_id))
        _validate_image(source_path, sample_id)

        # Write new manifest for the newly downloaded image.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {"path": str(source_path)},
                sort_keys=True,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        return str(source_path)

    def _image_manifest_path(self, root: Path, sample_id: SampleId) -> Path:
        """Path to the JSON manifest that records the on-disk image path for *sample_id*."""
        digest = _sample_digest(sample_id)
        return root / digest[:2] / f"{digest}.json"

    def _metadata_cache_path(self, sample_id: SampleId) -> Path:
        """Path to the cached JSON blob of merged metadata for *sample_id*."""
        digest = _sample_digest(sample_id)
        return self.metadata_root / digest[:2] / f"{digest}.json"
