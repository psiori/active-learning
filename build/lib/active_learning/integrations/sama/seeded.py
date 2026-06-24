"""Query previously seeded blob IDs from Sama batches."""

from __future__ import annotations

from collections.abc import Callable
import io
import json
import logging
from pathlib import Path
import time

import pandas as pd

from interface.service.sama import Sama

logger = logging.getLogger(__name__)


SKIPPED_BATCH_STATES = {"import_cancelled", "cancelling_import", "failed"}

# Maps str(batch_id) -> export_id string, or None if the batch had no usable tasks.
# Only written on successful 2xx responses; HTTP errors are not cached so they retry.
_CACHE_PATH = Path.home() / ".cache" / "crid_al" / "sama_batch_export_ids.json"


def _load_cache() -> dict[str, str | None]:
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict[str, str | None]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except Exception as exc:
        logger.warning("Could not save Sama batch cache: %s", exc)


def _resolve_export_id(sama: Sama, batch_id: int) -> str | None:
    """Fetch one task from a batch to extract the CRID export ID (client_batch_id)."""
    sleep = 1.2
    for attempt in range(5):
        res = sama.get(
            f"{sama.url}/tasks.json",
            params={
                "batch_id": batch_id,
                "page": 1,
                "page_size": 1,
                "omit_answers": "true",
            },
        )
        time.sleep(sleep)
        if res.status_code != 429:
            break
        sleep *= 2
        if attempt < 4:
            logger.warning(
                "Sama rate limited on batch %s (attempt %d/5), retrying after %.1fs",
                batch_id,
                attempt + 1,
                sleep,
            )
            time.sleep(sleep)
        else:
            logger.error(
                "Sama rate limited on batch %s after 5 attempts, skipping", batch_id
            )

    if not res.ok:
        if res.status_code != 429:
            logger.warning(
                "Sama returned %s for batch %s tasks, skipping",
                res.status_code,
                batch_id,
            )
        return None  # signal: do not cache, retry next time

    tasks = res.json().get("tasks", [])
    if not tasks:
        logger.warning("No tasks found for batch %s, skipping", batch_id)
        return ""  # signal: cache as "no export ID"

    export_id = tasks[0].get("data", {}).get("client_batch_id") or ""
    if not export_id:
        logger.warning("No client_batch_id in batch %s tasks, skipping", batch_id)
    return export_id


def query_seeded_ids(
    crid,
    sama_project_id: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> set[str]:
    """Return blob IDs that have been seeded to Sama for a given project.

    Walks all non-cancelled/failed Sama batches, resolves each batch's CRID
    export ID via a single task lookup (cached to disk), then reads the
    corresponding mapping CSV from Azure blob storage.

    Args:
        crid: CRID client instance (provides azure_client and config).
        sama_project_id: Sama project ID to query batches from.
        on_progress: Optional callback(completed, total) fired after each batch.
    """
    sama = Sama(project_id=sama_project_id)
    seeded_ids: set[str] = set()
    batches = sama.get_batches()
    cache = _load_cache()
    cache_dirty = False
    total = len(batches)

    for i, batch in enumerate(batches):
        batch_id = batch.get("id")
        if not batch_id:
            continue
        if str(batch.get("status", "")).lower() in SKIPPED_BATCH_STATES:
            continue

        batch_key = str(batch_id)
        if batch_key in cache:
            export_id = cache[batch_key]
        else:
            try:
                export_id = _resolve_export_id(sama, batch_id)
            except Exception as exc:
                logger.warning("Could not fetch tasks for batch %s: %s", batch_id, exc)
                continue  # do not cache exceptions — retry next time
            if export_id is None:
                continue  # HTTP error — do not cache, retry next time
            cache[batch_key] = export_id
            cache_dirty = True

        if not export_id:
            continue

        mapping_blob = f"{export_id}/{export_id}_mapping.csv"
        try:
            mapping_bytes = crid.azure_client.get_blob_to_bytes(
                crid.config.azure.exports_container,
                mapping_blob,
            )
        except Exception as exc:
            logger.warning(
                "Could not read batch mapping %s/%s: %s",
                crid.config.azure.exports_container,
                mapping_blob,
                exc,
            )
            continue

        mapping_df = pd.read_csv(
            io.BytesIO(mapping_bytes.content),
            names=["anon_URL", "anon", "blob"],
        )
        seeded_ids.update(mapping_df["blob"].tolist())

        if on_progress:
            on_progress(i + 1, total)

    if on_progress and total == 0:
        on_progress(1, 1)

    if cache_dirty:
        _save_cache(cache)

    return seeded_ids
