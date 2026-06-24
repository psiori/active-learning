from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from active_learning.api.jobs import job_manager
from active_learning.api.progress import ProgressReporter
from interface.client.sama.config import SamaConfig
from interface.crid import CRID
from interface.service.sama import PROJECTS, Sama, TaskStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sama", tags=["sama"])


DEFAULT_TASK_STATUS = [
    TaskStatus.ACKNOWLEDGED,
    TaskStatus.APPROVED,
    TaskStatus.DELIVERED,
]

ALL_STYLES = ("labels", "noholes", "scaled", "noholes_scaled")

CAMERA_BY_PROJECT_TYPE = {
    "trolley": "trolley",
    "sideview": "sideview",
    "cabin": "pbb",
}


# Extra consumers per Sama project id. Some Sama projects feed multiple CRID
# origins (e.g. cabin labels consumed by both aespjetway and pbbfraport).
# Entries are appended to whatever PROJECTS already declares.
EXTRA_SAMA_CONSUMERS: dict[int, list[dict[str, Any]]] = {
    57917: [
        {
            "origin": "pbbfraport",
            "project_type": "cabin",
            "environment": "production",
            "camera": "pbb",
        }
    ],
    57916: [
        {
            "origin": "pbbfraport",
            "project_type": "cabin",
            "environment": "training",
            "camera": "pbb",
        }
    ],
}


_crid_cache: dict[str, CRID] = {}
_crid_lock = threading.Lock()
_pipeline_lock = threading.Lock()


def dataset_cache_root() -> Path:
    return Path(os.environ.get("DATASET_CACHE_DIR", "/tmp/dataset_cache"))


def get_crid(origin: str) -> CRID:
    """Return a CRID instance for the given ORIGIN, instantiating once per ORIGIN.

    CRID reads env at construction time; swap the env var, instantiate, restore.
    Cached so subsequent calls reuse the same client pools.
    """
    with _crid_lock:
        if origin in _crid_cache:
            return _crid_cache[origin]
        previous = os.environ.get("ORIGIN")
        os.environ["ORIGIN"] = origin
        try:
            crid = CRID({})
        finally:
            if previous is None:
                os.environ.pop("ORIGIN", None)
            else:
                os.environ["ORIGIN"] = previous
        _crid_cache[origin] = crid
        return crid


def list_projects() -> list[dict[str, Any]]:
    """Flatten PROJECTS + EXTRA_SAMA_CONSUMERS into a list of
    {origin, project_type, env, project_id, camera}.

    Each (origin, project_id) pair appears once.
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for origin, type_map in PROJECTS.items():
        for project_type, env_map in type_map.items():
            camera = CAMERA_BY_PROJECT_TYPE.get(project_type)
            if camera is None:
                continue  # skip non-segmentation projects (bounding box etc.)
            if not isinstance(env_map, dict):
                continue
            for env_name, project_id in env_map.items():
                pid = int(project_id)
                key = (origin, pid)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "origin": origin,
                        "project_type": project_type,
                        "environment": env_name,
                        "project_id": pid,
                        "camera": camera,
                    }
                )
    for project_id, extras in EXTRA_SAMA_CONSUMERS.items():
        for extra in extras:
            key = (extra["origin"], int(project_id))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "origin": extra["origin"],
                    "project_type": extra["project_type"],
                    "environment": extra["environment"],
                    "project_id": int(project_id),
                    "camera": extra["camera"],
                }
            )
    return out


def consumers_for(project_id: int) -> list[dict[str, Any]]:
    """All consumer entries (origin/camera/...) for a given Sama project id."""
    project_id = int(project_id)
    out = [
        {
            "origin": entry["origin"],
            "project_type": entry["project_type"],
            "environment": entry["environment"],
            "camera": entry["camera"],
        }
        for entry in list_projects()
        if entry["project_id"] == project_id
    ]
    return out


def _sama_client(project_id: int) -> Sama:
    api_key = SamaConfig().load_from_env().api_key
    if not api_key:
        raise RuntimeError("SAMA_API_KEY is not configured.")
    return Sama(project_id=project_id, api_key=api_key)


def _imported_batch_ids(crid: CRID, project_id: int) -> set[str]:
    container = crid.config.azure.labels_container
    blob_name = f"imported_batched_{project_id}.csv"
    if not crid.azure_client.blob_exists(container, blob_name):
        return set()
    file_bytes = crid.azure_client.get_blob_to_bytes(container, blob_name)
    content = file_bytes.content.decode("utf-8")
    return {
        line.strip()
        for line in content.splitlines()
        if line.strip()
    }


def _topic_for_batch(crid: CRID, sama: Sama, batch_id: int) -> str | None:
    """Best-effort lookup of the dataset_topic that a batch produced.

    Reads one task to get the client_batch_id (export_id), then resolves
    the corresponding dataset name via export_service.
    """
    try:
        tasks = sama.get_batch_results(batch_id, TaskStatus.ANY, omit_answers=True)
    except Exception:  # noqa: BLE001
        return None
    if not tasks:
        return None
    export_id = tasks[0].get("data", {}).get("client_batch_id")
    if not export_id:
        return None
    try:
        export = crid.export_service.load_export(export_id)
    except Exception:  # noqa: BLE001
        return None
    if export is None:
        return None
    return export.dataset_name


def _masks_exist(dataset_topic: str) -> bool:
    cache_dir = dataset_cache_root() / dataset_topic
    if not (cache_dir / "data.json.gz").exists():
        return False
    for style in ALL_STYLES:
        label_dir = _label_dir(cache_dir, style)
        if label_dir.exists() and any(label_dir.iterdir()):
            return True
    pbb_dir = cache_dir / "door_only"
    if pbb_dir.exists() and any(pbb_dir.iterdir()):
        return True
    return False


def _approved(dataset_topic: str) -> bool:
    return (dataset_cache_root() / dataset_topic / "APPROVED").exists()


def _label_dir(cache_dir: Path, style: str) -> Path:
    if style == "labels":
        return cache_dir / "labels"
    if style == "noholes":
        return cache_dir / "labels_noholes"
    if style == "scaled":
        return cache_dir / "labels"
    if style == "noholes_scaled":
        return cache_dir / "labels_noholes"
    raise ValueError(f"Unsupported style: {style}")


# ---------------------------------------------------------------------------
# Reusable pipeline functions (importable from scripts and the API router).
# ---------------------------------------------------------------------------

def import_batches_for_project(
    origin: str,
    project_id: int,
    batch_ids: list[int] | None,
    dry_run: bool,
    import_in_progress: bool,
    skip_confirm: bool = True,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Import one or more Sama batches for a single project."""
    statuses = list(DEFAULT_TASK_STATUS)
    if import_in_progress:
        statuses.append(TaskStatus.IN_PROGRESS)

    crid = get_crid(origin)
    sama = next(
        (s for s in crid.sama_client.get_samas() if int(s.project_id) == project_id),
        None,
    )
    if sama is None:
        sama = _sama_client(project_id)

    if batch_ids is None:
        if dry_run:
            batches = sama.get_batches()
            return {
                "dry_run": True,
                "batches_inspected": len(batches),
                "dataset_topics": [],
            }
        topics = crid.sama_service.import_missing_batches(
            sama, task_status=statuses, overwrite=True, skip_confirm=skip_confirm
        )
        return {"dataset_topics": list(topics or [])}

    all_topics: list[str] = []
    for batch_id in batch_ids:
        if reporter is not None:
            reporter.status("import", f"Importing batch {batch_id}")
        if dry_run:
            count = 0
            for status in statuses:
                count += len(sama.get_batch_results(batch_id, status))
            all_topics.append(f"[dry-run] batch {batch_id}: {count} tasks")
            continue
        topics = crid.sama_service.import_batch(
            sama,
            batch_id,
            statuses,
            overwrite=True,
            skip_confirm=skip_confirm,
        )
        if topics:
            all_topics.extend(topics)
    return {"dataset_topics": all_topics, "dry_run": dry_run}


def generate_masks_for_topics(
    origin: str,
    dataset_topics: list[str],
    camera: str,
    redo: bool,
    styles: list[str] | None = None,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Generate segmentation masks + mosaic for the given dataset topics."""
    if not dataset_topics:
        return {"dataset_name": None, "skipped": True}

    keep_styles = set(styles) if styles else set(ALL_STYLES)
    include_noholes = any(s in keep_styles for s in {"noholes", "noholes_scaled"})

    stable_name = (
        dataset_topics[0]
        if len(dataset_topics) == 1
        else "merged_"
        + hashlib.sha1(",".join(dataset_topics).encode("utf-8")).hexdigest()[:12]
    )

    cache_dir = dataset_cache_root() / stable_name
    if not redo and _outputs_exist(cache_dir, keep_styles, camera=camera):
        return {
            "dataset_name": stable_name,
            "cache_dir": str(cache_dir),
            "skipped_existing": True,
        }

    crid = get_crid(origin)
    if reporter is not None:
        reporter.status(
            "masks", f"Loading and merging {len(dataset_topics)} dataset topic(s)"
        )
    crid_dataset = crid.load_and_merge_datasets(dataset_topics)
    crid_dataset.name = stable_name
    cache_dir = Path(crid.dataset_service.temp_dataset_cache) / crid_dataset.name
    images_folder = "images"

    if reporter is not None:
        reporter.status("masks", f"Generating {camera} masks")
    if camera == "trolley":
        crid.generate_trolley_masks(
            crid_dataset,
            include_noholes=include_noholes,
            overwrite=redo,
            images_folder=images_folder,
        )
    elif camera == "ska_trolley":
        crid.generate_ska_trolley_masks(crid_dataset, images_folder=images_folder)
    elif camera == "pbb":
        crid.generate_pbb_masks(
            crid_dataset, images_folder=images_folder, overwrite=redo
        )
    else:
        crid.generate_sideview_masks(
            crid_dataset,
            include_noholes=include_noholes,
            overwrite=redo,
            images_folder=images_folder,
        )

    source_size = _source_image_size(cache_dir / images_folder)
    resize_size = _choose_resize_size(source_size)
    if reporter is not None:
        reporter.status("masks", f"Resizing dataset to {resize_size}")
    crid.dataset_service.resize_dataset(crid_dataset, resize_size)

    mosaic_path = cache_dir / f"mosaic_{crid_dataset.name}_images.jpg"
    crid.create_image_mosaic(crid_dataset, str(mosaic_path), resize_height=220)

    _export_raw_dataset(cache_dir, crid_dataset.dataframe)
    _prune_unselected_dirs(cache_dir, keep_styles)

    return {
        "dataset_name": stable_name,
        "cache_dir": str(cache_dir),
        "mosaic_path": str(mosaic_path),
        "resize_size": list(resize_size),
        "dataset_topics": dataset_topics,
    }


def run_pipeline(
    origin: str,
    project_id: int,
    batch_ids: list[int] | None,
    dry_run: bool,
    import_in_progress: bool,
    redo_masks: bool,
    camera: str,
    skip_confirm: bool = True,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Sequential: import → masks. No publish step."""
    if reporter is not None:
        reporter.status("import", f"Importing batches for {origin}/{project_id}")
    import_result = import_batches_for_project(
        origin=origin,
        project_id=project_id,
        batch_ids=batch_ids,
        dry_run=dry_run,
        import_in_progress=import_in_progress,
        skip_confirm=skip_confirm,
        reporter=reporter,
    )

    topics = [t for t in import_result.get("dataset_topics", []) if not t.startswith("[dry-run]")]
    if dry_run or not topics:
        return {"import": import_result, "masks": None}

    mask_result = generate_masks_for_topics(
        origin=origin,
        dataset_topics=topics,
        camera=camera,
        redo=redo_masks,
        reporter=reporter,
    )
    return {"import": import_result, "masks": mask_result}


# ---------------------------------------------------------------------------
# Helpers shared with generate_masks logic.
# ---------------------------------------------------------------------------

def _outputs_exist(cache_dir: Path, keep_styles: set[str], camera: str = "") -> bool:
    if not (cache_dir / "data.json.gz").exists():
        return False
    if camera == "pbb":
        d = cache_dir / "door_only"
        return d.exists() and any(d.iterdir())
    for style in keep_styles:
        d = _label_dir(cache_dir, style)
        if not d.exists() or not any(d.iterdir()):
            return False
    return True


def _prune_unselected_dirs(cache_dir: Path, keep_styles: set[str]):
    for style in ALL_STYLES:
        if style in keep_styles:
            continue
        d = _label_dir(cache_dir, style)
        if d.exists():
            for child in d.iterdir():
                if child.is_file():
                    child.unlink()
            d.rmdir()


def _export_raw_dataset(cache_dir: Path, dataframe) -> None:
    raw_data_path = cache_dir / "data.json.gz"
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    records = dataframe.to_dict(orient="records")
    with gzip.open(raw_data_path, "wt", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, default=str))
            fp.write("\n")


def _source_image_size(images_dir: Path) -> tuple[int, int]:
    paths = sorted(
        [
            *images_dir.glob("*.png"),
            *images_dir.glob("*.jpg"),
            *images_dir.glob("*.jpeg"),
        ]
    )
    if not paths:
        raise RuntimeError(f"No source images found in {images_dir}")
    with Image.open(paths[0]) as image:
        return image.size


def _choose_resize_size(source_size: tuple[int, int]) -> tuple[int, int]:
    sw, sh = source_size
    candidates = [
        (320, 256),  # 5:4
        (320, 240),  # 4:3
        (256, 176),  # 16:11 (pbb cam_cabin_wide, e.g. 1600x1100)
        (240, 320),  # 3:4 portrait (pbb cam_cabin)
    ]
    for tw, th in candidates:
        if sw * th == sh * tw:
            return tw, th
    raise RuntimeError(
        f"Unable to choose a resize target for {sw}x{sh}; "
        f"no matching aspect ratio in {candidates}."
    )


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------

class ProjectEntry(BaseModel):
    origin: str
    project_type: str
    environment: str
    project_id: int
    camera: str


class ConsumerEntry(BaseModel):
    origin: str
    project_type: str
    environment: str
    camera: str


class BatchEntry(BaseModel):
    origin: str
    project_type: str
    environment: str
    project_id: int
    camera: str
    batch_id: int
    state: str | None = None
    updated_at: str | None = None
    total_tasks: int | None = None
    created_tasks: int | None = None
    status_counts: dict[str, int] = Field(default_factory=dict)
    dataset_topic: str | None = None
    imported: bool = False
    masks_exist: bool = False
    approved: bool = False
    consumers: list[ConsumerEntry] = Field(default_factory=list)


class BatchListResponse(BaseModel):
    batches: list[BatchEntry]


class ImportJobRequest(BaseModel):
    origin: str
    project_id: int
    batch_ids: list[int] | None = None
    dry_run: bool = False
    import_in_progress: bool = False
    skip_confirm: bool = True


class MaskJobRequest(BaseModel):
    origin: str
    dataset_topics: list[str]
    camera: str
    redo: bool = False
    styles: list[str] | None = None


class PipelineJobRequest(BaseModel):
    runs: list[ImportJobRequest]
    redo_masks: bool = False
    skip_confirm: bool = True


class ApprovalRequest(BaseModel):
    dataset_topic: str
    approved: bool = True


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@router.get("/projects", response_model=list[ProjectEntry])
def get_projects():
    return list_projects()


_batches_cache: dict[int, tuple[float, list[dict]]] = {}
_BATCHES_TTL_SECONDS = 120.0


def _cached_get_batches(sama: Sama) -> list[dict]:
    import time

    key = int(sama.project_id)
    now = time.monotonic()
    cached = _batches_cache.get(key)
    if cached and (now - cached[0]) < _BATCHES_TTL_SECONDS:
        return cached[1]
    batches = sama.get_batches()
    _batches_cache[key] = (now, batches)
    return batches


@router.get("/batches", response_model=BatchListResponse)
def get_batches(
    origins: str | None = None,
    days: int = 90,
    project_environment: str = "production",
    enrich: bool = True,
    batch_count: int = 5,
    as_origin: str | None = None,
):
    """List recent batches across selected origins.

    Deduped by (project_id, batch_id). Each batch carries the full list
    of consumer origins; the caller picks which one is the import target.
    `enrich` checks imported flag against `as_origin` (or the first
    matching consumer).
    """
    selected_origins = (
        [o.strip() for o in origins.split(",") if o.strip()]
        if origins
        else None
    )
    cutoff = datetime.now(UTC) - timedelta(days=days)

    # Group flat entries by Sama project_id, but only those that have a
    # consumer in the selected origins (or all if selected_origins is None).
    entries_by_project: dict[int, list[dict[str, Any]]] = {}
    for entry in list_projects():
        if entry["environment"] != project_environment:
            continue
        if selected_origins is not None and entry["origin"] not in selected_origins:
            continue
        entries_by_project.setdefault(entry["project_id"], []).append(entry)

    out: list[BatchEntry] = []
    for project_id, entries in entries_by_project.items():
        all_consumers = consumers_for(project_id)
        # Filter consumers to only the visible ones (those in selected_origins).
        visible_consumers = [
            ConsumerEntry(**c)
            for c in all_consumers
            if selected_origins is None or c["origin"] in selected_origins
        ]
        primary = visible_consumers[0] if visible_consumers else None
        if primary is None:
            continue

        # Pick which CRID to use for enrichment.
        enrich_origin = (
            as_origin
            if as_origin and any(c.origin == as_origin for c in visible_consumers)
            else primary.origin
        )

        try:
            sama = _sama_client(project_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping project %s: %s", project_id, exc)
            continue

        imported_ids: set[str] = set()
        if enrich:
            try:
                crid = get_crid(enrich_origin)
                imported_ids = _imported_batch_ids(crid, project_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Enrichment failed for %s: %s", enrich_origin, exc)

        try:
            batches = _cached_get_batches(sama)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_batches failed for project %s: %s", project_id, exc)
            continue

        filtered = []
        for batch in batches:
            updated_at_raw = batch.get("updated_at")
            if not updated_at_raw:
                continue
            try:
                updated_at = datetime.strptime(
                    updated_at_raw, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=UTC)
            except ValueError:
                continue
            if updated_at < cutoff:
                continue
            filtered.append((updated_at, batch))

        filtered.sort(key=lambda item: item[0], reverse=True)
        if batch_count > 0:
            filtered = filtered[:batch_count]

        for _, batch in filtered:
            batch_id = int(batch["id"])
            imported = str(batch_id) in imported_ids
            out.append(
                BatchEntry(
                    origin=primary.origin,
                    project_type=primary.project_type,
                    environment=primary.environment,
                    project_id=project_id,
                    camera=primary.camera,
                    batch_id=batch_id,
                    state=batch.get("state"),
                    updated_at=batch.get("updated_at"),
                    total_tasks=batch.get("total_tasks"),
                    created_tasks=batch.get("created_tasks"),
                    status_counts={},
                    dataset_topic=None,
                    imported=imported,
                    masks_exist=False,
                    approved=False,
                    consumers=visible_consumers,
                )
            )

    out.sort(key=lambda b: b.updated_at or "", reverse=True)
    return BatchListResponse(batches=out)


@router.get("/batches/{project_id}/{batch_id}/detail")
def get_batch_detail(project_id: int, batch_id: int, origin: str | None = None):
    """Pull richer per-batch info on demand: status counts + dataset_topic + flags."""
    try:
        sama = _sama_client(project_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    summary: dict[str, int] = {}
    try:
        summary = {
            k: int(v) for k, v in (sama.get_batch_status_summary(batch_id) or {}).items()
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("status summary failed for %s/%s: %s", project_id, batch_id, exc)

    dataset_topic = None
    masks = approved = imported = False
    if origin:
        try:
            crid = get_crid(origin)
            imported_ids = _imported_batch_ids(crid, project_id)
            imported = str(batch_id) in imported_ids
            if imported:
                dataset_topic = _topic_for_batch(crid, sama, batch_id)
                if dataset_topic:
                    masks = _masks_exist(dataset_topic)
                    approved = _approved(dataset_topic)
        except Exception as exc:  # noqa: BLE001
            logger.warning("detail enrichment failed: %s", exc)

    return {
        "project_id": project_id,
        "batch_id": batch_id,
        "status_counts": summary,
        "dataset_topic": dataset_topic,
        "imported": imported,
        "masks_exist": masks,
        "approved": approved,
    }


@router.post("/jobs/import")
def post_import_job(request: ImportJobRequest):
    def worker(reporter: ProgressReporter):
        return import_batches_for_project(
            origin=request.origin,
            project_id=request.project_id,
            batch_ids=request.batch_ids,
            dry_run=request.dry_run,
            import_in_progress=request.import_in_progress,
            skip_confirm=request.skip_confirm,
            reporter=reporter,
        )

    job = job_manager.start_job("sama_import", worker)
    return {"job_id": job.job_id, "kind": job.kind, "state": job.state}


@router.post("/jobs/masks")
def post_mask_job(request: MaskJobRequest):
    def worker(reporter: ProgressReporter):
        return generate_masks_for_topics(
            origin=request.origin,
            dataset_topics=request.dataset_topics,
            camera=request.camera,
            redo=request.redo,
            styles=request.styles,
            reporter=reporter,
        )

    job = job_manager.start_job("sama_masks", worker)
    return {"job_id": job.job_id, "kind": job.kind, "state": job.state}


@router.post("/jobs/pipeline")
def post_pipeline_job(request: PipelineJobRequest):
    runs = list(request.runs)
    redo_masks = request.redo_masks
    cameras = {
        (entry["origin"], entry["project_id"]): entry["camera"]
        for entry in list_projects()
    }

    def worker(reporter: ProgressReporter):
        if not _pipeline_lock.acquire(blocking=False):
            raise RuntimeError("Another pipeline run is in progress.")
        try:
            results = []
            for idx, run in enumerate(runs, start=1):
                reporter.status(
                    "pipeline",
                    f"[{idx}/{len(runs)}] {run.origin} (project {run.project_id})",
                )
                camera = cameras.get((run.origin, run.project_id))
                if camera is None:
                    raise RuntimeError(
                        f"No camera configured for {run.origin}/{run.project_id}"
                    )
                results.append(
                    {
                        "origin": run.origin,
                        "project_id": run.project_id,
                        "result": run_pipeline(
                            origin=run.origin,
                            project_id=run.project_id,
                            batch_ids=run.batch_ids,
                            dry_run=run.dry_run,
                            import_in_progress=run.import_in_progress,
                            redo_masks=redo_masks,
                            camera=camera,
                            skip_confirm=request.skip_confirm,
                            reporter=reporter,
                        ),
                    }
                )
            return {"runs": results}
        finally:
            _pipeline_lock.release()

    job = job_manager.start_job("sama_pipeline", worker)
    return {"job_id": job.job_id, "kind": job.kind, "state": job.state}


@router.post("/approve")
def post_approve(request: ApprovalRequest):
    cache_dir = dataset_cache_root() / request.dataset_topic
    if not cache_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset topic '{request.dataset_topic}' has no cache dir.",
        )
    marker = cache_dir / "APPROVED"
    if request.approved:
        marker.write_text(datetime.now(UTC).isoformat())
    elif marker.exists():
        marker.unlink()
    return {"dataset_topic": request.dataset_topic, "approved": request.approved}


@router.get("/mosaic/{dataset_topic}")
def get_mosaic(dataset_topic: str):
    cache_dir = dataset_cache_root() / dataset_topic
    candidates = list(cache_dir.glob(f"mosaic_{dataset_topic}_images.jpg"))
    if not candidates:
        candidates = list(cache_dir.glob("mosaic_*_images.jpg"))
    if not candidates:
        raise HTTPException(status_code=404, detail="Mosaic not found.")
    return FileResponse(candidates[0], media_type="image/jpeg")


@router.get("/topic/{dataset_topic}")
def get_topic_state(dataset_topic: str):
    cache_dir = dataset_cache_root() / dataset_topic
    return {
        "dataset_topic": dataset_topic,
        "exists": cache_dir.exists(),
        "masks_exist": _masks_exist(dataset_topic),
        "approved": _approved(dataset_topic),
    }
