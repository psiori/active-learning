from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from active_learning.api import config_service, query_service, thumbnail_service
from active_learning.api.jobs import job_manager
from active_learning.api.sama_pipeline import router as sama_pipeline_router
from active_learning.api.query_cache import QueryCacheStore
from active_learning.api.schemas import (
    ConfigResponse,
    ExclusionTagsRequest,
    ExclusionTagsResponse,
    ExportSeedResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobSnapshotResponse,
    OverlayMosaicRequest,
    ProjectConfigResponse,
    QueryPreviewRequest,
    QueryPreviewResponse,
    SaveProjectConfigRequest,
    SeedExportRequest,
)
from active_learning.core.config import ConfigError
from active_learning.core.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


app = FastAPI(title="Active Learning API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sama_pipeline_router)


@app.get("/api/al/config", response_model=ConfigResponse)
def get_config():
    try:
        return config_service.list_projects()
    except ConfigError as exc:
        logger.exception("GET /api/al/config failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/al/projects/{project_name}", response_model=ProjectConfigResponse)
def get_project(project_name: str):
    try:
        return config_service.get_project_config(project_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConfigError as exc:
        logger.exception("GET /api/al/projects/%s failed", project_name)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/al/projects/{project_name}")
def patch_project(project_name: str, request: SaveProjectConfigRequest):
    try:
        config_service.save_project_config(project_name, request)
        return {"ok": True}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, Exception) as exc:
        logger.exception("PATCH /api/al/projects/%s failed", project_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/al/query-preview", response_model=QueryPreviewResponse)
def post_query_preview(request: QueryPreviewRequest):
    try:
        return query_service.run_query_preview(request)
    except ConfigError as exc:
        logger.exception("POST /api/al/query-preview failed (config)")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (AttributeError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("POST /api/al/query-preview failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/al/strategy-preview", response_model=QueryPreviewResponse)
def post_strategy_preview(request: QueryPreviewRequest):
    try:
        return query_service.run_strategy_preview(request)
    except ConfigError as exc:
        logger.exception("POST /api/al/strategy-preview failed (config)")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (AttributeError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("POST /api/al/strategy-preview failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/al/jobs", response_model=JobCreateResponse)
def post_job(request: JobCreateRequest):
    def worker(reporter):
        if request.kind == "query":
            return query_service.run_query_preview(
                request, reporter=reporter
            ).model_dump()
        if request.kind == "strategy":
            return query_service.run_strategy_preview(
                request, reporter=reporter
            ).model_dump()
        raise ValueError(f"Unknown job kind '{request.kind}'.")

    try:
        job = job_manager.start_job(request.kind, worker)
        return JobCreateResponse(job_id=job.job_id, kind=job.kind, state=job.state)
    except ConfigError as exc:
        logger.exception("POST /api/al/jobs failed (config)")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (AttributeError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("POST /api/al/jobs failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/al/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    try:
        job_manager.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' not found."
        ) from exc


@app.get("/api/al/jobs/{job_id}", response_model=JobSnapshotResponse)
def get_job(job_id: str):
    try:
        return JobSnapshotResponse(**job_manager.get(job_id).to_dict())
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' not found."
        ) from exc


@app.get("/api/al/jobs/{job_id}/events")
def get_job_events(job_id: str):
    try:
        return StreamingResponse(
            job_manager.event_stream(job_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' not found."
        ) from exc


@app.post("/api/al/export/overlay-mosaic")
def post_overlay_mosaic_download(request: OverlayMosaicRequest):
    try:
        return query_service.build_overlay_mosaic_download(
            request.export_context,
            selected_ids=request.selected_ids,
        )
    except (AttributeError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("POST /api/al/export/overlay-mosaic failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/al/export/seed", response_model=ExportSeedResponse)
def post_seed_strategy_selection(request: SeedExportRequest):
    try:
        return query_service.seed_strategy_selection(
            request.export_context,
            selected_ids=request.selected_ids,
        )
    except (AttributeError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("POST /api/al/export/seed failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/al/export/exclusion-tags", response_model=ExclusionTagsResponse)
def post_exclusion_tags(request: ExclusionTagsRequest):
    try:
        count = query_service.write_exclusion_tags(
            request.selected_ids,
            request.excluded_ids,
        )
        if count > 0 and request.project_name:
            cfg = query_service.build_query_preview_config(
                QueryPreviewRequest(project_name=request.project_name)
            )
            QueryCacheStore(cfg.query.cache_root).invalidate_all()
        return ExclusionTagsResponse(updated_count=count)
    except (AttributeError, RuntimeError, ValueError) as exc:
        logger.exception("POST /api/al/export/exclusion-tags failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/al/thumbnails/{sample_id:path}")
def get_thumbnail(
    sample_id: str,
    cache_root: str = Query(default="data/active_learning/feature_cache"),
    use_full_res_images: bool = Query(default=False),
):
    try:
        return thumbnail_service.build_thumbnail_response(
            sample_id,
            cache_root=cache_root,
            use_full_res_images=use_full_res_images,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/al/masks/{token}")
def get_mask(
    token: str,
    cache_root: str = Query(default="data/active_learning/feature_cache"),
):
    from pathlib import Path
    from fastapi.responses import FileResponse

    path = Path(cache_root) / "webapp_masks" / f"{token}.png"
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Mask not found for token '{token}'."
        )
    return FileResponse(path, media_type="image/png")
