from __future__ import annotations

import hashlib
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

from interface.crid import CRID
from interface.model.models import SensorName
from fastapi.responses import FileResponse

from active_learning.api.config_service import (
    build_model_options,
    build_project_summary,
    get_default_config_path,
)
from active_learning.api.query_cache import (
    QueryCacheStore,
    build_query_fingerprint,
    serialize_query_artifacts,
)
from active_learning.api.schemas import (
    AlgesSettings,
    CoresetSettings,
    PreviewItem,
    QueryPreviewRequest,
    QueryPreviewResponse,
    QuerySettings,
    SelectionSettings,
    ExportSeedResponse,
    UncertaintySettings,
)
from active_learning.api.progress import ProgressReporter
from active_learning.core.config import (
    VALID_STRATEGIES,
    brightness_filter_inactive,
    build_seed_config,
    load_yaml,
    parse_query_datetime,
    resolve_project,
)
from active_learning.core.image_provider import ImageProvider
from active_learning.core.time_gap import filter_ids_by_min_milliseconds_between_images
from active_learning.core.types import SelectionResult
from active_learning.integrations.crid.export import (
    CridExportResult,
    build_export_description,
    export_selection,
)
from active_learning.integrations.crid.provider_source import CridImageProviderSource
from active_learning.integrations.crid.source import CridSource
from active_learning.scorers.brightness import filter_by_brightness
from active_learning.sinks.mosaic import render_overlay_mosaic, save_prediction_mask
from active_learning.sinks.sama import CridSamaSink


@dataclass(slots=True)
class QueryPreviewArtifacts:
    cfg: object
    all_ids: list[str]
    labeled_ids: list[str]
    seeded_ids_count: int
    pool_ids: list[str]
    candidate_ids: list[str]


_SHARED_CRID = None
_SHARED_PROVIDER: ImageProvider | None = None
_PROVIDER_LOCK = threading.Lock()


def get_crid():
    global _SHARED_CRID
    if _SHARED_CRID is None:
        _SHARED_CRID = CRID(globals())
    return _SHARED_CRID


def build_query_preview_config(
    request: QueryPreviewRequest,
    config_path: str | Path | None = None,
    *,
    ensure_model_downloads: bool = False,
):
    path = Path(config_path or get_default_config_path())
    raw = load_yaml(str(path))
    resolved = resolve_project(raw, request.project_name)
    resolved.setdefault("query", {})
    query = resolved["query"]
    for field_name in (
        "exclude_seeded",
        "exclude_al_excluded",
        "min_brightness",
        "max_brightness",
        "brightness_filter_enabled",
        "start",
        "end",
        "use_full_res_images",
    ):
        value = getattr(request, field_name)
        if value is not None:
            query[field_name] = value
    if request.strategy is not None:
        resolved.setdefault("selection", {})
        resolved["selection"]["strategy"] = request.strategy
    selection = resolved.setdefault("selection", {})
    if request.n_select is not None:
        selection["n_select"] = request.n_select
    if request.seed is not None:
        selection["seed"] = request.seed
    if request.min_milliseconds_between_images is not None:
        query["min_milliseconds_between_images"] = (
            request.min_milliseconds_between_images
        )
    if request.feature_model is not None:
        resolved.setdefault("coreset", {})
        resolved["coreset"]["feature_model"] = request.feature_model
        resolved.setdefault("uncertainty_coreset", {})
        resolved["uncertainty_coreset"]["feature_model"] = request.feature_model
    if request.uncertainty_model is not None:
        resolved.setdefault("uncertainty_coreset", {})
        resolved["uncertainty_coreset"]["uncertainty_model"] = request.uncertainty_model
    if (
        request.uncertainty_model_url is not None
        and request.uncertainty_model is not None
    ):
        resolved.setdefault("models", {})
        resolved["models"].setdefault(request.uncertainty_model, {})
        resolved["models"][request.uncertainty_model]["url"] = (
            request.uncertainty_model_url
        )
        resolved["models"][request.uncertainty_model].pop("path", None)
    if request.alpha is not None:
        resolved.setdefault("uncertainty_coreset", {})
        resolved["uncertainty_coreset"]["alpha"] = request.alpha
    if request.provider is not None:
        resolved.setdefault("uncertainty_coreset", {})
        resolved["uncertainty_coreset"]["provider"] = request.provider
    if request.mc_iterations is not None:
        resolved.setdefault("uncertainty_coreset", {})
        resolved["uncertainty_coreset"]["mc_iterations"] = request.mc_iterations
    if request.batch_size is not None:
        resolved.setdefault("uncertainty_coreset", {})
        resolved["uncertainty_coreset"]["batch_size"] = request.batch_size
    if request.aggregation is not None:
        resolved.setdefault("uncertainty_coreset", {})
        resolved["uncertainty_coreset"]["aggregation"] = request.aggregation
    if request.topk_fraction is not None:
        resolved.setdefault("uncertainty_coreset", {})
        resolved["uncertainty_coreset"]["topk_fraction"] = request.topk_fraction
    if request.candidate_multiplier is not None:
        resolved.setdefault("uncertainty_coreset", {})
        resolved["uncertainty_coreset"]["candidate_multiplier"] = (
            request.candidate_multiplier
        )
    if request.target_classes is not None:
        resolved.setdefault("uncertainty_coreset", {})
        resolved["uncertainty_coreset"]["target_classes"] = request.target_classes
    if request.alges_model is not None:
        resolved.setdefault("alges", {})
        resolved["alges"]["model"] = request.alges_model
    if request.alges_model_url is not None and request.alges_model is not None:
        resolved.setdefault("models", {})
        resolved["models"].setdefault(request.alges_model, {})
        resolved["models"][request.alges_model]["url"] = request.alges_model_url
        resolved["models"][request.alges_model].pop("path", None)
    if request.method is not None:
        resolved.setdefault("alges", {})
        resolved["alges"]["method"] = request.method
    if request.alges_batch_size is not None:
        resolved.setdefault("alges", {})
        resolved["alges"]["batch_size"] = request.alges_batch_size
    return build_seed_config(resolved, ensure_model_downloads=ensure_model_downloads)


def build_preview_provider(cfg, crid):
    global _SHARED_PROVIDER
    with _PROVIDER_LOCK:
        if _SHARED_PROVIDER is not None:
            return _SHARED_PROVIDER
        source = CridImageProviderSource(
            crid,
            Path(cfg.query.cache_root) / "webapp_source",
        )
        _SHARED_PROVIDER = ImageProvider(
            source,
            cache_root=Path(cfg.query.cache_root) / "webapp_provider",
            ignore_missing_blobs=True,
        )
        return _SHARED_PROVIDER


def apply_brightness_filter(
    cfg,
    provider,
    sample_ids: list[str],
    *,
    reporter: ProgressReporter | None = None,
) -> list[str]:
    if brightness_filter_inactive(cfg.query):
        if reporter:
            reporter.skip_stages("brightness_filter")
        return list(sample_ids)
    filtered_ids, _ = filter_by_brightness(
        sample_ids,
        provider,
        cfg.query.cache_root,
        min_brightness=cfg.query.min_brightness,
        max_brightness=cfg.query.max_brightness,
    )
    return filtered_ids


def collect_query_preview_artifacts(
    request: QueryPreviewRequest,
    config_path: str | Path | None = None,
    *,
    reporter: ProgressReporter | None = None,
) -> QueryPreviewArtifacts:
    if reporter:
        reporter.status("query_pool", "Listing candidate IDs in time range")
    cfg = build_query_preview_config(request, config_path=config_path)
    if reporter and not cfg.query.exclude_seeded:
        reporter.skip_stages("fetch_seeded")
    crid = get_crid()
    sensor = getattr(SensorName, cfg.query.sensor)
    source = CridSource(crid)

    def on_seeded_progress(completed: int, total: int) -> None:
        if reporter:
            reporter.progress(
                "fetch_seeded",
                completed=completed,
                total=total,
                message=f"Collecting seeded IDs from Sama ({completed}/{total} batches)",
            )

    tag_filter = ["al-exclusion"] if cfg.query.exclude_al_excluded else None
    start_dt = parse_query_datetime(cfg.query.start)
    end_dt = parse_query_datetime(cfg.query.end)

    all_ids = source.query_ids(
        sensor_name=sensor,
        exclude_crid_tags=tag_filter,
        start_datetime=start_dt,
        end_datetime=end_dt,
        max_size=1000000,
    )
    if reporter:
        reporter.progress(
            "query_pool",
            completed=1,
            total=1,
            message=f"Query pool: {len(all_ids)} candidate IDs in range",
        )
        reporter.check_cancelled()
        reporter.status("query_labeled", "Resolving labeled IDs in time range")
    labeled_ids = source.query_labeled_ids(
        sensor_name=sensor,
        start_datetime=start_dt,
        end_datetime=end_dt,
        max_size=1000000,
    )
    if reporter:
        reporter.progress(
            "query_labeled",
            completed=1,
            total=1,
            message=(
                f"Query labeled: {len(labeled_ids)} labeled IDs "
                f"(candidates in range: {len(all_ids)})"
            ),
        )
        reporter.check_cancelled()
    if reporter and cfg.query.exclude_seeded:
        reporter.status("fetch_seeded", "Collecting seeded IDs from Sama")
    queried = source.build_pool_with_seeded_exclusions(
        all_ids,
        labeled_ids,
        exclude_seeded=cfg.query.exclude_seeded,
        sama_project_id=cfg.export.sama_project_id,
        on_seeded_progress=on_seeded_progress if cfg.query.exclude_seeded else None,
    )
    pool_ids = queried.pool_ids
    if reporter:
        reporter.check_cancelled()
    if cfg.query.min_milliseconds_between_images > 0:
        if reporter:
            reporter.status("time_gap_filter", "Applying time-gap filter")
        pool_ids = filter_ids_by_min_milliseconds_between_images(
            pool_ids,
            cfg.query.min_milliseconds_between_images,
        )
        if reporter:
            reporter.progress(
                "time_gap_filter",
                completed=1,
                total=1,
                message=(
                    f"Time-gap filter: {len(pool_ids)}/{len(queried.pool_ids)} "
                    f"(min {cfg.query.min_milliseconds_between_images} ms)"
                ),
            )
    else:
        if reporter:
            reporter.skip_stages("time_gap_filter")
    skip_brightness_filter = brightness_filter_inactive(cfg.query)
    if reporter:
        reporter.check_cancelled()
    if reporter and not skip_brightness_filter:
        reporter.status("brightness_filter", "Computing brightness filter")
    provider = build_preview_provider(cfg, crid)
    candidate_ids = apply_brightness_filter(
        cfg,
        provider,
        pool_ids,
        reporter=reporter,
    )
    if reporter and not skip_brightness_filter:
        reporter.progress(
            "brightness_filter",
            completed=len(candidate_ids),
            total=max(len(pool_ids), 1),
            message=f"{len(candidate_ids)} images remain after brightness filter",
        )
    return QueryPreviewArtifacts(
        cfg=cfg,
        all_ids=all_ids,
        labeled_ids=queried.labeled_ids,
        seeded_ids_count=queried.seeded_ids_count,
        pool_ids=queried.pool_ids,
        candidate_ids=candidate_ids,
    )


def _query_cache_store(cfg) -> QueryCacheStore:
    return QueryCacheStore(cfg.query.cache_root)


def build_export_config(export_context) -> object:
    models = {
        model.name: SimpleNamespace(
            name=model.name,
            type=model.type,
            path=model.path,
            url=model.url,
            image_size=tuple(model.image_size),
        )
        for model in export_context.models
    }
    return SimpleNamespace(
        query=SimpleNamespace(
            cache_root=export_context.query.cache_root,
            sama_priority=export_context.export.sama_priority,
            start=export_context.query.start,
            end=export_context.query.end,
            use_full_res_images=True,
            sensor=export_context.project.query_sensor,
        ),
        selection=SimpleNamespace(
            strategy=export_context.selection.strategy,
            n_select=export_context.selection.n_select,
            seed=export_context.selection.seed,
        ),
        coreset=SimpleNamespace(feature_model=None),
        uncertainty_coreset=SimpleNamespace(
            feature_model=export_context.uncertainty_coreset.feature_model,
            uncertainty_model=export_context.uncertainty_coreset.uncertainty_model,
            alpha=export_context.uncertainty_coreset.alpha,
            provider=export_context.uncertainty_coreset.provider,
            mc_iterations=export_context.uncertainty_coreset.mc_iterations,
            batch_size=export_context.uncertainty_coreset.batch_size,
            aggregation=export_context.uncertainty_coreset.aggregation,
            topk_fraction=export_context.uncertainty_coreset.topk_fraction,
            candidate_multiplier=export_context.uncertainty_coreset.candidate_multiplier,
            target_classes=getattr(
                export_context.uncertainty_coreset,
                "target_classes",
                [],
            ),
        ),
        alges=SimpleNamespace(
            model=export_context.alges.model,
            method=export_context.alges.method,
            batch_size=export_context.alges.batch_size,
        ),
        export=SimpleNamespace(
            sama_project_id=export_context.export.sama_project_id,
            prefix=export_context.project.export_prefix,
            project=export_context.project.export_project,
            mosaic_path="mosaic.jpg",
        ),
        models=models,
    )


def _query_stage_reuse_plan(cfg) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    cached = ["query_pool", "query_labeled"]
    skipped: list[str] = []

    if cfg.query.exclude_seeded:
        cached.append("fetch_seeded")
    else:
        skipped.append("fetch_seeded")

    if cfg.query.min_milliseconds_between_images > 0:
        cached.append("time_gap_filter")
    else:
        skipped.append("time_gap_filter")

    if brightness_filter_inactive(cfg.query):
        skipped.append("brightness_filter")
    else:
        cached.append("brightness_filter")

    terminal_stage = cached[-1] if cached else "query_pool"
    return tuple(cached), tuple(skipped), terminal_stage


def _load_cached_query_artifacts(request: QueryPreviewRequest, config_path=None):
    if not request.query_result_token:
        return None
    cfg = build_query_preview_config(request, config_path=config_path)
    store = _query_cache_store(cfg)
    fingerprint = build_query_fingerprint(cfg, request.project_name)
    payload = store.load(request.query_result_token, fingerprint)
    if payload is None:
        return None
    return QueryPreviewArtifacts(
        cfg=cfg,
        all_ids=list(payload["all_ids"]),
        labeled_ids=list(payload["labeled_ids"]),
        seeded_ids_count=int(payload["seeded_ids_count"]),
        pool_ids=list(payload["pool_ids"]),
        candidate_ids=list(payload["candidate_ids"]),
    )


def _store_query_artifacts(
    request: QueryPreviewRequest, artifacts: QueryPreviewArtifacts
) -> str:
    store = _query_cache_store(artifacts.cfg)
    fingerprint = build_query_fingerprint(artifacts.cfg, request.project_name)
    return store.save(fingerprint, serialize_query_artifacts(artifacts))


def build_thumbnail_url(sample_id: str) -> str:
    return f"/api/al/thumbnails/{quote(sample_id, safe='')}"


def build_mask_url(token: str) -> str:
    return f"/api/al/masks/{token}"


def build_selection_settings(cfg) -> SelectionSettings:
    return SelectionSettings(
        strategy=cfg.selection.strategy,
        available_strategies=list(VALID_STRATEGIES),
        n_select=cfg.selection.n_select,
        seed=cfg.selection.seed,
    )


def build_coreset_settings(cfg) -> CoresetSettings:
    return CoresetSettings(feature_model=cfg.coreset.feature_model)


def build_uncertainty_settings(cfg) -> UncertaintySettings:
    return UncertaintySettings(
        feature_model=cfg.uncertainty_coreset.feature_model,
        uncertainty_model=cfg.uncertainty_coreset.uncertainty_model,
        alpha=cfg.uncertainty_coreset.alpha,
        provider=cfg.uncertainty_coreset.provider,
        mc_iterations=cfg.uncertainty_coreset.mc_iterations,
        batch_size=cfg.uncertainty_coreset.batch_size,
        aggregation=cfg.uncertainty_coreset.aggregation,
        topk_fraction=cfg.uncertainty_coreset.topk_fraction,
        candidate_multiplier=cfg.uncertainty_coreset.candidate_multiplier,
        target_classes=cfg.uncertainty_coreset.target_classes,
    )


def build_alges_settings(cfg) -> AlgesSettings:
    return AlgesSettings(
        model=cfg.alges.model,
        method=cfg.alges.method,
        batch_size=cfg.alges.batch_size,
    )


def _supports_overlay_mosaic(cfg) -> bool:
    strategy = cfg.selection.strategy
    if strategy in (
        "uncertainty_coreset",
        "uncertainty_topk",
        "uncertainty_topk_coreset",
    ):
        model_def = cfg.models[cfg.uncertainty_coreset.uncertainty_model]
        return model_def.type == "unet"
    if strategy in ("alges", "alges_coreset"):
        model_def = cfg.models[cfg.alges.model]
        return model_def.type == "unet"
    return False


def _predict_preview_overlay_probs(
    unet, image_paths: list[str], image_size, batch_size: int
):
    from PIL import Image

    import numpy as np

    from active_learning.providers.inference import build_infer_fn, extract_probs

    width, height = image_size
    infer = build_infer_fn(unet.model, training=False)
    probs_batches = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_images = []
        for path in batch_paths:
            image = Image.open(path).convert("RGB").resize((width, height))
            batch_images.append(np.asarray(image, dtype=np.float32) / 255.0)
        batch = np.stack(batch_images, axis=0)
        result = infer(batch)
        probs = extract_probs(result).numpy()
        probs_batches.extend(probs)
    return probs_batches


def _mask_token(
    sample_id: str, model_path: str, cache_root: str, use_full_res_images: bool
) -> str:
    payload = "|".join((sample_id, model_path, cache_root, str(use_full_res_images)))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def generate_preview_masks(cfg, provider, sample_ids: list[str]) -> dict[str, str]:
    strategy = cfg.selection.strategy
    if strategy not in (
        "uncertainty_coreset",
        "uncertainty_topk",
        "uncertainty_topk_coreset",
    ):
        return {}

    model_def = cfg.models[cfg.uncertainty_coreset.uncertainty_model]
    if model_def.type != "unet":
        return {}

    from active_learning.providers.unet import load_unet

    unet = load_unet(model_def.path)
    raw_paths = (
        provider.get_highres_batch(sample_ids, progress=False, allow_missing=True)
        if cfg.query.use_full_res_images
        else provider.get_lowres_batch(sample_ids, progress=False, allow_missing=True)
    )
    kept = [(sid, p) for sid, p in zip(sample_ids, raw_paths, strict=True) if p]
    if not kept:
        return {}

    kept_ids, image_paths = zip(*kept)
    kept_ids = list(kept_ids)
    image_paths = list(image_paths)
    probs_batch = _predict_preview_overlay_probs(
        unet,
        image_paths,
        image_size=tuple(model_def.image_size),
        batch_size=cfg.uncertainty_coreset.batch_size,
    )
    mask_dir = Path(cfg.query.cache_root) / "webapp_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for sample_id, probs in zip(kept_ids, probs_batch, strict=True):
        token = _mask_token(
            sample_id,
            str(model_def.path),
            str(cfg.query.cache_root),
            bool(cfg.query.use_full_res_images),
        )
        mask_path = mask_dir / f"{token}.png"
        save_prediction_mask(str(mask_path), probs)
        out[sample_id] = build_mask_url(token)
    return out


def run_query_preview(
    request: QueryPreviewRequest,
    config_path: str | Path | None = None,
    *,
    reporter: ProgressReporter | None = None,
) -> QueryPreviewResponse:
    artifacts = collect_query_preview_artifacts(
        request,
        config_path=config_path,
        reporter=reporter,
    )
    query_result_token = _store_query_artifacts(request, artifacts)
    sample_size = min(request.sample_size, len(artifacts.candidate_ids))
    if reporter:
        reporter.status("sample_preview", "Sampling preview thumbnails")
    preview_ids = (
        random.sample(artifacts.candidate_ids, sample_size) if sample_size else []
    )
    if reporter:
        reporter.progress(
            "sample_preview",
            completed=len(preview_ids),
            total=max(sample_size, 1),
            message=f"Prepared {len(preview_ids)} preview items",
        )
    summary = build_project_summary(
        config_path or get_default_config_path(),
        request.project_name,
    )
    query = QuerySettings(
        cache_root=str(artifacts.cfg.query.cache_root),
        sama_priority=artifacts.cfg.query.sama_priority,
        exclude_seeded=artifacts.cfg.query.exclude_seeded,
        exclude_al_excluded=artifacts.cfg.query.exclude_al_excluded,
        min_brightness=artifacts.cfg.query.min_brightness,
        max_brightness=artifacts.cfg.query.max_brightness,
        brightness_filter_enabled=artifacts.cfg.query.brightness_filter_enabled,
        start=artifacts.cfg.query.start,
        end=artifacts.cfg.query.end,
        use_full_res_images=artifacts.cfg.query.use_full_res_images,
        min_milliseconds_between_images=artifacts.cfg.query.min_milliseconds_between_images,
    )
    return QueryPreviewResponse(
        result_kind="query",
        project=summary,
        query=query,
        selection=build_selection_settings(artifacts.cfg),
        models=build_model_options(artifacts.cfg),
        coreset=build_coreset_settings(artifacts.cfg),
        uncertainty_coreset=build_uncertainty_settings(artifacts.cfg),
        alges=build_alges_settings(artifacts.cfg),
        all_ids_count=len(artifacts.all_ids),
        labeled_ids_count=len(artifacts.labeled_ids),
        seeded_ids_count=artifacts.seeded_ids_count,
        pool_ids_count=len(artifacts.pool_ids),
        brightness_filtered_ids_count=len(artifacts.candidate_ids),
        preview_sample_ids=preview_ids,
        preview_items=[
            PreviewItem(
                sample_id=sample_id, thumbnail_url=build_thumbnail_url(sample_id)
            )
            for sample_id in preview_ids
        ],
        query_result_token=query_result_token,
    )


def _prepare_strategy_models(cfg):
    strategy = cfg.selection.strategy
    uncertainty_model = None
    alges_model = None
    if strategy in (
        "uncertainty_coreset",
        "uncertainty_topk",
        "uncertainty_topk_coreset",
    ):
        from active_learning.providers.unet import enable_mc_dropout, load_unet

        uc = cfg.uncertainty_coreset
        uc_model_def = cfg.models[uc.uncertainty_model]
        unet = load_unet(uc_model_def.path)
        uncertainty_model = (
            enable_mc_dropout(unet) if uc.provider in {"mc_dropout", "bald"} else unet
        )

    if strategy in ("alges", "alges_coreset"):
        from active_learning.providers.unet import (
            create_penultimate_model,
            enable_mc_dropout,
            load_unet,
        )

        al = cfg.alges
        al_model_def = cfg.models[al.model]
        al_unet = load_unet(al_model_def.path)
        alges_model = create_penultimate_model(enable_mc_dropout(al_unet))

    return uncertainty_model, alges_model


def fetch_excluded_ids(crid, sample_ids: list[str]) -> set[str]:
    """Return the subset of sample_ids that carry the al-exclusion tag in ES."""
    result = crid.elasticsearch_client.find_by_ids(sample_ids)
    excluded: set[str] = set()
    for doc in result.get("docs", []):
        if doc.get("found") and "al-exclusion" in doc.get("_source", {}).get(
            "crid_tags", []
        ):
            excluded.add(doc["_id"])
    return excluded


def run_strategy_preview(
    request: QueryPreviewRequest,
    config_path: str | Path | None = None,
    *,
    reporter: ProgressReporter | None = None,
) -> QueryPreviewResponse:
    from active_learning.core.selection import run_selection

    cache_hit = False
    artifacts = _load_cached_query_artifacts(request, config_path=config_path)
    if artifacts is not None:
        cache_hit = True
        if reporter:
            cached_stages, skipped_stages, stage = _query_stage_reuse_plan(
                artifacts.cfg
            )
            reporter.cache_stages(*cached_stages)
            reporter.skip_stages(*skipped_stages)
            reporter.status(stage, "Reusing cached query results")
            reporter.progress(
                stage,
                completed=len(artifacts.candidate_ids),
                total=max(len(artifacts.pool_ids), 1),
                message=(
                    f"Reused cached query results with "
                    f"{len(artifacts.candidate_ids)} candidate images"
                ),
            )
    else:
        artifacts = collect_query_preview_artifacts(
            QueryPreviewRequest(**request.model_dump()),
            config_path=config_path,
            reporter=reporter,
        )
    artifacts.cfg = build_query_preview_config(
        request,
        config_path=config_path,
        ensure_model_downloads=True,
    )
    query_result_token = (
        request.query_result_token if cache_hit and request.query_result_token else None
    )
    if query_result_token is None:
        query_result_token = _store_query_artifacts(request, artifacts)
    crid = get_crid()
    provider = build_preview_provider(artifacts.cfg, crid)
    if reporter:
        reporter.check_cancelled()
        reporter.status("load_model", "Loading strategy models")
    uncertainty_model, alges_model = _prepare_strategy_models(artifacts.cfg)
    selection_progress = None
    if reporter:
        reporter.progress(
            "load_model", completed=1, total=1, message="Strategy models loaded"
        )

        def selection_progress(completed: int, total: int, msg: str | None) -> None:
            reporter.check_cancelled()
            reporter.progress(
                "select_samples",
                completed=completed,
                total=total,
                message=msg or "Selecting candidate images",
            )

    selection_result = run_selection(
        artifacts.cfg,
        candidate_ids=artifacts.candidate_ids,
        seed_ids=artifacts.labeled_ids,
        image_provider=provider,
        uncertainty_model=uncertainty_model,
        alges_model=alges_model,
        progress=False,
        selection_progress=selection_progress,
    )
    if reporter:
        reporter.progress(
            "select_samples",
            completed=len(selection_result.selected_ids),
            total=max(artifacts.cfg.selection.n_select, 1),
            message=f"Selected {len(selection_result.selected_ids)} images",
        )
        reporter.check_cancelled()
        reporter.status("materialize_preview", "Preparing preview thumbnails")
    preview_ids = list(selection_result.selected_ids)
    mask_urls = generate_preview_masks(artifacts.cfg, provider, preview_ids)
    excluded_ids = fetch_excluded_ids(crid, preview_ids)
    if reporter:
        reporter.progress(
            "materialize_preview",
            completed=len(preview_ids),
            total=max(len(preview_ids), 1),
            message=f"Prepared {len(preview_ids)} preview items",
        )
    summary = build_project_summary(
        config_path or get_default_config_path(),
        request.project_name,
    )
    query = QuerySettings(
        cache_root=str(artifacts.cfg.query.cache_root),
        sama_priority=artifacts.cfg.query.sama_priority,
        exclude_seeded=artifacts.cfg.query.exclude_seeded,
        exclude_al_excluded=artifacts.cfg.query.exclude_al_excluded,
        min_brightness=artifacts.cfg.query.min_brightness,
        max_brightness=artifacts.cfg.query.max_brightness,
        brightness_filter_enabled=artifacts.cfg.query.brightness_filter_enabled,
        start=artifacts.cfg.query.start,
        end=artifacts.cfg.query.end,
        use_full_res_images=artifacts.cfg.query.use_full_res_images,
        min_milliseconds_between_images=artifacts.cfg.query.min_milliseconds_between_images,
    )
    return QueryPreviewResponse(
        result_kind="strategy",
        project=summary,
        query=query,
        selection=build_selection_settings(artifacts.cfg),
        models=build_model_options(artifacts.cfg),
        coreset=build_coreset_settings(artifacts.cfg),
        uncertainty_coreset=build_uncertainty_settings(artifacts.cfg),
        alges=build_alges_settings(artifacts.cfg),
        all_ids_count=len(artifacts.all_ids),
        labeled_ids_count=len(artifacts.labeled_ids),
        seeded_ids_count=artifacts.seeded_ids_count,
        pool_ids_count=len(artifacts.pool_ids),
        brightness_filtered_ids_count=len(artifacts.candidate_ids),
        preview_sample_ids=preview_ids,
        preview_items=[
            PreviewItem(
                sample_id=sample_id,
                thumbnail_url=build_thumbnail_url(sample_id),
                mask_url=mask_urls.get(sample_id),
                excluded=sample_id in excluded_ids,
            )
            for sample_id in preview_ids
        ],
        query_result_token=query_result_token,
        query_cache_hit=cache_hit,
        selected_ids_count=len(selection_result.selected_ids),
        selected_ids=list(selection_result.selected_ids),
        overlay_available=_supports_overlay_mosaic(artifacts.cfg),
    )


def build_overlay_mosaic_download(export_context, selected_ids: list[str]):
    export_ids = list(dict.fromkeys(selected_ids))
    if not export_ids:
        raise ValueError("No images selected for export.")

    cfg = build_export_config(export_context)
    if not _supports_overlay_mosaic(cfg):
        raise ValueError(
            "The selected strategy does not have a UNet overlay model available."
        )

    original_model = cfg.alges.model
    original_batch_size = cfg.alges.batch_size
    use_uncertainty_model = cfg.selection.strategy in (
        "uncertainty_coreset",
        "uncertainty_topk",
        "uncertainty_topk_coreset",
    )
    if use_uncertainty_model:
        cfg.alges.model = cfg.uncertainty_coreset.uncertainty_model
        cfg.alges.batch_size = cfg.uncertainty_coreset.batch_size

    try:
        crid = get_crid()
        provider = build_preview_provider(cfg, crid)
        output_path = render_overlay_mosaic(
            SelectionResult(selected_ids=export_ids),
            provider,
            cfg=cfg,
            progress=False,
        )
    finally:
        cfg.alges.model = original_model
        cfg.alges.batch_size = original_batch_size

    filename = Path(output_path).name
    return FileResponse(
        output_path,
        media_type="image/jpeg",
        filename=filename,
    )


def seed_strategy_selection(
    export_context, selected_ids: list[str]
) -> ExportSeedResponse:
    export_ids = list(dict.fromkeys(selected_ids))
    if not export_ids:
        raise ValueError("No images selected for export.")

    cfg = build_export_config(export_context)
    result = SelectionResult(selected_ids=export_ids)
    export_id, description = build_export_description(
        result,
        cfg,
        cfg.selection.strategy,
    )
    crid = get_crid()
    if cfg.export.sama_project_id:
        final_export = CridSamaSink(
            crid,
            sama_project_id=cfg.export.sama_project_id,
            priority=cfg.query.sama_priority,
            overwrite=False,
        ).submit(result, export_id=export_id, description=description)
        return ExportSeedResponse(
            export_id=final_export.export_id,
            description=final_export.description,
            sama_batch_id=final_export.sama_batch_id,
            image_count=final_export.image_count,
        )

    export_result = export_selection(
        crid,
        result,
        export_id=export_id,
        description=description,
        overwrite=False,
    )
    final_export = CridExportResult(
        export_id=export_result.export_id,
        description=export_result.description,
    )
    return ExportSeedResponse(
        export_id=final_export.export_id,
        description=final_export.description,
        image_count=len(result.selected_ids),
    )


_ADD_AL_EXCLUSION_TAG_SCRIPT = """
if (ctx._source.crid_tags == null) {
  ctx._source.crid_tags = new ArrayList();
}
if (!ctx._source.crid_tags.contains(params.tag)) {
  ctx._source.crid_tags.add(params.tag);
}
""".strip()


def write_exclusion_tags(selected_ids: list[str], excluded_ids: list[str]) -> int:
    from elasticsearch import helpers

    allowed = set(selected_ids)
    valid = [sid for sid in excluded_ids if sid in allowed]
    if not valid:
        return 0
    es = get_crid().elasticsearch_client
    actions = [
        {
            "_op_type": "update",
            "_index": es.index,
            "_id": sid,
            "script": {
                "source": _ADD_AL_EXCLUSION_TAG_SCRIPT,
                "lang": "painless",
                "params": {"tag": "al-exclusion"},
            },
        }
        for sid in valid
    ]
    helpers.bulk(es.es_client, actions, raise_on_error=True)
    es.es_client.indices.refresh(index=es.index)
    return len(valid)
