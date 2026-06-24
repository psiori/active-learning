"""Select images using active learning and export for annotation."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    script_dir = Path(__file__).resolve().parent
    package_root = Path(__file__).resolve().parent.parent
    script_dir_str = str(script_dir)
    sys.path[:] = [path for path in sys.path if path not in {"", script_dir_str}]
    sys.path.insert(0, str(package_root))

from active_learning.core.logger import configure_logging
from active_learning.core.logger_tf import (
    describe_tensorflow_device,
    ensure_tensorflow_log_suppression,
)

ensure_tensorflow_log_suppression()

from interface.crid import CRID
from interface.model.models import SensorName

from active_learning.cli.common import (
    SEED_CLI_MAP,
    add_common_seed_args,
    add_crid_seed_args,
    prepare_strategy_models,
)
from active_learning.core import (
    apply_brightness_predicate,
    build_image_provider,
    run_selection,
)
from active_learning.core.config import (
    _UNSET,
    ConfigError,
    brightness_filter_inactive,
    build_seed_config,
    handle_model_path_override,
    load_yaml,
    merge_cli,
    parse_query_datetime,
    resolve_project,
)
from active_learning.core.time_gap import filter_ids_by_min_milliseconds_between_images
from active_learning.core.types import SelectionResult
from active_learning.integrations.crid.export import (
    build_export_description,
    export_selection,
)
from active_learning.integrations.crid.provider_source import CridImageProviderSource
from active_learning.integrations.crid.source import CridSource
from active_learning.sinks.mosaic import (
    mosaic_output_path,
    render_mosaic,
    render_overlay_mosaic,
)
from active_learning.sinks.sama import CridSamaSink
from active_learning.sinks.yaml import write_interactive_seed_payload

LOGGER_NAME = "active_learning.seed" if __name__ == "__main__" else __name__
logger = logging.getLogger(LOGGER_NAME)
CLI_PROGRESS = True


for key, value in list(os.environ.items()):
    if key.startswith("ENV_"):
        os.environ[key[4:]] = value


def select_samples(cfg, image_provider) -> SelectionResult:
    sensor = getattr(SensorName, cfg.query.sensor)
    crid = CRID(globals())
    source = CridSource(crid)
    logger.info("Querying pool (Elasticsearch, labeled, exclusions)...")
    queried = source.query_pool_and_labeled_ids(
        sensor_name=sensor,
        sama_project_id=cfg.export.sama_project_id,
        exclude_seeded=cfg.query.exclude_seeded,
        start_datetime=parse_query_datetime(cfg.query.start),
        end_datetime=parse_query_datetime(cfg.query.end),
        max_size=cfg.query.max_size,
    )
    logger.info(
        f"Pool ready: {len(queried.pool_ids)} candidate ids, "
        f"{len(queried.labeled_ids)} labeled ids for seed features."
    )
    pool_ids = queried.pool_ids
    if cfg.query.min_milliseconds_between_images > 0:
        n_before = len(pool_ids)
        pool_ids = filter_ids_by_min_milliseconds_between_images(
            pool_ids,
            cfg.query.min_milliseconds_between_images,
        )
        n_after = len(pool_ids)
        print(
            f"Time spacing on pool (min {cfg.query.min_milliseconds_between_images} ms between "
            f"images, from sample ids): {n_after}/{n_before} before brightness/download.",
            flush=True,
        )
        if n_after == 0:
            raise RuntimeError(
                "min_milliseconds_between_images left no pool candidates. "
                "Lower the value or widen the query window."
            )
    if brightness_filter_inactive(cfg.query):
        candidate_ids = list(pool_ids)
    else:
        candidate_ids, _ = apply_brightness_predicate(
            pool_ids,
            image_provider,
            cfg.query.cache_root,
            min_brightness=cfg.query.min_brightness,
            max_brightness=cfg.query.max_brightness,
        )
    if not candidate_ids:
        raise RuntimeError("Brightness filtering removed all images from the pool.")

    uncertainty_model, alges_model = prepare_strategy_models(cfg)

    return run_selection(
        cfg,
        candidate_ids=candidate_ids,
        seed_ids=queried.labeled_ids,
        image_provider=image_provider,
        uncertainty_model=uncertainty_model,
        alges_model=alges_model,
        progress=CLI_PROGRESS,
    )


def persist_outputs(cfg, result: SelectionResult, crid, image_provider) -> None:
    mosaic_path = render_mosaic(
        result,
        image_provider,
        mosaic_output_path(cfg),
        use_highres=cfg.query.use_full_res_images,
        resize_height=150,
        max_images=200,
        rows=10,
        cols=20,
        progress=CLI_PROGRESS,
    )
    logger.info("Mosaic saved to %s", mosaic_path)

    interactive_seed_path = write_interactive_seed_payload(
        result,
        cfg,
        mosaic_path=mosaic_path,
    )
    logger.info("Interactive seed payload saved to %s", interactive_seed_path)

    if cfg.export.overlay:
        overlay_path = render_overlay_mosaic(
            result,
            image_provider,
            cfg=cfg,
            progress=CLI_PROGRESS,
        )
        logger.info("Overlay mosaic saved to %s", overlay_path)

    export_id, description = build_export_description(
        result,
        cfg,
        cfg.selection.strategy,
    )
    if not cfg.export.seed:
        logger.info("Skipping Sama seeding/export (seed=false)")
        logger.info("Would seed/export %d images", len(result.selected_ids))
        logger.info("Export name (export_id): %s", export_id)
        logger.info("Description: %s", description)
        return

    if cfg.export.sama_project_id:
        submission_result = CridSamaSink(
            crid,
            sama_project_id=cfg.export.sama_project_id,
            priority=cfg.query.sama_priority,
            overwrite=False,
        ).submit(
            result,
            export_id=export_id,
            description=description,
        )
        logger.info("Export done! Container: %s", submission_result.export_id)
        if submission_result.sama_batch_id:
            logger.info("Sama batch: %s", submission_result.sama_batch_id)
        return

    export_result = export_selection(
        crid,
        result,
        export_id=export_id,
        description=description,
        overwrite=False,
    )
    logger.info("Export done! Container: %s", export_result.export_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select images using active learning")
    add_common_seed_args(parser)
    add_crid_seed_args(parser)
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    yaml_dict = load_yaml(args.config)
    project = args.project if args.project is not _UNSET else None
    yaml_dict = resolve_project(yaml_dict, project)
    merged = merge_cli(yaml_dict, args, SEED_CLI_MAP)
    merged = handle_model_path_override(merged, args)
    try:
        cfg = build_seed_config(merged)
    except ConfigError as exc:
        parser.error(str(exc))

    logger.info(
        f"Selecting {cfg.selection.n_select} images (strategy={cfg.selection.strategy})...",
    )
    logger.info("%s", describe_tensorflow_device())
    start = time.time()
    crid = CRID(globals())
    provider_source = CridImageProviderSource(
        crid,
        Path(cfg.query.cache_root) / "downloads",
    )
    image_provider = build_image_provider(
        provider_source,
        cfg.query.cache_root,
    )
    result = select_samples(cfg, image_provider)
    logger.info(
        "Selected %d in %.1fs",
        len(result.selected_ids),
        time.time() - start,
    )
    persist_outputs(cfg, result, crid, image_provider)


if __name__ == "__main__":
    main()
