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

from active_learning.core import (
    apply_brightness_predicate,
    build_image_provider,
    run_selection,
)
from active_learning.core.config import (
    _UNSET,
    SEED_CLI_MAP,
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
from active_learning.providers import (
    create_penultimate_model,
    enable_mc_dropout,
    load_unet,
)
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

    strategy = cfg.selection.strategy
    uncertainty_model = None
    alges_model = None

    if strategy in (
        "uncertainty_coreset",
        "uncertainty_topk",
        "uncertainty_topk_coreset",
    ):
        uc = cfg.uncertainty_coreset
        uc_model_def = cfg.models[uc.uncertainty_model]
        unet = load_unet(uc_model_def.path)
        uncertainty_model = (
            enable_mc_dropout(unet) if uc.provider in {"mc_dropout", "bald"} else unet
        )

    if strategy in ("alges", "alges_coreset"):
        al = cfg.alges
        al_model_def = cfg.models[al.model]
        al_unet = load_unet(al_model_def.path)
        alges_model = create_penultimate_model(enable_mc_dropout(al_unet))

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
    U = _UNSET
    parser.add_argument("-c", "--config", default=None, help="Path to YAML config file")
    parser.add_argument("-p", "--project", default=U, help="Project preset name")
    parser.add_argument("--strategy", default=U)
    parser.add_argument("-n", "--n-select", type=int, default=U)
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=U,
        help="Random seed for the active-learning selector.",
    )
    parser.add_argument(
        "--min-milliseconds-between-images",
        type=float,
        default=U,
        metavar="MS",
        help="Before brightness/selection, thin the pool by capture time in sample IDs "
        "(0 disables). Avoids downloading images that will not enter the pool.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=U,
        help="Maximum number of ES documents to fetch (default 100000).",
    )
    parser.add_argument("-s", "--sensor", default=U)
    parser.add_argument("--cache-root", default=U)
    parser.add_argument("--start", default=U)
    parser.add_argument("--end", default=U)
    parser.add_argument("--sama-project-id", type=int, default=U)
    parser.add_argument("--sama-priority", type=int, default=U)
    parser.add_argument("--exclude-seeded", default=U)
    parser.add_argument("--min-brightness", type=float, default=U)
    parser.add_argument("--max-brightness", type=float, default=U)
    parser.add_argument("--use-full-res-images", action="store_true", default=U)
    parser.add_argument("--feature-model", default=U)
    parser.add_argument("-a", "--alpha", type=float, default=U)
    parser.add_argument(
        "--provider",
        default=U,
        choices=["mc_dropout", "entropy", "bald"],
    )
    parser.add_argument("--mc-iterations", type=int, default=U)
    parser.add_argument("--batch-size", type=int, default=U)
    parser.add_argument(
        "--aggregation",
        default=U,
        choices=["mean", "topk_mean", "max"],
    )
    parser.add_argument("--topk-fraction", type=float, default=U)
    parser.add_argument("--candidate-multiplier", type=int, default=U)
    parser.add_argument("--method", default=U, choices=["image", "semantic"])
    parser.add_argument("--export-prefix", default=U)
    parser.add_argument("--mosaic-path", default=U)
    parser.add_argument(
        "--export-sama",
        dest="export_sama",
        action="store_true",
        default=U,
        help="If provided, submit the selected images to Sama/CRID.",
    )
    parser.add_argument("--overlay", action="store_true", default=U)
    parser.add_argument("--model-path", default=U)
    parser.add_argument("--model-name", default=U)
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    yaml_dict = load_yaml(args.config)
    project = args.project if args.project is not _UNSET else None
    yaml_dict = resolve_project(yaml_dict, project)
    yaml_dict = handle_model_path_override(yaml_dict, args)
    merged = merge_cli(yaml_dict, args, SEED_CLI_MAP)
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
