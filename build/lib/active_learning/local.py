"""Select images from a local directory using active-learning strategies."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from active_learning.cli.common import (
    LOCAL_CLI_MAP,
    add_common_seed_args,
    apply_local_defaults,
    prepare_strategy_models,
)
from active_learning.core import (
    apply_brightness_predicate,
    build_image_provider,
    run_selection,
)
from active_learning.core.config import (
    ConfigError,
    build_seed_config,
    brightness_filter_inactive,
    handle_model_path_override,
    load_yaml,
    merge_cli,
)
from active_learning.core.logger import configure_logging
from active_learning.core.logger_tf import (
    describe_tensorflow_device,
    ensure_tensorflow_log_suppression,
)
from active_learning.core.time_gap import filter_ids_by_min_milliseconds_between_images
from active_learning.integrations.local import LocalImageProviderSource
from active_learning.sinks.mosaic import mosaic_output_path, render_mosaic
from active_learning.sinks.yaml import write_local_selection_payload

ensure_tensorflow_log_suppression()

logger = logging.getLogger("active_learning.local")
CLI_PROGRESS = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select local images using active learning",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory scanned recursively for local images.",
    )
    add_common_seed_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        yaml_dict = apply_local_defaults(load_yaml(args.config))
        merged = merge_cli(yaml_dict, args, LOCAL_CLI_MAP)
        merged = handle_model_path_override(merged, args)
        cfg = build_seed_config(merged, require_sensor=False)
        source = LocalImageProviderSource.from_directory(args.images_dir)
    except (ConfigError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    logger.info(
        "Selecting %d local images from %s (strategy=%s)...",
        cfg.selection.n_select,
        Path(args.images_dir).expanduser().resolve(),
        cfg.selection.strategy,
    )
    logger.info("%s", describe_tensorflow_device())
    start = time.time()
    image_provider = build_image_provider(source, cfg.query.cache_root)
    candidate_ids = list(source.sample_ids)

    if cfg.query.min_milliseconds_between_images > 0:
        before = len(candidate_ids)
        candidate_ids = filter_ids_by_min_milliseconds_between_images(
            candidate_ids,
            cfg.query.min_milliseconds_between_images,
        )
        logger.info(
            "Time spacing kept %d/%d local images.",
            len(candidate_ids),
            before,
        )

    if brightness_filter_inactive(cfg.query):
        filtered_ids = candidate_ids
    else:
        filtered_ids, _ = apply_brightness_predicate(
            candidate_ids,
            image_provider,
            cfg.query.cache_root,
            min_brightness=cfg.query.min_brightness,
            max_brightness=cfg.query.max_brightness,
            progress=CLI_PROGRESS,
        )

    if not filtered_ids:
        raise SystemExit("No local images remain after filtering.")
    if cfg.selection.n_select > len(filtered_ids):
        raise SystemExit(
            f"n_select={cfg.selection.n_select} exceeds available local candidates "
            f"after filtering ({len(filtered_ids)}).",
        )

    result = select_local_samples(cfg, image_provider, filtered_ids)
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
    yaml_path = write_local_selection_payload(
        result,
        cfg,
        mosaic_path=mosaic_path,
        images_dir=args.images_dir,
    )
    logger.info(
        "Selected %d in %.1fs",
        len(result.selected_ids),
        time.time() - start,
    )
    logger.info("Mosaic saved to %s", mosaic_path)
    logger.info("YAML handoff saved to %s", yaml_path)


def select_local_samples(cfg, image_provider, candidate_ids: list[str]):
    uncertainty_model, alges_model = prepare_strategy_models(cfg)

    return run_selection(
        cfg,
        candidate_ids=candidate_ids,
        seed_ids=[],
        image_provider=image_provider,
        uncertainty_model=uncertainty_model,
        alges_model=alges_model,
        progress=CLI_PROGRESS,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
