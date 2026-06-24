"""Select images from a local directory using active-learning strategies."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

from active_learning.core import (
    apply_brightness_predicate,
    build_image_provider,
    run_selection,
)
from active_learning.core.config import (
    _UNSET,
    ConfigError,
    VALID_ALGES_METHODS,
    VALID_PROVIDERS,
    VALID_UNCERTAINTY_AGGREGATIONS,
    build_seed_config,
    brightness_filter_inactive,
    handle_model_path_override,
    load_yaml,
    merge_cli,
    parse_boolish,
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

LOCAL_DEFAULTS = {
    "query": {
        "cache_root": "data/local_active_learning",
        "exclude_seeded": False,
        "exclude_al_excluded": False,
    },
    "selection": {
        "strategy": "coreset",
    },
    "export": {
        "mosaic_path": "data/local_active_learning/mosaic.jpg",
        "seed": False,
    },
}

LOCAL_CLI_MAP = {
    "strategy": ("selection", "strategy"),
    "n_select": ("selection", "n_select"),
    "rng_seed": ("selection", "seed"),
    "min_milliseconds_between_images": (
        "query",
        "min_milliseconds_between_images",
    ),
    "cache_root": ("query", "cache_root"),
    "min_brightness": ("query", "min_brightness"),
    "max_brightness": ("query", "max_brightness"),
    "brightness_filter_enabled": ("query", "brightness_filter_enabled"),
    "start": ("query", "start"),
    "end": ("query", "end"),
    "use_full_res_images": ("query", "use_full_res_images"),
    "feature_model": ("coreset", "feature_model"),
    "alpha": ("uncertainty_coreset", "alpha"),
    "provider": ("uncertainty_coreset", "provider"),
    "mc_iterations": ("uncertainty_coreset", "mc_iterations"),
    "batch_size": ("uncertainty_coreset", "batch_size"),
    "aggregation": ("uncertainty_coreset", "aggregation"),
    "topk_fraction": ("uncertainty_coreset", "topk_fraction"),
    "candidate_multiplier": ("uncertainty_coreset", "candidate_multiplier"),
    "method": ("alges", "method"),
    "mosaic_path": ("export", "mosaic_path"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select local images using active learning",
    )
    U = _UNSET
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory scanned recursively for local images.",
    )
    parser.add_argument("-c", "--config", default=None, help="Path to YAML config file")
    parser.add_argument("--strategy", default=U)
    parser.add_argument("-n", "--n-select", type=int, default=U)
    parser.add_argument("--rng-seed", type=int, default=U)
    parser.add_argument(
        "--min-milliseconds-between-images",
        type=float,
        default=U,
        metavar="MS",
    )
    parser.add_argument("--cache-root", default=U)
    parser.add_argument("--start", default=U)
    parser.add_argument("--end", default=U)
    parser.add_argument("--min-brightness", type=float, default=U)
    parser.add_argument("--max-brightness", type=float, default=U)
    parser.add_argument(
        "--brightness-filter-enabled",
        type=parse_boolish,
        default=U,
    )
    parser.add_argument("--use-full-res-images", action="store_true", default=U)
    parser.add_argument("--feature-model", default=U)
    parser.add_argument("-a", "--alpha", type=float, default=U)
    parser.add_argument("--provider", default=U, choices=VALID_PROVIDERS)
    parser.add_argument("--mc-iterations", type=int, default=U)
    parser.add_argument("--batch-size", type=int, default=U)
    parser.add_argument(
        "--aggregation",
        default=U,
        choices=VALID_UNCERTAINTY_AGGREGATIONS,
    )
    parser.add_argument("--topk-fraction", type=float, default=U)
    parser.add_argument("--candidate-multiplier", type=int, default=U)
    parser.add_argument("--method", default=U, choices=VALID_ALGES_METHODS)
    parser.add_argument("--mosaic-path", default=U)
    parser.add_argument("--model-path", default=U)
    parser.add_argument("--model-name", default=U)
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        yaml_dict = _with_local_defaults(load_yaml(args.config))
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
    strategy = cfg.selection.strategy
    uncertainty_model = None
    alges_model = None

    if strategy in (
        "uncertainty_coreset",
        "uncertainty_topk",
        "uncertainty_topk_coreset",
    ):
        from active_learning.providers import enable_mc_dropout, load_unet

        uc = cfg.uncertainty_coreset
        model_def = cfg.models[uc.uncertainty_model]
        unet = load_unet(model_def.path)
        uncertainty_model = (
            enable_mc_dropout(unet) if uc.provider in {"mc_dropout", "bald"} else unet
        )

    if strategy in ("alges", "alges_coreset"):
        from active_learning.providers import (
            create_penultimate_model,
            enable_mc_dropout,
            load_unet,
        )

        al = cfg.alges
        model_def = cfg.models[al.model]
        al_unet = load_unet(model_def.path)
        alges_model = create_penultimate_model(enable_mc_dropout(al_unet))

    return run_selection(
        cfg,
        candidate_ids=candidate_ids,
        seed_ids=[],
        image_provider=image_provider,
        uncertainty_model=uncertainty_model,
        alges_model=alges_model,
        progress=CLI_PROGRESS,
    )


def _with_local_defaults(yaml_dict: dict) -> dict:
    merged = _deep_merge_dict(LOCAL_DEFAULTS, yaml_dict)
    merged.setdefault("query", {})
    merged["query"]["exclude_seeded"] = False
    merged["query"]["exclude_al_excluded"] = False
    merged.setdefault("export", {})
    merged["export"]["seed"] = False
    return merged


def _deep_merge_dict(base: dict, overlay: dict) -> dict:
    result = deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
