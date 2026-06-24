"""Shared CLI plumbing for active-learning entrypoints."""

from __future__ import annotations

import argparse

from active_learning.core.config import (
    _UNSET,
    VALID_ALGES_METHODS,
    VALID_PROVIDERS,
    VALID_UNCERTAINTY_AGGREGATIONS,
    deep_merge,
    parse_boolish,
)


COMMON_CLI_MAP = {
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

CRID_CLI_MAP = {
    "max_size": ("query", "max_size"),
    "sensor": ("query", "sensor"),
    "sama_project_id": ("export", "sama_project_id"),
    "sama_priority": ("query", "sama_priority"),
    "exclude_seeded": ("query", "exclude_seeded"),
    "export_prefix": ("export", "prefix"),
    "export_sama": ("export", "seed"),
    "overlay": ("export", "overlay"),
}

LOCAL_CLI_MAP = dict(COMMON_CLI_MAP)
SEED_CLI_MAP = {**COMMON_CLI_MAP, **CRID_CLI_MAP}

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


def add_common_seed_args(parser: argparse.ArgumentParser) -> None:
    U = _UNSET
    parser.add_argument("-c", "--config", default=None, help="Path to YAML config file")
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
        help="Before brightness/selection, thin the pool by capture time in sample IDs.",
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


def add_crid_seed_args(parser: argparse.ArgumentParser) -> None:
    U = _UNSET
    parser.add_argument("-p", "--project", default=U, help="Project preset name")
    parser.add_argument(
        "--max-size",
        type=int,
        default=U,
        help="Maximum number of ES documents to fetch (default 100000).",
    )
    parser.add_argument("-s", "--sensor", default=U)
    parser.add_argument("--sama-project-id", type=int, default=U)
    parser.add_argument("--sama-priority", type=int, default=U)
    parser.add_argument("--exclude-seeded", default=U)
    parser.add_argument("--export-prefix", default=U)
    parser.add_argument(
        "--export-sama",
        dest="export_sama",
        action="store_true",
        default=U,
        help="If provided, submit the selected images to Sama/CRID.",
    )
    parser.add_argument("--overlay", action="store_true", default=U)


def apply_local_defaults(yaml_dict: dict) -> dict:
    merged = deep_merge(LOCAL_DEFAULTS, yaml_dict)
    merged.setdefault("query", {})
    merged["query"]["exclude_seeded"] = False
    merged["query"]["exclude_al_excluded"] = False
    merged.setdefault("export", {})
    merged["export"]["seed"] = False
    return merged


def prepare_strategy_models(cfg):
    """Load optional strategy models needed by run_selection."""
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

    return uncertainty_model, alges_model
