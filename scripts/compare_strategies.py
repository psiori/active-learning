"""Run selection strategies and generate comparison mosaics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from active_learning.core.config import (
    _UNSET,
    ConfigError,
    SEED_CLI_MAP,
    brightness_filter_inactive,
    build_seed_config,
    handle_model_path_override,
    load_yaml,
    merge_cli,
    resolve_project,
)
from active_learning.integrations.crid.provider_source import CridImageProviderSource
from active_learning.integrations.crid.source import CridSource
from active_learning.core import (
    apply_brightness_predicate,
    build_image_provider,
    run_alges_selection,
    run_coreset_selection,
    run_uncertainty_selection,
)
from active_learning.core.time_gap import filter_ids_by_min_milliseconds_between_images
from active_learning.providers import (
    create_penultimate_model,
    enable_mc_dropout,
    load_unet,
)
from active_learning.sinks.mosaic import render_mosaic
from interface.crid import CRID
from interface.model.models import SensorName


for key, value in list(os.environ.items()):
    if key.startswith("ENV_"):
        os.environ[key[4:]] = value


ALL_STRATEGIES = [
    "coreset",
    "uncertainty_coreset",
    "uncertainty_topk",
    "uncertainty_topk_coreset",
    "alges",
    "alges_coreset",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare active learning strategies with mosaics",
    )
    U = _UNSET
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-p", "--project", default=U)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=ALL_STRATEGIES,
        choices=ALL_STRATEGIES,
    )
    parser.add_argument("-o", "--output-dir", default="/crid/jupyterhub/robin")
    parser.add_argument("-n", "--n-select", type=int, default=U)
    parser.add_argument("--rng-seed", type=int, default=U)
    parser.add_argument("-s", "--sensor", default=U)
    parser.add_argument("--cache-root", default=U)
    parser.add_argument("--start", default=U)
    parser.add_argument("--end", default=U)
    parser.add_argument("--sama-project-id", type=int, default=U)
    parser.add_argument("--no-sama", action="store_true", default=U)
    parser.add_argument("--model-path", default=U)
    parser.add_argument("--model-name", default=U)
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
    )
    parser.add_argument("--exclude-seeded", default=U)
    args = parser.parse_args()

    no_sama = getattr(args, "no_sama", U)
    args.exclude_seeded = False if no_sama is not U and no_sama else U
    yaml_dict = load_yaml(args.config)
    project = args.project if args.project is not _UNSET else None
    yaml_dict = resolve_project(yaml_dict, project)
    yaml_dict = handle_model_path_override(yaml_dict, args)
    args_for_merge = argparse.Namespace(**vars(args))
    args_for_merge.strategy = args.strategies[0]
    merged = merge_cli(yaml_dict, args_for_merge, SEED_CLI_MAP)
    merged.setdefault("selection", {})["strategy"] = "coreset"
    try:
        cfg = build_seed_config(merged)
    except ConfigError as exc:
        parser.error(str(exc))

    crid = CRID(globals())
    source = CridSource(crid)
    provider_source = CridImageProviderSource(
        crid,
        Path(cfg.query.cache_root) / "downloads",
    )
    image_provider = build_image_provider(provider_source, cfg.query.cache_root)
    queried = source.query_pool_and_labeled_ids(
        sensor_name=getattr(SensorName, cfg.query.sensor),
        sama_project_id=cfg.export.sama_project_id,
        exclude_seeded=cfg.query.exclude_seeded,
    )
    pool_ids = queried.pool_ids
    if cfg.selection.min_milliseconds_between_images > 0:
        n_before = len(pool_ids)
        pool_ids = filter_ids_by_min_milliseconds_between_images(
            pool_ids,
            cfg.selection.min_milliseconds_between_images,
        )
        n_after = len(pool_ids)
        print(
            f"Time spacing on pool (min {cfg.selection.min_milliseconds_between_images} ms): "
            f"{n_after}/{n_before} before brightness.",
            flush=True,
        )
        if n_after == 0:
            parser.error(
                "min_milliseconds_between_images left no pool candidates; "
                "lower the value or widen the window.",
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for strategy in args.strategies:
        if strategy == "coreset":
            result = run_coreset_selection(
                candidate_ids,
                seed_ids=queried.labeled_ids,
                image_provider=image_provider,
                cache_root=cfg.query.cache_root,
                n=cfg.selection.n_select,
                feature_model=cfg.coreset.feature_model,
                seed=cfg.selection.seed,
            )
        elif strategy in (
            "uncertainty_coreset",
            "uncertainty_topk",
            "uncertainty_topk_coreset",
        ):
            uc = cfg.uncertainty_coreset
            model_def = cfg.models[uc.uncertainty_model]
            unet = load_unet(model_def.path)
            uncertainty_model = (
                enable_mc_dropout(unet)
                if uc.provider in {"mc_dropout", "bald"}
                else unet
            )
            result = run_uncertainty_selection(
                candidate_ids,
                seed_ids=queried.labeled_ids,
                image_provider=image_provider,
                cache_root=cfg.query.cache_root,
                n=cfg.selection.n_select,
                uncertainty_kind=uc.provider,
                uncertainty_model=uncertainty_model,
                feature_model=uc.feature_model,
                model_name=model_def.name,
                model_path=model_def.path,
                alpha=uc.alpha,
                candidate_multiplier=uc.candidate_multiplier,
                iterations=uc.mc_iterations,
                image_size=model_def.image_size,
                batch_size=uc.batch_size,
                aggregation=uc.aggregation,
                topk_fraction=uc.topk_fraction,
                strategy=strategy,
                seed=cfg.selection.seed,
            )
        else:
            al = cfg.alges
            model_def = cfg.models[al.model]
            unet = load_unet(model_def.path)
            penultimate_model = create_penultimate_model(enable_mc_dropout(unet))
            if strategy == "alges":
                result = run_alges_selection(
                    candidate_ids,
                    image_provider=image_provider,
                    cache_root=cfg.query.cache_root,
                    model=penultimate_model,
                    model_name=model_def.name,
                    model_path=model_def.path,
                    n=cfg.selection.n_select,
                    method=al.method,
                    image_size=model_def.image_size,
                    batch_size=al.batch_size,
                    seed=cfg.selection.seed,
                )
            else:
                stage1_n = min(
                    len(candidate_ids),
                    max(cfg.selection.n_select, cfg.selection.n_select * 4),
                )
                stage1 = run_alges_selection(
                    candidate_ids,
                    image_provider=image_provider,
                    cache_root=cfg.query.cache_root,
                    model=penultimate_model,
                    model_name=model_def.name,
                    model_path=model_def.path,
                    n=stage1_n,
                    method=al.method,
                    image_size=model_def.image_size,
                    batch_size=al.batch_size,
                    seed=cfg.selection.seed,
                )
                result = run_coreset_selection(
                    stage1.selected_ids,
                    seed_ids=queried.labeled_ids,
                    image_provider=image_provider,
                    cache_root=cfg.query.cache_root,
                    n=cfg.selection.n_select,
                    feature_model=cfg.coreset.feature_model,
                    seed=cfg.selection.seed,
                )

        output_path = output_dir / f"compare_{strategy}.jpg"
        render_mosaic(
            result,
            image_provider,
            output_path,
            resize_height=150,
            max_images=200,
            rows=10,
            cols=20,
        )
        print(f"{strategy}: saved {output_path}")


if __name__ == "__main__":
    main()
